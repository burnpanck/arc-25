import dataclasses
import functools
import inspect
import types
import typing
from types import SimpleNamespace
from typing import Literal

import attrs
import click
import json5


def _get_config_class(type_hint) -> type | None:
    """Check if a type is an attrs class or dataclass."""
    origin = typing.get_origin(type_hint)
    if origin in {types.UnionType, typing.Union, typing.Optional}:
        # we can only handle effective "optinals"
        alternatives = typing.get_args(type_hint)
        non_none = [t for t in alternatives if t is not type(None)]
        if len(non_none) != 1:
            raise ValueError(f"Cannot handle union {type_hint}")
        (inner,) = non_none
        type_hint = inner

    if isinstance(type_hint, type) and (
        attrs.has(type_hint) or dataclasses.is_dataclass(type_hint)
    ):
        return type_hint
    return None


def _get_fields(cls):
    """Get fields from attrs class or dataclass."""
    if attrs.has(cls):
        return attrs.fields(cls)
    elif dataclasses.is_dataclass(cls):
        return dataclasses.fields(cls)
    return []


def _get_field_info(field):
    """Extract field info uniformly from attrs or dataclass field."""
    if isinstance(field, attrs.Attribute):
        return SimpleNamespace(
            name=field.name,
            type=field.type,
            default=field.default,
            metadata=field.metadata,
            has_default=(field.default != attrs.NOTHING),
        )
    else:  # dataclass field
        return SimpleNamespace(
            name=field.name,
            type=field.type,
            default=field.default if field.default != dataclasses.MISSING else None,
            metadata=field.metadata,
            has_default=(
                field.default != dataclasses.MISSING
                or field.default_factory != dataclasses.MISSING
            ),
        )


def _flatten_config_fields(cls, prefix_path=()):
    """
    Recursively flatten hierarchical config class.
    Returns list of (path, field_info) tuples where path is a list of field names.
    """

    flattened = []

    for field in _get_fields(cls):
        info = _get_field_info(field)
        field_path = prefix_path + (info.name,)

        # Check if field is a nested config class
        if nested := _get_config_class(info.type):
            # Recurse into nested config
            flattened.extend(_flatten_config_fields(nested, field_path))
        else:
            flattened.append((field_path, info))

    return flattened


def unflatten_config(flat_dict):
    """
    Convert both flat {'model.config': 'small'} and nested {'model': {'config': 'small'}}
    formats into a consistent nested structure.
    """
    result = {}

    for key, value in flat_dict.items():
        if "." in key:
            # Flat format: split on dots
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            # Could be top-level key or already nested dict
            if isinstance(value, dict):
                # Nested format: preserve structure
                result[key] = value
            else:
                # Top-level scalar
                result[key] = value

    return result


def attrs_to_click_options(func):
    """
    Decorator that automatically generates Click options from attrs/dataclass fields.
    Supports hierarchical configs with hyphenated names (e.g., --model-config).
    DRY approach - single source of truth for configuration parameters.
    """
    (args_cls,) = inspect.get_annotations(func).values()
    assert _get_config_class(args_cls)

    # Flatten hierarchical structure
    flat_fields = _flatten_config_fields(args_cls)

    # Build mapping from option name to field path
    option_to_path = {}

    # Create wrapper that will receive Click arguments
    @functools.wraps(func)
    def wrapper(config_json, **kwargs):
        # Convert CLI kwargs to nested dict using the captured mapping
        cli_dict = kwargs_to_nested_dict(kwargs, option_to_path)

        # Start with config from JSON if provided
        config_dict = {}
        if config_json:
            raw_json = json5.loads(config_json)
            config_dict = unflatten_config(raw_json)

        # Merge CLI args into config_dict (CLI args override JSON)
        def merge_dicts(base, override):
            """Recursively merge override into base."""
            for key, value in override.items():
                if isinstance((bv := base.get(key)), dict) and isinstance(value, dict):
                    merge_dicts(bv, value)
                else:
                    base[key] = value

        merge_dicts(config_dict, cli_dict)

        # Reconstruct the hierarchical config object
        args = reconstruct_hierarchical_config(args_cls, config_dict)

        return func(args)

    # Apply Click decorators to the wrapper (in reverse order for Click)
    # Start with JSON config option
    wrapper = click.option(
        "--config-json",
        type=str,
        help="JSON string with configuration (supports both flat 'a.b.c' and nested {'a':{'b':{'c':...}}} formats)",
    )(wrapper)

    # Introspect fields in reverse order (Click applies decorators bottom-up)
    for field_path, info in reversed(flat_fields):
        # Convert path to hyphenated option name
        # e.g., ['model', 'config'] -> 'model-config'
        option_name_hyphenated = "-".join(field_path).replace("_", "-")
        full_option_name = f"--{option_name_hyphenated}"

        # Store reverse mapping with underscores (Click converts hyphens to underscores in kwargs)
        option_name_underscored = option_name_hyphenated.replace("-", "_")
        option_to_path[option_name_underscored] = field_path

        # Determine Click type from field type annotation
        field_type = info.type
        path_str = ".".join(field_path)
        click_kwargs = dict(
            help=info.metadata.get("help", f"{path_str} parameter"),
        )

        # Handle Literal types (enum-like)
        origin = typing.get_origin(field_type)
        if origin in {types.UnionType, typing.Union, typing.Optional}:
            # we can only handle effective "optinals"
            alternatives = typing.get_args(field_type)
            non_none = [t for t in alternatives if t is not type(None)]
            if len(non_none) != 1:
                raise ValueError(f"Cannot handle union {field_type}")
            (inner,) = non_none
            field_type = inner
            origin = typing.get_origin(field_type)
        if origin is Literal:
            choices = typing.get_args(field_type)
            click_type = click.Choice([str(c) for c in choices])
        # Handle list/set/frozenset types
        elif origin is tuple:
            args = typing.get_args(field_type)
            inner_type = args[0]
            assert all(t == inner_type for t in args)
            click_type = inner_type
            click_kwargs["nargs"] = len(args)
        elif origin in {list, set, frozenset}:
            args = typing.get_args(field_type)
            if len(args) != 1:
                raise ValueError(f"Cannot handle {field_type}")
            (inner_type,) = args
            inner_origin = typing.get_origin(inner_type)
            if inner_origin is Literal:
                conv = str
            else:
                conv = inner_type
            assert conv in {
                int,
                str,
            }, f"{inner_type!r} is not currently supported in variable length arguments"
            click_kwargs["callback"] = (
                lambda ctx, param, v, *, conv=conv, origin=origin, inner_type=inner_type: (
                    None
                    if v is None
                    else origin(
                        [conv(n.strip()) for n in v.split(",")] if v.strip() else []
                    )
                )
            )
            click_type = None
        # Handle basic types
        elif field_type in {int, float, str, bool}:
            click_type = field_type
        else:
            raise TypeError(
                f"Unsupported annotation origin {origin!r} for field {field_path} ({field_type!r})"
            )
        click_kwargs["type"] = click_type
        del click_type

        # Apply Click option decorator to wrapper
        wrapper = click.option(full_option_name, **click_kwargs)(wrapper)

    return wrapper


def kwargs_to_nested_dict(kwargs, option_to_path_mapping):
    """
    Convert Click kwargs to nested dict using the option-to-path mapping.

    Args:
        kwargs: Dict from Click with hyphenated option names
        option_to_path_mapping: Dict mapping option names to field paths

    Returns:
        Nested dict suitable for reconstructing config
    """
    result = {}

    for option_name, value in kwargs.items():
        if value is None:
            continue

        # Look up the path for this option
        field_path = option_to_path_mapping.get(option_name)
        if field_path is None:
            raise KeyError(
                f"Unknown option {option_name}; known are: {', '.join(sorted(option_to_path_mapping))}"
            )

        # Navigate to the right place in the nested dict
        current = result
        for part in field_path[:-1]:
            current = current.setdefault(part, {})
        current[field_path[-1]] = value

    return result


def reconstruct_hierarchical_config(cls, config_dict):
    """
    Recursively reconstruct hierarchical config from nested dict.
    Handles both attrs and dataclasses.
    """
    kwargs = {}

    for field in _get_fields(cls):
        info = _get_field_info(field)
        field_name = info.name

        if field_name not in config_dict:
            continue

        value = config_dict[field_name]

        # If field is a nested config class, recurse
        if nested := _get_config_class(info.type):
            if isinstance(value, dict):
                kwargs[field_name] = reconstruct_hierarchical_config(nested, value)
            else:
                # Value is already an instance (shouldn't happen with CLI, but handle it)
                kwargs[field_name] = value
        else:
            # Handle collection types that need conversion
            origin = typing.get_origin(info.type)
            if origin in {frozenset, set}:
                kwargs[field_name] = (
                    origin(value) if not isinstance(value, origin) else value
                )
            elif origin is list and not isinstance(value, list):
                kwargs[field_name] = list(value)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)
