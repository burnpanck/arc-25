import dataclasses
import typing
from typing import Literal

import attrs
import click


def _is_config_class(type_hint):
    """Check if a type is an attrs class or dataclass."""
    if isinstance(type_hint, type):
        return attrs.has(type_hint) or dataclasses.is_dataclass(type_hint)
    return False


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
        return dict(
            name=field.name,
            type=field.type,
            default=field.default,
            metadata=field.metadata,
            has_default=(field.default != attrs.NOTHING),
        )
    else:  # dataclass field
        return dict(
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
        field_path = prefix_path + (info["name"],)

        # Check if field is a nested config class
        if _is_config_class(info["type"]):
            # Recurse into nested config
            flattened.extend(_flatten_config_fields(info["type"], field_path))
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


def attrs_to_click_options(attrs_cls):
    """
    Decorator that automatically generates Click options from attrs/dataclass fields.
    Supports hierarchical configs with hyphenated names (e.g., --model-config).
    DRY approach - single source of truth for configuration parameters.
    """

    def decorator(func):
        # Flatten hierarchical structure
        flat_fields = _flatten_config_fields(attrs_cls)

        # Build mapping from option name to field path
        option_to_path = {}

        # Create wrapper that will receive Click arguments
        def wrapper(config_json, **kwargs):
            # Convert CLI kwargs to nested dict using the captured mapping
            cli_dict = kwargs_to_nested_dict(kwargs, option_to_path)

            # Call the actual function with processed args
            return func(config_json=config_json, cli_dict=cli_dict)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

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
            field_type = info["type"]
            path_str = ".".join(field_path)
            click_kwargs = dict(
                help=info["metadata"].get("help", f"{path_str} parameter")
            )

            # Handle Literal types (enum-like)
            origin = typing.get_origin(field_type)
            if origin is Literal:
                choices = typing.get_args(field_type)
                click_kwargs["type"] = click.Choice([str(c) for c in choices])
            # Handle list/set/frozenset types
            elif origin in {list, set, frozenset}:
                args = typing.get_args(field_type)
                inner_type = args[0] if args else str
                click_kwargs["multiple"] = True
                click_kwargs["type"] = inner_type
            # Handle basic types
            elif field_type in {int, float, str, bool}:
                click_kwargs["type"] = field_type
            else:
                # Default to string for complex types
                click_kwargs["type"] = str

            # Set default value
            if isinstance(info["default"], (attrs.Factory, dataclasses.Field)):
                # Handle factory defaults
                if isinstance(info["default"], attrs.Factory):
                    if not info["default"].takes_self:
                        click_kwargs["default"] = info["default"].factory()
                elif isinstance(info["default"], dataclasses.Field):
                    if info["default"].default_factory != dataclasses.MISSING:
                        click_kwargs["default"] = info["default"].default_factory()
            elif info["has_default"] and info["default"] is not None:
                click_kwargs["default"] = info["default"]
            elif not info["has_default"]:
                click_kwargs["required"] = True

            # Apply Click option decorator to wrapper
            wrapper = click.option(full_option_name, **click_kwargs)(wrapper)

        return wrapper

    return decorator


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
        field_name = info["name"]

        if field_name not in config_dict:
            continue

        value = config_dict[field_name]

        # If field is a nested config class, recurse
        if _is_config_class(info["type"]):
            if isinstance(value, dict):
                kwargs[field_name] = reconstruct_hierarchical_config(
                    info["type"], value
                )
            else:
                # Value is already an instance (shouldn't happen with CLI, but handle it)
                kwargs[field_name] = value
        else:
            # Handle collection types that need conversion
            origin = typing.get_origin(info["type"])
            if origin in {frozenset, set}:
                kwargs[field_name] = (
                    origin(value) if not isinstance(value, origin) else value
                )
            elif origin is list and not isinstance(value, list):
                kwargs[field_name] = list(value)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)
