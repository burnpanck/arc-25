import dataclasses
import enum
import functools
from types import MappingProxyType

import attrs
import numpy as np

try:
    import jax
except Exception:
    jax = None

from .dsl import types
from .symmetry import D4, PermRepBase


def serialise(obj):
    match obj:
        case dict() | MappingProxyType():
            return {k: serialise(v) for k, v in obj.items()}
        case tuple() | list():
            return type(obj)(serialise(v) for v in obj)
        case np.ndarray():
            return dict(
                __type__=type(obj).__qualname__,
                shape=obj.shape,
                dtype=str(obj.dtype),
                data=bytes(obj.ravel()),
            )
        case enum.Enum():
            return dict(__type__=type(obj).__qualname__, name=obj.name)
        case _ if attrs.has(type(obj)):
            dct = attrs.asdict(
                obj,
                recurse=False,
                filter=lambda attr, value: not attr.metadata.get("is_cache"),
            )
            dct["__type__"] = type(obj).__qualname__
            return serialise(dct)
        case np.dtype():
            return dict(__type__=type(obj).__qualname__, name=str(obj))
        case type():
            return serialise(dict(__type__=obj.__qualname__, __just_type__=True))
        case _ if dataclasses.is_dataclass(obj):
            dct = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
            dct["__type__"] = type(obj).__qualname__
            return serialise(dct)
        case _:
            return obj


_known_types = {
    t.__qualname__: t
    for t in [np.ndarray, D4, np.dtype] + list(PermRepBase._known_reps)
} | {
    t.__qualname__: t
    for k, t in vars(types).items()
    if isinstance(t, type) and not k.startswith("_")
}

_renamed_attrs = {}


def serialisable(t=None, *, renamed_attrs=None):
    if t is None:
        kw = {
            k: v
            for k, v in dict(
                renamed_attrs=renamed_attrs,
            ).items()
            if v is not None
        }
        return functools.partial(serialisable, **kw)
    assert (
        t.__qualname__ not in _known_types
        or t.__module__ == _known_types[t.__qualname__].__module__
    )
    _known_types[t.__qualname__] = t
    if renamed_attrs is not None:
        _renamed_attrs[t] = renamed_attrs
    return t


if jax is not None:
    import jax.numpy as jnp

    for dtype in [jnp.float16, jnp.bfloat16, jnp.float32]:
        serialisable(dtype)


def deserialise(data):
    match data:
        case dict():
            typ = data.pop("__type__", None)
            if typ is None:
                return {k: deserialise(v) for k, v in data.items()}
        case tuple() | list():
            return type(data)(deserialise(v) for v in data)
        case _:
            return data
    match typ:
        case "DType":
            name = data.pop("dtype")
            assert not data
            return np.dtype(name)
    cls = _known_types.get(typ)
    if cls is None:
        raise TypeError(f"Unsupported type {typ}")
    if data.pop("__just_type__", False):
        assert not data
        return cls
    if cls is np.ndarray:
        return np.frombuffer(data["data"], dtype=data["dtype"]).reshape(data["shape"])
    elif issubclass(cls, enum.Enum):
        return cls[data["name"]]
    else:
        assert attrs.has(cls) or dataclasses.is_dataclass(cls)
    a = deserialise(data)
    if (renamed_attrs := _renamed_attrs.get(cls)) is not None:
        a = {renamed_attrs.get(k, k): v for k, v in a.items()}
    return cls(**a)
