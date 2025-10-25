import contextlib
import io
import lzma
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import etils.epath
import jax
import jaxlib
import msgpack
import numpy as np
import tqdm.auto
from flax import nnx

from ..serialisation import deserialise, serialise


def save_model(
    model: nnx.Module,
    path_or_fh: etils.epath.PathLike | BinaryIO,
    *,
    metadata: dict | None = None,
):
    """Save model to a file path or file-like object.

    Args:
        model: The model to save
        path: Either a Path to write to, or a file-like object (e.g., io.BytesIO)
        metadata: Optional metadata to include in the checkpoint
    """
    with contextlib.ExitStack() as stack:
        if isinstance(path_or_fh, etils.epath.Path):
            path_or_fh = stack.enter_context(path_or_fh.open("wb"))
        fh = stack.enter_context(lzma.LZMAFile(path_or_fh, mode="wb"))

        def write(*data):
            serialised = msgpack.dumps(tuple(data))
            fh.write(serialised)

        if metadata is not None:
            write("M", serialise(metadata))

        graphdef, flatstate = nnx.graph.flatten(nnx.pure(nnx.state(model, nnx.Param)))
        write("G", pickle.dumps(graphdef))
        pointer = ()
        for path, s in flatstate:  # tqdm.auto.tqdm(flatstate):
            pstr = ".".join(str(v) for v in path)  # noqa: F841
            match s:
                case jaxlib._jax.ArrayImpl():
                    d = np.asarray(jax.device_get(s))
                case np.ndarray():
                    d = s
                case _:
                    raise TypeError(type(s).__qualname__)
            neq = 0
            for i, (a, b) in enumerate(zip(pointer, path)):
                neq = i
                if a != b:
                    break
            key = (neq, path[neq:])  # noqa: F841
            write("A", neq, path[neq:], d.shape, str(d.dtype), bytes(d.data))
            pointer = path


def load_model(path_or_fh: etils.epath.PathLike | BinaryIO) -> SimpleNamespace:
    metadata = {}
    state = {}
    graphdef = None

    with contextlib.ExitStack() as stack:
        if isinstance(path_or_fh, etils.epath.Path):
            path_or_fh = stack.enter_context(path_or_fh.open("rb"))
        fh = stack.enter_context(lzma.LZMAFile(path_or_fh, mode="rb"))

        path = ()

        for data in msgpack.Unpacker(fh, raw=False, strict_map_key=False):
            code, args = data[0], data[1:]

            match code:
                case "C" | "M":
                    (m,) = args
                    m = deserialise(m)
                    if code == "C":
                        m = dict(config=m)
                    metadata.update(m)
                case "G":
                    graphdef = pickle.loads(*args)
                case "A":
                    neq, path_ext, shape, dtype, data = args
                    path = path[:neq] + tuple(path_ext)
                    array = np.frombuffer(data, dtype=dtype).reshape(shape)
                    state[path] = array
                case _:
                    raise KeyError(f"Unknown code {code!r}")

    state = nnx.statelib.FlatState.from_sorted_keys_values(state.keys(), state.values())
    state = nnx.merge(graphdef, state)
    return SimpleNamespace(
        metadata=metadata, config=metadata.pop("config", None), state=state
    )
