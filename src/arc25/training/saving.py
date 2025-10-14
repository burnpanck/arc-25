import lzma
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jaxlib
import msgpack
import numpy as np
import tqdm.auto
from flax import nnx

from ..serialisation import deserialise, serialise


def save_model(model: nnx.Module, path: Path, *, metadata: dict | None = None):
    with lzma.open(path, "wb") as fh:

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


def load_model(path: Path) -> SimpleNamespace:
    metadata = {}
    state = {}
    graphdef = None

    with lzma.open(path, "rb") as fh:
        path = ()

        for data in msgpack.Unpacker(fh, raw=False):
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
