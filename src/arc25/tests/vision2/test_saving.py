import tempfile
from pathlib import Path
from types import SimpleNamespace

import flax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import serialisation, symmetry
from ...symmetry import D4
from ...training import saving
from ...vision2.classification import ARCClassifier
from ...vision2.fields import CoordinateGrid, Field, FieldDims
from ...vision2.symrep import RepSpec, SymDecompDims

flax_version = tuple(int(v) for v in flax.__version__.split("."))


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(SymDecompDims(1, 2, 3), id="SymDecompDims"),
        pytest.param(symmetry.FullRep, id="PermRepBase"),
        pytest.param(np.float32, id="float32"),
        pytest.param(jnp.bfloat16, id="bfloat16"),
        pytest.param(frozenset([1, 2, 3]), id="set"),
        pytest.param(
            SimpleNamespace(
                some_int=1,
                some_str="str",
                some_type=SimpleNamespace,
                some_path=Path("/temp"),
            ),
            id="SimpleNamespace",
        ),
    ],
)
def test_serialisation(example):
    serialised = serialisation.serialise(example, log_all_types=True)
    try:
        deserialised = serialisation.deserialise(serialised)
    except Exception as ex:
        raise RuntimeError(f"Failed to deserialise {serialised!r}") from ex
    assert example == deserialised


def test_saving():
    hidden_size = FieldDims(
        context=SymDecompDims(43, 11, 7),
        cells=SymDecompDims(41, 5, 3),
        context_tokens=3,
    )

    rngs = nnx.Rngs(42)
    arc_cls = ARCClassifier(
        num_layers=1,
        num_heads=4,
        num_groups=2,
        hidden_size=hidden_size,
        num_perceiver_tokens=7,
        num_perceiver_layers=1,
        # qk_head_width only applies to context path
        # Use TrivialRep for space as axial attention already has directional info
        qk_head_width=SymDecompDims(
            invariant=12,
            space=4,
            flavour=4,
            rep=RepSpec(symmetry.TrivialRep, 10),
        ),
        v_head_width=SymDecompDims(13, 5, 3),
        dtype=np.float32,
        use_chirality_rep=True,
        dropout_rate=0.0,
        rngs=rngs,
    )

    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        model_file = tdir / "test.msgpack.xz"

        saving.save_model(arc_cls, model_file, metadata=dict(config=arc_cls.config))
        loaded = saving.load_model(model_file)

        assert arc_cls.config == loaded.config

        if flax_version >= (0, 12):
            expected = dict(nnx.flatten(nnx.pure(nnx.state(arc_cls, nnx.Param)))[1])
            actual = dict(nnx.flatten(loaded.state)[1])
            for k in set(expected) | set(actual):
                e = expected.get(k)
                a = actual.get(k)
                assert e is not None, f"Unexpected key {k}"
                assert a is not None, f"Missing key {k}"
                try:
                    assert np.allclose(e, a), f"Values differ in {k}"
                except Exception as ex:
                    raise RuntimeError(
                        f"Failed to compare {k}: {type(a).__name__} vs {type(e).__name__}."
                        f"\na = {str(a)[:1000]}\ne = {str(e)[:1000]}"
                    ) from ex
