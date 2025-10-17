import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.classification import ARCClassifier
from ...vision2.fields import CoordinateGrid, Field, FieldDims
from ...vision2.symrep import RepSpec, SymDecompDims
from .conftest import quant


@pytest.mark.parametrize("use_chirality", [False, True])
def test_ARCClassifier_symmetry(use_chirality):
    """Test that ARCClassifier preserves D4 symmetry."""
    with jax.enable_x64():
        # Prepare example encoder with small dimensions

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
                rep=RepSpec(
                    symmetry.TrivialRep if use_chirality else symmetry.ChiralityRep, 10
                ),
            ),
            v_head_width=SymDecompDims(13, 5, 3),
            dtype=np.float64,
            use_chirality_rep=use_chirality,
            dropout_rate=0.0,
            rngs=rngs,
        )

        # Generate example input
        Bs = (3,)
        Y, X = 13, 17
        inp = jax.random.randint(rngs(), Bs + (Y, X), 0, 10)
        size = jnp.tile(jnp.array((Y, X)), Bs + (1,))

        # Classify the input
        expected = arc_cls(inp, size, deterministic=True, unroll=False, remat=False)

        # Verify symmetry preservation for all D4 operations
        for op in D4:
            # Transform input image
            tinp = symmetry.transform_image(op, inp, ydim=1, xdim=2)

            # Transform size (height and width may swap under transpose)
            if op.value & D4.t.value:
                # Transpose: swap Y and X
                tsize = size[..., ::-1]
            else:
                tsize = size

            # Classify transformed input
            opi = {v: k for k, v in enumerate(D4)}
            actual = arc_cls(
                tinp,
                tsize,
                transform=[opi[op.inverse]],
                deterministic=True,
                unroll=False,
                remat=False,
            )

            # Verify logits match
            a = actual
            v = expected
            assert a.shape == v.shape, f"{op}: {a.shape} <> {v.shape}"
            max_diff = np.abs(a - v).max()
            rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
            bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
            n_failed = int(bad.sum())
            assert np.allclose(
                a, v, rtol=1e-4, atol=1e-3
            ), f"{op}: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}"
            print(f"{op}: {max_diff=:.3e}, {rel_diff=:.5f}")
