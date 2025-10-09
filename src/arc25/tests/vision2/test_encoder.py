import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.encoder import ARCEncoder
from ...vision2.fields import CoordinateGrid, Field, FieldDims
from ...vision2.symrep import RepSpec, SymDecompDims
from .conftest import quant


@pytest.mark.parametrize("use_chirality", [False, True])
def test_ARCEncoder_symmetry(use_chirality):
    """Test that ARCEncoder preserves D4 symmetry."""
    with jax.experimental.enable_x64():
        # Prepare example encoder with small dimensions

        hidden_size = FieldDims(
            context=SymDecompDims(43, 11, 7),
            cells=SymDecompDims(41, 5, 3),
            context_tokens=3,
        )

        rngs = nnx.Rngs(42)
        encoder = ARCEncoder(
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

        # Encode the input
        enc = encoder.encode(inp, size)
        out = encoder(inp, size, deterministic=True, unroll=False, remat=False)

        # Verify symmetry preservation for all D4 operations
        for op in D4:

            def tfo_field(obj: Field) -> Field:
                # Get transformation indices for space representations
                tmp = attrs.evolve(
                    obj,
                    **{
                        k: attrs.evolve(
                            v,
                            space=v.space[
                                ...,
                                symmetry.transform_coeff_in_basis(op, v.rep.space),
                                :,
                            ],
                        )
                        for k, v in obj.projections.items()
                    },
                )
                grid = tmp.grid
                if op.value & D4.t.value:
                    grid = attrs.evolve(
                        grid,
                        xpos=grid.ypos,
                        ypos=grid.xpos,
                        mask=grid.mask.transpose(0, 2, 1),
                    )
                return attrs.evolve(
                    tmp,
                    cells=tmp.cells.map_elementwise(
                        lambda v: symmetry.transform_image(op, v, ydim=1, xdim=2)
                    ),
                    grid=grid,
                )

            # Transform input image
            tinp = symmetry.transform_image(op, inp, ydim=1, xdim=2)

            # Transform size (height and width may swap under transpose)
            if op.value & D4.t.value:
                # Transpose: swap Y and X
                tsize = size[..., ::-1]
            else:
                tsize = size

            # Encode transformed input
            tenc = encoder.encode(tinp, tsize)
            tout = encoder(tinp, tsize, deterministic=True, unroll=False, remat=False)

            # Transform the original output to get expected result
            eenc = tfo_field(enc)
            expected = tfo_field(out)

            # Verify both context and cells match
            for kk, (ev, av) in dict(
                pre_encoded=(eenc, tenc), output=(expected, tout)
            ).items():
                for proj_name in ["context", "cells"]:
                    expected_proj = getattr(ev, proj_name)
                    actual_proj = getattr(av, proj_name)

                    for k, v in expected_proj.representations.items():
                        a = getattr(actual_proj, k)
                        assert (
                            a.shape == v.shape
                        ), f"{op} {kk}.{proj_name}.{k}: {a.shape} <> {v.shape}"
                        if not a.size:
                            continue
                        max_diff = np.abs(a - v).max()
                        rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
                        bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
                        n_failed = int(bad.sum())
                        assert np.allclose(
                            a, v, rtol=1e-4, atol=1e-3
                        ), f"{op} {kk}.{proj_name}.{k}: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}"
                        print(
                            f"{op} {kk}.{proj_name}.{k}: {max_diff=:.3e}, {rel_diff=:.5f}"
                        )
