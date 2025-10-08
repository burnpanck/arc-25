import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.fields import CoordinateGrid, Field, FieldDims
from ...vision2.symrep import SymDecompDims
from ...vision2.transformer import FieldTransformer
from .conftest import quant, verify_swap


@pytest.mark.parametrize("use_chirality", [False, True])
def test_FieldTransformer_symmetry(use_chirality):
    """Test that ARCEncoder preserves D4 symmetry."""
    with jax.experimental.enable_x64():
        # Prepare example encoder with small dimensions
        from ...vision2.symrep import RepSpec

        hidden_size = FieldDims(
            context=SymDecompDims(43, 11, 7),
            cells=SymDecompDims(41, 5, 3),
            context_tokens=3,
        )

        rngs = nnx.Rngs(42)
        layer = FieldTransformer(
            num_heads=4,
            num_groups=2,
            hidden_size=hidden_size,
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
        feat = hidden_size.make_empty(batch=Bs, shape=(Y, X), dtype=jnp.float32)
        feat = feat.map_projections(
            lambda proj: proj.map_representations(
                lambda rep: quant(rngs(), rep.shape, rep.dtype)
            )
        )

        def fix_grid(obj: Field) -> Field:
            shape = obj.cells.batch_shape[1:3]
            grid = CoordinateGrid.for_batch(
                *shape,
                jnp.tile(jnp.array(shape), Bs + (1,)),
                dtype=jnp.float32,
            )
            return attrs.evolve(obj, grid=grid)

        feat = fix_grid(feat)

        # process the input
        out = layer(feat, deterministic=True)

        # Verify symmetry preservation for all D4 operations
        for op in D4:
            # Get transformation indices for space representations
            rtfo = symmetry.transform_coeff_in_basis(op, symmetry.FullRep)

            def tfo_field(obj: Field) -> Field:
                tmp = attrs.evolve(
                    obj,
                    **{
                        k: attrs.evolve(v, space=v.space[..., rtfo, :])
                        for k, v in obj.projections.items()
                    },
                )
                return fix_grid(
                    attrs.evolve(
                        tmp,
                        cells=tmp.cells.map_elementwise(
                            lambda v: symmetry.transform_image(op, v, ydim=1, xdim=2)
                        ),
                    )
                )

            # Transform input image
            tinp = tfo_field(feat)

            # process transformed input
            tout = layer(tinp, deterministic=True)

            # Transform the original output to get expected result
            expected = tfo_field(out)

            # Verify both context and cells match
            for proj_name in ["context", "cells"]:
                expected_proj = getattr(expected, proj_name)
                actual_proj = getattr(tout, proj_name)

                for k, v in expected_proj.representations.items():
                    a = getattr(actual_proj, k)
                    max_diff = np.abs(a - v).max()
                    rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
                    bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
                    n_failed = int(bad.sum())
                    assert np.allclose(
                        a, v, rtol=1e-4, atol=1e-3
                    ), f"{op} {proj_name}.{k}: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}"
                    print(f"{op} {proj_name}.{k}: {max_diff=:.3e}, {rel_diff=:.5f}")
