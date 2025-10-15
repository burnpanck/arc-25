import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.fields import CoordinateGrid, Field, FieldDims
from ...vision2.symrep import RepSpec, SymDecompDims
from ...vision2.transformer import FieldTransformer
from .conftest import quant, verify_swap


@pytest.mark.parametrize("use_chirality", [False, True])
@pytest.mark.parametrize("style", ["co-attention", "perceiver", "decoder"])
@pytest.mark.parametrize("norm_per", ["basis-nnx", "all", "rep", "basis"])
@pytest.mark.parametrize("with_attention_maps", [True])
def test_FieldTransformer_symmetry(use_chirality, style, norm_per, with_attention_maps):
    """Test that FieldTransformer preserves D4 symmetry."""
    global_head_rep = symmetry.FullRep
    with jax.experimental.enable_x64():
        # Prepare example transformer layer with small dimensions
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
            head_rep=global_head_rep,
            dtype=np.float64,
            use_chirality_rep=use_chirality,
            norm_per=norm_per,
            dropout_rate=0.0,
            style=style,
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
        res = layer(feat, deterministic=True, with_attention_maps=with_attention_maps)
        if with_attention_maps:
            out, intermediates = res
        else:
            out = res

        # Verify symmetry preservation for all D4 operations
        for op in D4:
            # Get transformation indices for space representations
            rtfo_full = symmetry.transform_coeff_in_basis(op, symmetry.FullRep)
            rtfo_axial = symmetry.transform_coeff_in_basis(op, symmetry.AxialDirRep)
            extra_rep = symmetry.ChiralityRep if use_chirality else symmetry.TrivialRep
            rtfo_extra = symmetry.transform_coeff_in_basis(op, extra_rep)
            rtfo_global = symmetry.transform_coeff_in_basis(op, global_head_rep)

            def tfo_field(obj: Field) -> Field:
                tmp = attrs.evolve(
                    obj,
                    **{
                        k: attrs.evolve(v, space=v.space[..., rtfo_full, :])
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
            tres = layer(
                tinp, deterministic=True, with_attention_maps=with_attention_maps
            )
            if with_attention_maps:
                tout, trafo_intermediates = tres
            else:
                tout = tres

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

            # Verify attention maps respect symmetry if requested
            if with_attention_maps:
                for k in ["sa_maps", "ca_maps"]:
                    v = intermediates[k]
                    for kk, vv in v.items():
                        transformed_attn_map = trafo_intermediates[k][kk]
                        expected_map = vv

                        if kk == "cells":
                            # Self-attention (cells->cells): Shape: ...yxdv (batch, spatial, directional, extra_rep)
                            # Cross-attention (context->cells): Shape: ...yxv (batch, spatial, global_head_rep)
                            expected_map = symmetry.transform_image(
                                op, expected_map, ydim=len(Bs) + 0, xdim=len(Bs) + 1
                            )
                        if k == "sa_maps" and kk == "cells":
                            # Shape: ...dv (batch, ..., directional, extra_rep)
                            assert expected_map.shape[-2:] == (
                                len(rtfo_axial),
                                len(rtfo_extra),
                            )
                            expected_map = expected_map[..., rtfo_axial, :]
                            expected_map = expected_map[..., rtfo_extra]
                        else:
                            # Shape: ...dv (batch, ..., global_head_rep)
                            assert expected_map.shape[-1] == len(rtfo_global)
                            expected_map = expected_map[..., rtfo_global]

                        max_diff = np.abs(transformed_attn_map - expected_map).max()
                        rel_diff = (
                            np.abs(transformed_attn_map - expected_map)
                            / (np.abs(expected_map) + 1e-8)
                        ).max()

                        assert np.allclose(
                            transformed_attn_map, expected_map, rtol=1e-4, atol=1e-3
                        ), f"{op} attn {k}.{kk}: {max_diff=:.3e}, {rel_diff=:.5f}"
                        print(f"{op} attn {k}.{kk}: {max_diff=:.3e}, {rel_diff=:.5f}")
