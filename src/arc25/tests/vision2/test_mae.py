import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from ... import symmetry
from ...symmetry import D4
from ...vision2.fields import CoordinateGrid, Field, FieldDims
from ...vision2.mae import MaskedAutoencoder
from ...vision2.symrep import RepSpec, SymDecompDims
from .conftest import quant


@pytest.mark.parametrize(
    "config,with_attention_map",
    [
        pytest.param(
            dict(
                norm_per="basis-nnx",
                rope_freq_scaling="linear-k",
                learnable_rope_freqs="tied",
                decoder_cell_infusion="legacy",
                use_chirality_rep=False,
                head_rep=symmetry.TrivialRep,
                with_final_norm=False,
            ),
            False,
            id="legacy",
        ),
        pytest.param(
            dict(
                norm_per="all",
                rope_freq_scaling="linear-freqs",
                learnable_rope_freqs="none",
                decoder_cell_infusion="backbone+semantic",
                use_chirality_rep=False,
                head_rep=symmetry.TrivialRep,
                with_final_norm=True,
            ),
            False,
            id="modern",
        ),
        pytest.param(
            dict(
                norm_per="rep",
                rope_freq_scaling="linear-freqs",
                learnable_rope_freqs="per-head",
                decoder_cell_infusion="backbone+semantic",
                use_chirality_rep=True,
                head_rep=symmetry.AxisRep,
                with_final_norm=True,
            ),
            True,
            id="exotic",
        ),
    ],
)
def test_MaskedAutoencoder_symmetry(config, with_attention_map):
    """Test that ARCEncoder preserves D4 symmetry."""
    with jax.enable_x64():
        # Prepare example encoder with small dimensions

        use_chirality = config["use_chirality_rep"]

        hidden_size = FieldDims(
            context=SymDecompDims(43, 11, 7),
            cells=SymDecompDims(41, 5, 3),
            context_tokens=3,
        )

        rngs = nnx.Rngs(42)
        mae = MaskedAutoencoder(
            num_layers=1,
            num_heads=4,
            num_groups=2,
            hidden_size=hidden_size,
            num_perceiver_tokens=7,
            num_perceiver_layers=1,
            num_decoder_layers=1,
            decoder_cell_width=(
                SymDecompDims(
                    invariant=37,
                    space=7,
                    flavour=5,
                )
                if config["decoder_cell_infusion"] != "legacy"
                else None
            ),
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
            dropout_rate=0.0,
            rngs=rngs,
            **config,
        )

        # Generate example input
        Bs = (3,)
        Y, X = 13, 17
        inp = jax.random.randint(rngs(), Bs + (Y, X), 0, 10)
        size = jnp.tile(jnp.array((Y, X)), Bs + (1,))
        mask = jax.random.randint(rngs(), Bs + (Y, X), 0, 10) < 5

        # Encode the input
        res = mae(
            inp,
            size,
            mask=mask,
            deterministic=True,
            unroll=False,
            remat=False,
            with_attention_maps=with_attention_map,
        )
        if with_attention_map:
            out, attn = res  # noqa: F841
        else:
            out = res
            attn = None  # noqa: F841

        # Verify symmetry preservation for all D4 operations
        for op in D4:
            # Transform input image
            tinp = symmetry.transform_image(op, inp, ydim=1, xdim=2)
            tmask = symmetry.transform_image(op, mask, ydim=1, xdim=2)

            # Transform size (height and width may swap under transpose)
            if op.value & D4.t.value:
                # Transpose: swap Y and X
                tsize = size[..., ::-1]
            else:
                tsize = size

            # Encode transformed input
            res = mae(
                tinp,
                tsize,
                mask=tmask,
                deterministic=True,
                unroll=False,
                remat=False,
                with_attention_maps=with_attention_map,
            )
            if with_attention_map:
                tout, tattn = res  # noqa: F841
            else:
                tout = res
                tattn = None  # noqa: F841

            # Transform the original output to get expected result
            expected = symmetry.transform_image(op, out, ydim=1, xdim=2)

            a = tout
            v = expected
            assert a.shape == v.shape, f"{op}: {a.shape} <> {v.shape}"
            if not a.size:
                continue
            max_diff = np.abs(a - v).max()
            rel_diff = (np.abs(a - v) / (np.abs(v) + 1e-8)).max()
            bad = np.abs(a - v) > 1e-4 * np.abs(v) + 1e-3
            n_failed = int(bad.sum())
            assert np.allclose(
                a, v, rtol=1e-4, atol=1e-3
            ), f"{op}: {max_diff=:.3e}, {rel_diff=:.5f}, {n_failed=}/{v.size}"
            print(f"{op}: {max_diff=:.3e}, {rel_diff=:.5f}")
