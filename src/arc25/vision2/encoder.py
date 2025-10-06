import typing

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import nnx
from flax.nnx.nn.linear import default_bias_init
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .. import symmetry
from ..lib import nnx_compat
from ..lib.misc import first_from
from .fields import CoordinateGrid, Field, FieldDims
from .linear import SymDecompLinear
from .symrep import RepSpec, SplitSymDecomp, SymDecompBase, SymDecompDims, standard_rep
from .transformer import FieldTransformer


class ARCEncoder(nnx.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_layers: int = 12,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        hidden_size: FieldDims,
        qk_head_width: FieldDims | None = None,
        v_head_width: FieldDims | None = None,
        swiglu_width_factor: float | None = None,
        use_chirality_rep: bool = True,
        per_head_rope_freq: bool = True,
        num_groups: int | None = None,
        dropout_rate: float = 0.1,
        keep_rngs: bool = True,
        remat: bool | None = True,
        unroll: int | bool | None = 1,
        rngs: nnx.Rngs,
    ):
        self.num_layers = num_layers
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = hidden_size
        self.remat = remat
        self.unroll = unroll
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        embedding_dims = dict(
            context=SymDecompDims(
                invariant=0,
                space=1,
                flavour=2,
                rep=RepSpec(symmetry.AxisRep, hidden_size.context.n_flavours),
            ),
            cells=SymDecompDims(
                invariant=2,
                space=2,
                flavour=1,
                rep=RepSpec(symmetry.AxialDirRep, hidden_size.cells.n_flavours),
            ),
        )
        self.embedding = hidden_size.map_projections(
            lambda k, v: SymDecompLinear(
                embedding_dims[k],
                v,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                use_bias=k != "context",
            ),
            cls=nnx_compat.Dict,
        )
        self.context_tokens = nnx.Param(
            default_bias_init(
                rngs.params(),
                (hidden_size.context_tokens, hidden_size.context.invariant),
                param_dtype,
            )
        )

        # As Block contains dropout op, we prefer
        # to split RNG into num_layers of RNGs
        # using @nnx.split_rngs decorator.
        # Next, nnx.vmap creates a vectorized version of Block.
        # in_axes and out_axes define vectorization axis
        # of the input splitted rngs and the output Block instance.
        # Both axes should be 0.
        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs: nnx.Rngs):
            return FieldTransformer(
                hidden_size=hidden_size,
                qk_head_width=qk_head_width,
                v_head_width=v_head_width,
                swiglu_width_factor=swiglu_width_factor,
                use_chirality_rep=use_chirality_rep,
                per_head_rope_freq=per_head_rope_freq,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout_rate=dropout_rate,
                dtype=dtype,
                param_dtype=param_dtype,
                # attention_dtype=attention_dtype,
                precision=precision,
                # we can't hold on to the Rngs, as we want to carry it through the scan later
                keep_rngs=False,
                rngs=rngs,
            )

        self.blocks = create_block(rngs)

        # Layer normalization with `flax.nnx.LayerNorm`.
        self.final_norm = hidden_size.map_representations(
            lambda k, kk, v: nnx.LayerNorm(
                v,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        )

        self.rngs = rngs if keep_rngs else nnx_compat.data(None)

    def __call__(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
    ) -> Field:
        """
        Encodes an ARC-style input grid into a `SymDecomp` of dimensions `(..., F, R?, C)`;
        Here, `F = num_colours+1`, where the first flavour is colour-indepenent, and
        the others map to the input colours.
        """
        pre_embedding = self.encode(x, size)
        print(f"{pre_embedding.shapes=}")
        embedding = pre_embedding.map_projections(lambda v, f: f(v), self.embedding)
        print(f"{embedding.shapes=}")
        embedding = attrs.evolve(
            embedding,
            context=attrs.evolve(
                embedding.context,
                invariant=embedding.context.invariant + self.context_tokens.value,
            ),
        )

        # Apply the dropout layer to embedded patches.
        embedding = embedding.map_representations(
            self.dropout, rngs=rngs, deterministic=deterministic
        )

        # Transformer encoder blocks.
        # Process the embedded patches through the transformer encoder layers.

        # We use nnx.scan to apply sequentially the blocks
        # on the input, for example with num_layers=3
        # output = block[0](x)
        # output = block[1](output)
        # output = block[2](output)
        #
        # In `forward` function defined below:
        # - x represents the loop carry value
        # - model is the data to scan along the leading axis
        # nnx.scan args:
        # - in_axes marks the inputs: x is marked as carry
        # and the model is to scan along the axis 0
        # - out_axes marks the output as carry

        def forward(
            carry: tuple[Field, nnx.Rngs], block: FieldTransformer
        ) -> tuple[Field, nnx.Rngs]:
            x, rngs = carry
            x = block(x, rngs=rngs, deterministic=deterministic, mode=mode)
            return x, rngs

        remat = nnx.module.first_from(
            remat,
            self.remat,
            error_msg="ARCEncoder needs `remat`, either in constructor or call",
        )
        if remat is not False:
            forward = nnx.remat(
                forward,
                prevent_cse=False,
                policy=(
                    jax.checkpoint_policies.checkpoint_dots if remat is True else remat
                ),
            )
        unroll = nnx.module.first_from(
            unroll,
            self.unroll,
            error_msg="ARCEncoder needs `unroll`, either in constructor or call",
        )
        forward = nnx.scan(
            forward, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry, unroll=unroll
        )

        rngs = nnx.module.first_from(
            rngs,
            self.rngs,
            error_msg="""No `rngs` argument was provided to ARCEncoder
as either a __call__ argument or class attribute""",
        )

        carry = (embedding, rngs)
        carry = forward(carry, self.blocks)
        x, rngs = carry
        # Apply final layer normalization
        y = x.map_representations(lambda v, f: f(v), self.final_norm)
        return y

    def encode(
        self, x: jt.Int[jt.Array, "... Y X"], size: jt.Int[jt.Array, "... 2"]
    ) -> Field:
        batch = x.shape[:-2]
        Y, X = shape = x.shape[-2:]  # noqa: F841
        hs = self.hidden_size
        T = hs.context_tokens

        grid = CoordinateGrid.for_batch(Y, X, size)

        dtype = self.dtype or jnp.float32

        x = x[..., :, :, None]
        print(f"{x.shape=} {grid.mask.shape=}")

        colour_idx = np.r_[: hs.rep.n_flavours]
        presence = (x == colour_idx) & grid.mask[..., None]
        count = presence.sum((-3, -2)).astype(dtype)
        prevalence = count / (
            1 + grid.mask.astype(dtype).sum((-2, -1))[..., None]
        ).astype(dtype)
        intensity = 1 / (1 + count)
        print(
            f"{presence.shape=} {count.shape=} {grid.mask.shape=} {prevalence.shape=} {intensity.shape=}"
        )
        # size comes in as (height, width)
        axl = [symmetry.AxisRep.v, symmetry.AxisRep.h]
        context = SplitSymDecomp(
            invariant=jnp.empty(batch + (T, 0), dtype),
            flavour=jnp.tile(
                jnp.concatenate(
                    [prevalence[..., None], intensity[..., None]], dtype=dtype, axis=-1
                )[..., None, :, :],
                (T, 1, 1),
            ),
            space=jnp.tile(
                size.astype(dtype)[
                    ..., None, [axl.index(basis) for basis in symmetry.AxisRep], None
                ],
                (T, 1, 1),
            ),
            rep=RepSpec(symmetry.AxisRep, hs.context.n_flavours),
        )

        cells_flavour = presence[..., :, :, :, None].astype(dtype)
        cells_invariant = jnp.where(
            grid.mask[..., None],
            jnp.concatenate(
                [
                    jnp.take_along_axis(prevalence[..., None, None, :], x, axis=-1),
                    jnp.take_along_axis(intensity[..., None, None, :], x, axis=-1),
                ],
                dtype=dtype,
                axis=-1,
            ),
            0,
        )

        def encd(arr, abso=0, relo=0):
            # input is (absolute, relative)
            return jnp.concatenate(
                [1 / jnp.maximum(1, 1 + arr[..., :1] + abso), arr[..., 1:] + relo],
                axis=-1,
            )

        encoded_dir_info = {
            symmetry.AxialDirRep.d: encd(grid.ypos[..., :, None, :]),
            symmetry.AxialDirRep.u: encd(
                -grid.ypos[..., :, None, :], size[..., None, None, :1], 1
            ),
            symmetry.AxialDirRep.r: encd(grid.xpos[..., None, :, :]),
            symmetry.AxialDirRep.l: encd(
                -grid.xpos[..., None, :, :], size[..., None, None, 1:], 1
            ),
        }
        cells_space = jnp.where(
            grid.mask[..., None, None],
            jnp.concatenate(
                [
                    jnp.broadcast_to(encoded_dir_info[basis], batch + (Y, X, 2))[
                        ..., :, :, None, :
                    ]
                    for basis in symmetry.AxialDirRep
                ],
                axis=-2,
            ),
            0,
        )

        cells = SplitSymDecomp(
            invariant=cells_invariant,
            flavour=cells_flavour,
            space=cells_space,
            rep=RepSpec(symmetry.AxialDirRep, hs.cells.n_flavours),
        )
        print(f"{context.shapes=} {cells.shapes=}")
        return Field(
            context=context,
            cells=cells,
            grid=grid,
        )
