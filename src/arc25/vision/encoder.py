import typing

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import nnx
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .fields import Field, FieldDims
from .linear import SymmetricLinear
from .symrep import SymDecomp, standard_rep
from .transformer import FieldTransformer


class ARCEncoder(nnx.Module):
    def __init__(
        self,
        *,
        num_colours: int = 10,
        num_layers: int = 12,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        hidden_size: FieldDims,
        mha_features: int,
        mlp_width_factor: float,
        num_heads: int,
        num_groups: int | None = None,
        dropout_rate: float = 0.1,
        keep_rngs: bool = True,
        remat: bool | None = True,
        unroll: int | bool | None = 1,
        rngs: nnx.Rngs,
    ):
        self.num_colours = num_colours
        self.num_layers = num_layers
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = hidden_size
        self.remat = remat
        self.unroll = unroll
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.embedding = hidden_size.map_projections(
            lambda k, v: SymmetricLinear(
                attrs.evolve(v, inv=dict(context=2, cells=2).get(k, 0), equiv=0),
                v,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
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
                mha_features=mha_features,
                mlp_width_factor=mlp_width_factor,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout_rate=dropout_rate,
                dtype=dtype,
                param_dtype=param_dtype,
                attention_dtype=jnp.promote_types(dtype, jnp.float32),
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

        self.rngs = rngs if keep_rngs else nnx.data(None)

    def __call__(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
    ) -> Field:
        """
        Encodes an ARC-style input grid into a `SymDecomp` of dimensions `(..., F, R?, C)`;
        Here, `F = num_colours+1`, where the first flavour is colour-indepenent, and
        the others map to the input colours.
        """
        pre_embedding = self.encode(x, size)
        # print(f"{pre_embedding.shapes=}")
        embedding = pre_embedding.map_projections(lambda v, f: f(v), self.embedding)
        # print(f"{embedding.shapes=}")

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
            x = block(x, rngs=rngs, deterministic=deterministic)
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
        Y, X = shape = x.shape[-2:]
        Fc = self.num_colours
        F = Fc + 1
        R = standard_rep.dim

        dtype = self.dtype or jnp.float32

        x = x[..., :, :, None]
        sY = size[..., 0, None]
        sX = size[..., 1, None]
        xpos = jnp.concatenate(
            [
                jnp.tile(np.r_[:X][:, None].astype(dtype), batch + (1, 1)),
                np.r_[:X][:, None].astype(dtype) / sX[..., :, None].astype(dtype),
            ],
            dtype=dtype,
            axis=-1,
        )
        ypos = jnp.concatenate(
            [
                jnp.tile(np.r_[:Y][:, None].astype(dtype), batch + (1, 1)),
                np.r_[:Y][:, None].astype(dtype) / sY[..., :, None].astype(dtype),
            ],
            dtype=dtype,
            axis=-1,
        )
        rmsk = np.r_[:Y] < sY
        cmsk = np.r_[:X] < sX
        mask = rmsk[..., :, None] & cmsk[..., None, :]
        # print(f"{x.shape=} {sY.shape=} {sX.shape=} {xpos.shape=} {rmsk.shape=} {mask.shape=}")

        colour_idx = np.r_[: self.num_colours]
        presence = (x == colour_idx) & mask[..., None]
        # TODO: product instead of mask?
        prevalence = presence.sum((-3, -2)).astype(dtype) / (
            1 + mask.astype(dtype).sum((-2, -1))[..., None]
        ).astype(dtype)
        intensity = 1 / (1 + prevalence)

        special_ind = jnp.concatenate(
            [
                jnp.ones(batch + (1, 1), dtype),
                jnp.zeros(batch + (Fc, 1), dtype),
            ],
            axis=-2,
        )
        prevalence_ind = jnp.concatenate(
            [
                jnp.zeros(batch + (1, 1), dtype),
                prevalence[..., :, None],
            ],
            axis=-2,
        )
        context = jnp.concatenate([special_ind, prevalence_ind], dtype=dtype, axis=-1)

        presence_ind = jnp.concatenate(
            [
                jnp.zeros(batch + (Y, X, 1, 1), dtype),
                presence[..., :, None].astype(dtype),
            ],
            axis=-2,
        )
        intensity_ind = jnp.concatenate(
            [
                jnp.take_along_axis(intensity[..., None, None, :], x, axis=-1)[
                    ..., :, :, :, None
                ],
                jnp.zeros(batch + (Y, X, Fc, 1), dtype),
            ],
            axis=-2,
        )
        cells = jnp.concatenate([presence_ind, intensity_ind], dtype=dtype, axis=-1)
        # print(f"{batch=} {context.shape=} {cells.shape=}")

        rrep = self.hidden_size.rows.rep
        crep = self.hidden_size.cols.rep
        return Field(
            context=SymDecomp(inv=context, equiv=jnp.empty(batch + (F, R, 0), dtype)),
            rows=SymDecomp(
                inv=jnp.empty(batch + (Y, F, 0), dtype),
                equiv=jnp.empty(batch + (Y, F, rrep.dim, 0), dtype),
                rep=rrep,
            ),
            cols=SymDecomp(
                inv=jnp.empty(batch + (X, F, 0), dtype),
                equiv=jnp.empty(batch + (X, F, crep.dim, 0), dtype),
                rep=crep,
            ),
            cells=SymDecomp(
                inv=cells, equiv=jnp.empty(batch + shape + (F, R, 0), dtype)
            ),
            xpos=xpos,
            ypos=ypos,
            rmsk=rmsk,
            cmsk=cmsk,
            mask=mask,
        )
