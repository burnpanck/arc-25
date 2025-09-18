import typing

import attrs
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import nnx

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
        dtype: typing.Any | None = None,
        hidden_size: FieldDims,
        mha_features: int,
        mlp_width_factor: float,
        num_heads: int,
        num_groups: int | None = None,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ):
        self.num_colours = num_colours
        self.dtype = dtype
        self.hidden_size = hidden_size

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.embedding = hidden_size.map_projections(
            lambda k, v: SymmetricLinear(
                attrs.evolve(v, inv=dict(context=2, cells=2).get(k, 0), equiv=0),
                v,
                rngs=rngs,
            )
        )
        self.encoder = nnx.Sequential(
            *[
                FieldTransformer(
                    hidden_size=hidden_size,
                    mha_features=mha_features,
                    mlp_width_factor=mlp_width_factor,
                    num_heads=num_heads,
                    num_groups=num_groups,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                )
                for i in range(num_layers)
            ]
        )

        # Layer normalization with `flax.nnx.LayerNorm`.
        self.final_norm = hidden_size.map_representations(
            lambda k, kk, v: nnx.LayerNorm(v, rngs=rngs)
        )

    def __call__(
        self, x: jt.Int[jt.Array, "... Y X"], size: jt.Int[jt.Array, "... 2"]
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
        embedding = embedding.map_representations(self.dropout)

        # Transformer encoder blocks.
        # Process the embedded patches through the transformer encoder layers.
        x = self.encoder(embedding)
        # Apply final layer normalization
        return x.map_representations(lambda v, f: f(v), self.final_norm)

    def encode(
        self, x: jt.Int[jt.Array, "... Y X"], size: jt.Int[jt.Array, "... 2"]
    ) -> Field:
        batch = x.shape[:-2]
        Y, X = shape = x.shape[-2:]
        Fc = self.num_colours
        F = Fc + 1
        R = standard_rep.dim

        dtype = self.dtype

        x = x[..., :, :, None]
        sY = size[..., 0, None]
        sX = size[..., 1, None]
        xpos = jnp.concatenate(
            [
                jnp.tile(np.r_[:X][:, None].astype(dtype), batch + (1, 1)),
                np.r_[:X][:, None].astype(dtype) / sX[..., :, None],
            ],
            axis=-1,
        )
        ypos = jnp.concatenate(
            [
                jnp.tile(np.r_[:Y][:, None].astype(dtype), batch + (1, 1)),
                np.r_[:Y][:, None].astype(dtype) / sY[..., :, None],
            ],
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
            1 + mask.sum((-2, -1))[..., None].astype(dtype)
        )
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
        context = jnp.concatenate([special_ind, prevalence_ind], axis=-1)

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
        cells = jnp.concatenate([presence_ind, intensity_ind], axis=-1)
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
