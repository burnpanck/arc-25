import typing

import attrs
import jax.numpy as jnp
import jaxtyping as jt
from flax import nnx
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .. import symmetry
from ..lib import nnx_compat
from .encoder import ARCEncoder, LayerStack
from .fields import CoordinateGrid, Field, FieldDims
from .linear import SymDecompLinear
from .symrep import RepSpec, SplitSymDecomp, SymDecompBase, SymDecompDims, standard_rep
from .transformer import FieldTransformer


class MaskedAutoencoder(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: FieldDims,
        num_decoder_layers: int = 4,
        mlp_features_per_flavour: int = 4,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        keep_rngs: bool = True,
        rngs: nnx.Rngs,
        **kw,
    ):
        self.dtype = dtype
        self.hidden_size = hidden_size

        self.encoder = ARCEncoder(
            hidden_size=hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            **kw,
        )
        self.config = dict(self.encoder.config)
        self.config.update(
            num_decoder_layers=num_decoder_layers,
            mlp_features_per_flavour=mlp_features_per_flavour,
        )

        stack_kw = dict(self.encoder.config)
        for k in [
            "num_layers",
            "hidden_size",
            "num_perceiver_layers",
            "num_perceiver_tokens",
        ]:
            stack_kw.pop(k)

        dhs = self.decoder_hidden_size = self.encoder.perceiver_hidden_size
        n_flavours = dhs.rep.n_flavours

        self.mask_embedding = SymDecompLinear(
            SymDecompDims(
                invariant=1,
                space=2,
                flavour=0,
                rep=RepSpec(symmetry.AxialDirRep, n_flavours),
            ),
            dhs.cells,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.decoder = LayerStack(
            num_layers=num_decoder_layers,
            hidden_size=dhs,
            style="decoder",
            keep_rngs=keep_rngs,
            rngs=rngs,
            **stack_kw,
        )

        mlp_hidden_dims = SymDecompDims(
            flavour=mlp_features_per_flavour,
            invariant=0,
            space=0,
            rep=RepSpec(symmetry.TrivialRep, n_flavours=n_flavours),
        )

        final_dtype = (
            jnp.promote_types(dtype, jnp.float32) if dtype is not None else jnp.float32
        )

        self.reduction = SymDecompLinear(
            dhs.cells,
            mlp_hidden_dims,
            dtype=final_dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.classifier_activation = nnx.gelu
        self.classifier = nnx.Linear(
            mlp_features_per_flavour,
            1,
            dtype=final_dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        mask: jt.Bool[jt.Array, "... Y X"],
        *,
        with_stats: bool = False,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        **kw,
    ) -> jt.Float[jt.Array, "... F"]:
        batch = x.shape[:-2]
        Y, X = x.shape[-2:]

        lin_kw = dict(
            mode=mode,
        )
        enc_kw = dict(
            deterministic=deterministic,
            unroll=unroll,
            remat=remat,
            rngs=rngs,
            **lin_kw,
            **kw,
        )
        res = self.encoder(x, size, mask=mask, **enc_kw, with_stats=with_stats)
        if with_stats:
            encoded, encoder_stats = res
        else:
            encoded = res

        dtype = encoded.cells.invariant.dtype
        n_flavours = encoded.cells.rep.n_flavours

        # no more mask here!
        new_grid = CoordinateGrid.for_batch(Y, X, size, dtype=dtype)

        # TODO: we could probably share this one with the original call within self.encoder above.
        # the only difference is in the mask.
        pos_enc, pos_enc_rep = self.encoder.encode_positions(size, new_grid)
        mask_enc = SplitSymDecomp(
            invariant=(new_grid.mask & ~encoded.grid.mask).astype(dtype)[..., None],
            flavour=jnp.zeros(batch + (Y, X, n_flavours, 0), dtype),
            space=pos_enc.astype(dtype),
            rep=RepSpec(pos_enc_rep, n_flavours),
        )

        y = Field(
            context=encoded.context,
            cells=self.mask_embedding(mask_enc, **lin_kw),
            grid=new_grid,
        ).as_split()

        res = self.decoder(y, **enc_kw, with_stats=with_stats)
        if with_stats:
            decoded, decoder_stats = res
        else:
            decoded = res

        z = self.reduction(decoded.cells, **lin_kw).as_split().flavour

        z = self.classifier_activation(z)
        z = self.classifier(z)

        logits = z.reshape(z.shape[:-1])

        if with_stats:
            stats = dict(encoder=encoder_stats, decoder=decoder_stats)
            return logits, stats
        return logits
