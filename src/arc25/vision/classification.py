import typing

import attrs
import jax.numpy as jnp
import jaxtyping as jt
from flax import nnx
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .encoder import ARCEncoder
from .fields import Field, FieldDims
from .linear import SymmetricLinear
from .symrep import SymDecomp, standard_rep
from .transformer import FieldTransformer


class ARCClassifier(nnx.Module):
    def __init__(
        self,
        *,
        num_classes: int = 1000,
        num_colours: int = 10,
        num_layers: int = 12,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        hidden_size: FieldDims,
        mha_features: int,
        mlp_width_factor: float = 4,
        num_heads: int,
        num_groups: int | None = None,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
        **kw,
    ):
        self.num_colours = num_colours
        self.dtype = dtype
        self.hidden_size = hidden_size

        self.encoder = ARCEncoder(
            num_colours=num_colours,
            num_layers=num_layers,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            hidden_size=hidden_size,
            mha_features=mha_features,
            mlp_width_factor=mlp_width_factor,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            rngs=rngs,
            **kw,
        )

        # Classification head (maps the transformer encoder to class probabilities).
        rep = hidden_size.context.rep
        n_base = hidden_size.context.inv
        n_equiv = min(hidden_size.context.equiv, n_base // rep.dim // 2)
        self.equiv_reduction = nnx.Linear(
            in_features=hidden_size.context.equiv,
            out_features=n_equiv,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        n_colour = min(hidden_size.context.equiv, n_base // self.num_colours // 2)
        self.colour_reduction = nnx.Linear(
            in_features=hidden_size.context.inv,
            out_features=n_colour,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # print(f"{n_base=} {n_equiv=} {n_colour=}")
        self.classifier_activation = nnx.gelu
        self.classifier = nnx.Linear(
            n_base + n_equiv * rep.dim + n_colour * self.num_colours,
            num_classes,
            dtype=jnp.promote_types(dtype, jnp.float32),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self, x: jt.Int[jt.Array, "... Y X"], size: jt.Int[jt.Array, "... 2"], **kw
    ) -> jt.Float[jt.Array, "... L"]:
        batch = x.shape[:-2]

        x = self.encoder(x, size, **kw)

        x = x.context

        base = x.inv[..., 0, :]
        equiv = self.equiv_reduction(x.equiv[..., 0, :, :]).reshape(*batch, -1)
        colour = self.colour_reduction(x.inv[..., 1:, :]).reshape(*batch, -1)
        # print(f"{base.shape=} {equiv.shape=} {colour.shape=}")
        x = jnp.concatenate([base, equiv, colour], axis=-1)
        # print(f"{x.shape=}")
        x = self.classifier_activation(x)

        # Predict class probabilities based on the CLS token embedding.
        return self.classifier(x)
