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
from .encoder import ARCEncoder
from .fields import Field, FieldDims
from .linear import SymDecompLinear
from .symrep import SymDecompBase, SymDecompDims, standard_rep
from .transformer import FieldTransformer


class ARCClassifier(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: FieldDims,
        num_classes: int = 1000,
        num_perceiver_tokens: int = 24,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        rngs: nnx.Rngs,
        **kw,
    ):
        self.dtype = dtype
        self.hidden_size = hidden_size

        self.encoder = ARCEncoder(
            dtype=dtype,
            hidden_size=hidden_size,
            param_dtype=param_dtype,
            precision=precision,
            num_perceiver_tokens=num_perceiver_tokens,
            rngs=rngs,
            **kw,
        )

        # Classification head (maps the transformer encoder to class probabilities).
        dims = hidden_size.context
        n_base = min(dims.invariant, 2 * num_classes // num_perceiver_tokens)
        mult = dict(invariant=1, space=dims.rep.n_space, flavour=dims.rep.n_flavours)
        reduced_dims = {
            k: min(getattr(dims, k), n_base // (v * (2 if k != "invariant" else 1)))
            for k, v in mult.items()
        }
        self.reduction = nnx_compat.Dict(
            {
                k: nnx.Linear(
                    in_features=getattr(dims, k),
                    out_features=v,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for k, v in reduced_dims.items()
            }
        )

        self.classifier_activation = nnx.gelu
        self.classifier = nnx.Linear(
            num_perceiver_tokens * sum(v * mult[k] for k, v in reduced_dims.items()),
            num_classes,
            dtype=(
                jnp.promote_types(dtype, jnp.float32)
                if dtype is not None
                else jnp.float32
            ),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        transform: jt.Int[jt.Array, "..."] | None = None,
        **kw,
    ) -> jt.Float[jt.Array, "... L"]:
        """When `transform` is not None, it is interpreted as an index into
        the standard operation sequence of D4, to be applied to the image
        before classifying. The implementation relies on the D4-equivariance
        of the encoder and applies it to the CLS token just before
        entering the final classifier (which intentionally breaks D4-equivariance).
        This is mostly relevant with respect to transposes,
        as this enables bucketing tall with wide images.
        """
        batch = x.shape[:-2]

        x = self.encoder(x, size, **kw)

        x = x.context

        if transform is not None:
            x = x.as_split()
            # Apply D4 operation
            basis_lookup = jnp.array(
                [
                    symmetry.transform_coeff_in_basis(op, x.rep.space)
                    for op in symmetry.D4
                ]
            )
            idx = basis_lookup[transform, :]
            # x.space has dimensions (*batch, T, R, C), while index for now is (*batch, R)
            idx = idx[..., None, :, None]
            x = attrs.evolve(x, space=jnp.take_along_axis(x.space, idx, axis=-2))

        # print(x.shapes)
        x = jnp.concatenate(
            [v(getattr(x, k)).reshape(*batch, -1) for k, v in self.reduction.items()],
            axis=-1,
        )
        # print(f"{x.shape=}")

        x = self.classifier_activation(x)

        # Predict class probabilities based on the CLS token embedding.
        return self.classifier(x)
