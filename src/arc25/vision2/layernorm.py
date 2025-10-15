import typing

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import nnx
from flax.nnx import rnglib
from flax.typing import Dtype, Initializer

from ..lib import nnx_compat
from ..lib.compat import Self
from ..lib.misc import first_from
from .symrep import FlatSymDecomp, SplitSymDecomp, SymDecompBase, SymDecompDims


class SymDecompLayerNorm(nnx.Module):
    """Layer normalization that respects symmetry decomposition.

    Normalizes each representation (invariant, space, flavour) separately
    with its own scale and bias parameters. Supports both flat and split
    representations efficiently.
    """

    def __init__(
        self,
        features: SymDecompDims,
        *,
        epsilon: float = 1e-6,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        use_bias: bool = True,
        use_scale: bool = True,
        bias_init: Initializer = nnx.initializers.zeros,
        scale_init: Initializer = nnx.initializers.ones,
        mode: typing.Literal["flat", "split"] | None = None,
        rngs: rnglib.Rngs,
    ):
        self.features = features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.mode = mode

        # Create scale and bias parameters for each representation
        def make_param(init_fn, shape, name_suffix: str):
            if (
                not shape
                or (name_suffix == "scale" and not use_scale)
                or (name_suffix == "bias" and not use_bias)
            ):
                return None
            return nnx.Param(init_fn(rngs.params(), shape, param_dtype))

        # Invariant: shape (invariant,)
        self.scale_invariant = make_param(
            scale_init, (features.invariant,) if features.invariant else (), "scale"
        )
        self.bias_invariant = make_param(
            bias_init, (features.invariant,) if features.invariant else (), "bias"
        )

        # Space: shape (space,) - shared across all space basis elements to preserve D4 equivariance
        self.scale_space = make_param(
            scale_init,
            (features.space,) if features.space else (),
            "scale",
        )
        self.bias_space = make_param(
            bias_init,
            (features.space,) if features.space else (),
            "bias",
        )

        # Flavour: shape (flavour,) - shared across all flavours to preserve flavour permutation symmetry
        self.scale_flavour = make_param(
            scale_init,
            (features.flavour,) if features.flavour else (),
            "scale",
        )
        self.bias_flavour = make_param(
            bias_init,
            (features.flavour,) if features.flavour else (),
            "bias",
        )

    def _normalize(
        self,
        x: jt.Array,
        scale: jt.Array | None,
        bias: jt.Array | None,
        reduction_axes: tuple[int, ...],
    ) -> jt.Array:
        """Normalize array over specified axes with optional scale and bias."""
        # Compute statistics
        mean = jnp.mean(x, axis=reduction_axes, keepdims=True)
        var = jnp.var(x, axis=reduction_axes, keepdims=True)

        # Normalize
        x = (x - mean) * jax.lax.rsqrt(var + self.epsilon)

        # Apply scale and bias
        if scale is not None:
            x = x * scale
        if bias is not None:
            x = x + bias

        return x

    def __call__(
        self,
        x: SymDecompBase,
        *,
        mode: typing.Literal["flat", "split"] | None = None,
    ) -> SymDecompBase:
        """Apply layer normalization to symmetry-decomposed input.

        Args:
            x: Input with symmetry decomposition
            mode: Whether to use "flat" or "split" computation (defaults to input type)

        Returns:
            Normalized output with same type as input
        """
        assert self.features.validate(x), self.features.validation_problems(x)

        mode = first_from(mode, self.mode)
        if mode is None:
            match x:
                case FlatSymDecomp():
                    mode = "flat"
                case SplitSymDecomp():
                    mode = "split"
                case _:
                    raise TypeError(
                        f"Unsupported symmetry decomposition {type(x).__name__}"
                    )

        match mode:
            case "flat":
                return self._apply_flat(x.as_flat())
            case "split":
                return self._apply_split(x.as_split())
            case _:
                raise KeyError(f"Unsupported mode {mode!r}")

    def _apply_split(self, x: SplitSymDecomp) -> SplitSymDecomp:
        """Apply normalization in split representation mode."""
        # Get dtype for computation
        dtype = self.dtype or x.invariant.dtype

        # Normalize each representation separately
        # Invariant: normalize over last axis (channel dimension)
        invariant = (
            self._normalize(
                x.invariant.astype(dtype),
                (
                    self.scale_invariant.value
                    if self.scale_invariant is not None
                    else None
                ),
                self.bias_invariant.value if self.bias_invariant is not None else None,
                reduction_axes=(-1,),
            )
            if self.features.invariant
            else x.invariant
        )

        # Space: normalize over last two axes (n_space, space)
        # Scale and bias have shape (space,), need to broadcast across n_space
        space_scale = None
        space_bias = None
        if self.features.space:
            if self.scale_space is not None:
                space_scale = self.scale_space.value[None, :]  # (1, space)
            if self.bias_space is not None:
                space_bias = self.bias_space.value[None, :]  # (1, space)
            space = self._normalize(
                x.space.astype(dtype),
                space_scale,
                space_bias,
                reduction_axes=(-2, -1),
            )
        else:
            space = x.space

        # Flavour: normalize over last two axes (n_flavours, flavour)
        # Scale and bias have shape (flavour,), need to broadcast across n_flavours
        flavour_scale = None
        flavour_bias = None
        if self.features.flavour:
            if self.scale_flavour is not None:
                flavour_scale = self.scale_flavour.value[None, :]  # (1, flavour)
            if self.bias_flavour is not None:
                flavour_bias = self.bias_flavour.value[None, :]  # (1, flavour)
            flavour = self._normalize(
                x.flavour.astype(dtype),
                flavour_scale,
                flavour_bias,
                reduction_axes=(-2, -1),
            )
        else:
            flavour = x.flavour

        return SplitSymDecomp(
            invariant=invariant,
            space=space,
            flavour=flavour,
            rep=x.rep,
        )

    def _apply_flat(self, x: FlatSymDecomp) -> FlatSymDecomp:
        """Apply normalization efficiently in flat representation mode."""
        # Get dtype for computation
        dtype = self.dtype or x.data.dtype
        data = x.data.astype(dtype)

        # Compute slice boundaries for each representation
        # Order: invariant, space, flavour (see FlatSymDecomp.subrep_seq)
        feat = self.features
        n_inv = feat.invariant
        n_space = feat.n_space * feat.space
        n_flavour = feat.n_flavours * feat.flavour

        # Split data into representations
        batch_shape = x.batch_shape
        inv_data = data[..., :n_inv] if n_inv else None
        space_data = data[..., n_inv : n_inv + n_space] if n_space else None
        flavour_data = (
            data[..., n_inv + n_space : n_inv + n_space + n_flavour]
            if n_flavour
            else None
        )

        parts = []

        # Normalize invariant
        if inv_data is not None:
            inv_norm = self._normalize(
                inv_data,
                (
                    self.scale_invariant.value.reshape(-1)
                    if self.scale_invariant is not None
                    else None
                ),
                (
                    self.bias_invariant.value.reshape(-1)
                    if self.bias_invariant is not None
                    else None
                ),
                reduction_axes=(-1,),
            )
            parts.append(inv_norm)

        # Normalize space
        if space_data is not None:
            # Reshape to expose (n_space, space) structure for normalization
            space_reshaped = space_data.reshape(*batch_shape, feat.n_space, feat.space)
            # Scale and bias have shape (space,), need to broadcast across n_space
            space_scale = None
            space_bias = None
            if self.scale_space is not None:
                space_scale = self.scale_space.value[None, :]  # (1, space)
            if self.bias_space is not None:
                space_bias = self.bias_space.value[None, :]  # (1, space)

            space_norm = self._normalize(
                space_reshaped,
                space_scale,
                space_bias,
                reduction_axes=(-2, -1),
            )
            # Flatten back
            space_norm = space_norm.reshape(*batch_shape, n_space)
            parts.append(space_norm)

        # Normalize flavour
        if flavour_data is not None:
            # Reshape to expose (n_flavours, flavour) structure for normalization
            flavour_reshaped = flavour_data.reshape(
                *batch_shape, feat.n_flavours, feat.flavour
            )
            # Scale and bias have shape (flavour,), need to broadcast across n_flavours
            flavour_scale = None
            flavour_bias = None
            if self.scale_flavour is not None:
                # Expand to (1, flavour) for broadcasting
                flavour_scale = self.scale_flavour.value[None, :]
            if self.bias_flavour is not None:
                flavour_bias = self.bias_flavour.value[None, :]

            flavour_norm = self._normalize(
                flavour_reshaped,
                flavour_scale,
                flavour_bias,
                reduction_axes=(-2, -1),
            )
            # Flatten back
            flavour_norm = flavour_norm.reshape(*batch_shape, n_flavour)
            parts.append(flavour_norm)

        # Concatenate all parts back together
        normalized_data = jnp.concatenate(parts, axis=-1) if parts else data

        return FlatSymDecomp(data=normalized_data, dim=feat)
