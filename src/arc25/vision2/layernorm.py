import typing
from dataclasses import dataclass

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


@dataclass
class ScaleBias(nnx.Pytree):
    """Container for scale and bias parameters."""

    scale: nnx.Param | None
    bias: nnx.Param | None


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
        norm_per: typing.Literal["all", "rep", "basis"] = "all",
        mode: typing.Literal["flat", "split"] | None = None,
        use_fast_variance: bool = True,
        rngs: rnglib.Rngs,
    ):

        self.features = features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.norm_per = norm_per
        self.use_fast_variance = use_fast_variance
        self.mode = mode

        # Create scale and bias parameters for each representation
        # Invariant: shape (invariant,)
        # Space: shape (space,) - shared across all space basis elements to preserve D4 equivariance
        # Flavour: shape (flavour,) - shared across all flavours to preserve flavour permutation symmetry
        self.params = nnx_compat.Dict(
            {
                k: ScaleBias(
                    scale=(
                        nnx.Param(scale_init(rngs.params(), (v,), param_dtype))
                        if use_scale
                        else None
                    ),
                    bias=(
                        nnx.Param(bias_init(rngs.params(), (v,), param_dtype))
                        if use_bias
                        else None
                    ),
                )
                for k, v in dict(
                    invariant=features.invariant,
                    space=features.space,
                    flavour=features.flavour,
                ).items()
            }
        )

    def _compute_stats(
        self,
        x: jt.Array,
        n_reduction_axes: int = 1,
    ) -> tuple[jt.Array, jt.Array]:
        dtype = jnp.result_type(x) if self.dtype is None else self.dtype
        # promote x to at least float32, this avoids half precision computation
        # but preserves double or complex floating points
        dtype = jnp.promote_types(dtype, jnp.float32)
        x = jnp.asarray(x, dtype)
        axes = tuple(range(-n_reduction_axes, 0))
        mu = x.mean(axis=axes, keepdims=True)
        if self.use_fast_variance:
            mu2 = jax.lax.square(x).mean(axis=axes, keepdims=True)
            # mean2 - jax.lax.square(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - jax.lax.square(mu))
        else:
            var = jax.lax.square(x - mu).mean(axis=axes, keepdims=True)
        return mu, var

    def _normalise(
        self,
        x: jt.Array,
        *,
        mean: jt.Array,
        var: jt.Array,
        scale: jt.Array | None,
        bias: jt.Array | None,
    ):
        y = x - mean
        args = [x]
        mul = jax.lax.rsqrt(var + self.epsilon)
        if scale is not None:
            mul *= scale
            args.append(scale)
        y *= mul
        if bias is not None:
            y += bias
            args.append(bias)
        dtype = nnx.nn.dtypes.canonicalize_dtype(*args, dtype=self.dtype)
        return jnp.asarray(y, dtype)

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

        if self.norm_per == "all":
            # TODO: this one may have an efficient implementation even in "split" mode
            ret = self._apply_flat(x.as_flat())
        else:
            ret = self._apply_split(x.as_split())

        match mode:
            case "flat":
                return ret.as_flat()
            case "split":
                return ret.as_split()
            case _:
                raise KeyError(f"Unsupported mode {mode!r}")

    def _apply_split(self, x: SplitSymDecomp) -> SplitSymDecomp:
        """Apply normalization in split representation mode."""

        # Get dtype for computation
        def normalise(x, params: ScaleBias, n_potential_reduction_axes: int):
            n_reduction_axes = min(
                n_potential_reduction_axes,
                dict(
                    rep=2,
                    basis=1,
                )[self.norm_per],
            )
            mean, var = self._compute_stats(
                x=x,
                n_reduction_axes=n_reduction_axes,
            )
            return self._normalise(
                x, mean=mean, var=var, scale=params.scale, bias=params.bias
            )

        return attrs.evolve(
            x,
            **{
                k: normalise(v, self.params[k], dict(invariant=1).get(k, 2))
                for k, v in x.representations.items()
            },
        )

    def _apply_flat(self, x: FlatSymDecomp) -> FlatSymDecomp:
        """Apply normalization efficiently in flat representation mode."""
        assert self.norm_per == "all"
        mean, var = self._compute_stats(
            x=x.data,
            n_reduction_axes=1,
        )
        rep = self.features.rep
        scale = [] if self.use_scale else None
        bias = [] if self.use_bias else None
        for k in FlatSymDecomp.subrep_seq:
            for kk, vv in dict(scale=scale, bias=bias).items():
                v = getattr(self.params[k], kk)
                if v is None:
                    continue
                n = dict(
                    space=rep.n_space,
                    flavour=rep.n_flavours,
                ).get(k)
                if n is not None:
                    v = jnp.tile(v, n)
                vv.append(v)
        if scale:
            scale = jnp.concatenate(scale)
        if bias:
            bias = jnp.concatenate(bias)

        return attrs.evolve(
            x,
            data=self._normalise(x.data, mean=mean, var=var, scale=scale, bias=bias),
        )
