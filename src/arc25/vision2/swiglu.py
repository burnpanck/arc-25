import typing
from types import SimpleNamespace

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn.linear import default_bias_init, default_kernel_init, initializers
from flax.typing import (
    Dtype,
    Initializer,
    PrecisionLike,
)

from .. import symmetry
from ..dsl.types import Vector
from ..lib import nnx_compat
from ..lib.attrs import AttrsModel
from ..symmetry import transform_vector
from .linear import SpaceSymmetricTensor, SymDecompLinear
from .symrep import FlatSymDecomp, SplitSymDecomp, SymDecompBase, SymDecompDims


class SwiGLU(nnx.Module):
    def __init__(
        self,
        in_features: SymDecompDims,
        out_features: SymDecompDims | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        activation: typing.Any = jax.nn.silu,
        hidden_features: SymDecompDims | None = None,
        linear_mode: typing.Literal["flat", "split"] | None = None,
        width_factor: float | None = None,
        dropout_rate: float = 0.0,
        keep_rngs: bool = True,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ):
        assert (hidden_features is None) or (
            width_factor is None
        ), "Need at most one of `hidden_features` or `width_factor`"
        if out_features is None:
            out_features = in_features
        if hidden_features is None:
            assert (
                out_features.rep == in_features.rep
            ), "width_factor only works for matching reps"
            if width_factor is None:
                width_factor = 8 / 3
            hidden_features = in_features.map_representations(
                lambda k, v: int(
                    round(np.sqrt(v * getattr(out_features, k)) * width_factor)
                )
            )
        self.in_features = in_features
        self.out_features = out_features
        if False:
            self.dtype = dtype
            self.param_dtype = param_dtype
            self.precision = precision
            self.hidden_features = hidden_features
            self.linear_mode = linear_mode
        self.activation = activation

        def make_linear(in_feat: SymDecompDims, out_feat: SymDecompDims, **kw):
            return SymDecompLinear(
                in_feat,
                out_feat,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                use_bias=use_bias,
                mode=linear_mode,
                rngs=rngs,
            )

        self.up = make_linear(
            in_features, hidden_features.map_representations(lambda k, v: 2 * v)
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs if keep_rngs else None)
        self.down = make_linear(hidden_features, out_features)

    def __call__(
        self,
        x: SymDecompBase,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
    ) -> SymDecompBase:
        x = self.up(x, mode=mode)
        u = x.map_elementwise(
            lambda a: a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)[..., 0]
        )
        v = x.map_elementwise(
            lambda a: a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)[..., 1]
        )
        x = u.map_elementwise(lambda u, v: self.activation(u) * v, v)
        x = x.map_elementwise(self.dropout, rngs=rngs, deterministic=deterministic)
        x = self.down(x, mode=mode)
        return x
