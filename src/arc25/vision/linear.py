import typing

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.nnx.nn.linear import default_bias_init, default_kernel_init
from flax.typing import (
    DotGeneralT,
    Dtype,
    Initializer,
    PrecisionLike,
    PromoteDtypeFn,
)
from jax import lax

from .symrep import Embedding, EmbeddingDims


class SymmetricLinear(nnx.Module):
    """Representation-elemet-wise pointwise operation across symmetry and groups of channels; respecting symmetry.

    Weight format is (C,R,C')
    The layer computes o(n,h,w,r',c') = b(c') + sum_cr k(c,r^-1.r',c') * i(n,h,w,r,c)
    """

    def __init__(
        self,
        in_features: EmbeddingDims,
        out_features: EmbeddingDims,
        *,
        constraint_mode: typing.Literal[
            "gather-then-concat", "concat-then-gather"
        ] = "concat-then-gather",
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        dot_general: DotGeneralT = lax.dot_general,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        preferred_element_type: Dtype | None = None,
        rngs: rnglib.Rngs,
    ):
        R = min(in_features.rep.dim, out_features.rep.dim)
        kw = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dot_general=dot_general,
            promote_dtype=promote_dtype,
            # preferred_element_type = preferred_element_type,
        )
        self.iso2iso = (
            nnx.Linear(
                in_features.iso,
                out_features.iso,
                use_bias=use_bias,
                **kw,
                rngs=rngs,
            )
            if in_features.iso and out_features.iso
            else nnx.data(None)
        )
        self.iso2full = (
            nnx.Linear(
                in_features.iso,
                out_features.full,
                use_bias=use_bias,
                **kw,
                rngs=rngs,
            )
            if in_features.iso and out_features.full
            else nnx.data(None)
        )
        self.full2iso = (
            nnx.Linear(
                in_features.full,
                out_features.iso,
                use_bias=False,
                **kw,
                rngs=rngs,
            )
            if in_features.full and out_features.iso
            else nnx.data(None)
        )
        kernel_key = rngs.params()
        self.full2full = (
            nnx.Param(
                kernel_init(
                    kernel_key, (in_features.full, R, out_features.full), param_dtype
                )
            )
            if in_features.full and out_features.full
            else nnx.data(None)
        )

        self.in_features = in_features
        self.out_features = out_features
        self.constraint_mode = dict(
            gtc="gather-then-concat", ctg="concat-then-gather"
        ).get(constraint_mode, constraint_mode)
        self.use_bias = use_bias
        for k, v in kw.items():
            setattr(self, k, v)

    def _prepare_kernel(self, kernel):
        r"""
        The layer computes o(n,h,w,r',c') = sum_mr k(c,r^-1.r',c') * i(n,h,w,r,c)
        This is re-cast into a "standard" linear as o'(n;h;w;r',c') = sum_mr i'(n;h;w;r,c)*k'(r,c;r',c')

        Thus, we return a kernel of shape (R,C,R',C')

        We want the kernel to be symmetry-invariant; that is \rho(s) Y = \rho(s) K X = K \rho(s) X.
        Let X = \sum_v X_v |v>, such that \rho(s) X = \sum_v X_v |s.v>
        Then Y = \sum_uv K_uv X_v |u>
        Thus \rho(s) Y = \sum_uv K_uv X_v |s.u> = \sum_uvw K_uw X_v |u><w|s.v>
        -> <s.u|K|v> = <u|K|s.v> for all u,v,s
        We will select an arbitrary channel o, keep <u|K|o> as the kernel element u
        and derive the rest from it.
        -> <u|K|v> = <o.v^-1.u|K|o> for all u,v
        """
        fi = self.in_features
        fo = self.out_features
        ri = fi.rep
        ro = fo.rep

        assert (
            ri == ro
        ), "Not sure if the representation logic below is correct for representation changes"

        R = ri.dim  # noqa: F841
        C = fi.full  # noqa: F841
        Rp = ro.dim  # noqa: F841
        Cp = fo.full  # noqa: F841

        mode = self.constraint_mode

        # this is that |o>
        refop = ri.opseq[0]

        ret = []
        for u in ro.opseq:
            # apply the representation
            ki = np.array(
                [ro.op2idx[refop.combine(v.inverse).combine(u)] for v in ri.opseq], "i4"
            )
            # reshape into the format we need
            if mode == "gather-then-concat":
                k = kernel[None, :, ki, :]
                ret.append(k)
            elif mode == "concat-then-gather":
                ret.append(ki)
            else:
                raise ValueError(mode)

        match mode:
            case "gather-then-concat":
                ret = jnp.concatenate(ret, axis=-4)
            case "concat-then-gather":
                ki = np.array(ret)
                k = kernel[:, ki, :]
                k = k.transpose(1, 0, 2, 3)
                ret = k
            case _:
                raise KeyError(mode)

        return ret

    def __call__(self, inputs: Embedding) -> Embedding:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        assert self.in_features.validate(inputs), self.in_features.validation_problems(
            inputs
        )

        xi, xf = self.promote_dtype((inputs.iso, inputs.full), dtype=self.dtype)

        xfa = jnp.mean(xf, axis=-2)

        of = self.out_features
        yi = [
            lin(inp)
            for lin, inp in [(self.iso2iso, xi), (self.full2iso, xfa)]
            if lin is not None
        ]
        yi = sum(yi) if yi else jnp.empty(xi.shape[:-1] + (of.iso,), xi.dtype)

        if self.full2full is not None:
            # TODO: should the preparation be applied before promotion instead?
            kernel_base = self.full2full.value
            kernel_base = self.promote_dtype(kernel_base, dtype=self.dtype)
            kernel = self._prepare_kernel(kernel_base)

            # We use dot_general_kwargs for BC compatibility with
            # user custom self.dot_general method which may not have
            # preferred_element_type argument to avoid breaking
            # existing code
            dot_general_kwargs = {}
            if False and self.preferred_element_type is not None:
                dot_general_kwargs["preferred_element_type"] = (
                    self.preferred_element_type
                )
            yf = self.dot_general(
                xf,
                kernel,
                (((xf.ndim - 2, xf.ndim - 1), (0, 1)), ((), ())),
                precision=self.precision,
                **dot_general_kwargs,
            )
        else:
            yf = None
        if self.iso2full is not None:
            yfa = self.iso2full(xi)[..., None, :]
            if yf is not None:
                yf = yf + yfa
            else:
                yf = jnp.tile(yfa, (of.rep.dim, 1))
        elif yf is None:
            yf = jnp.empty(xf.shape[:-2] + (of.rep.dim, 0), self.dtype)
        ret = Embedding(iso=yi, full=yf, rep=of.rep)
        assert self.out_features.validate(ret), self.out_features.validation_problems(
            ret
        )
        return ret
