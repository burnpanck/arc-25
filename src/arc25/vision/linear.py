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

from ..lib import nnx_compat
from .symrep import SymDecomp, SymDecompDims


class SymmetricLinear(nnx.Module):
    """Representation-elemet-wise pointwise operation across symmetry and groups of channels; respecting symmetry.

    Weight format is (C,R,C')
    The layer computes o(n,h,w,r',c') = b(c') + sum_cr k(c,r^-1.r',c') * i(n,h,w,r,c)
    """

    def __init__(
        self,
        in_features: SymDecompDims,
        out_features: SymDecompDims,
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
        )
        param_key = rngs.params()
        self.inv_bias = (
            nnx.Param(bias_init(param_key, (out_features.inv,), param_dtype))
            if use_bias and out_features.inv
            else nnx_compat.data(None)
        )
        param_key = rngs.params()
        self.equiv_bias = (
            nnx.Param(bias_init(param_key, (out_features.equiv,), param_dtype))
            if use_bias and out_features.equiv
            else nnx_compat.data(None)
        )
        self.inv2inv = (
            nnx.Linear(
                in_features.inv,
                out_features.inv,
                use_bias=False,
                **kw,
                rngs=rngs,
            )
            if in_features.inv and out_features.inv
            else nnx_compat.data(None)
        )
        self.inv2equiv = (
            nnx.Linear(
                in_features.inv,
                out_features.equiv,
                use_bias=False,
                **kw,
                rngs=rngs,
            )
            if in_features.inv and out_features.equiv
            else nnx_compat.data(None)
        )
        self.equiv2inv = (
            nnx.Linear(
                in_features.equiv,
                out_features.inv,
                use_bias=False,
                **kw,
                rngs=rngs,
            )
            if in_features.equiv and out_features.inv
            else nnx_compat.data(None)
        )
        kernel_key = rngs.params()
        self.equiv2equiv = (
            nnx.Param(
                kernel_init(
                    kernel_key, (in_features.equiv, R, out_features.equiv), param_dtype
                )
            )
            if in_features.equiv and out_features.equiv
            else nnx_compat.data(None)
        )

        self.in_features = in_features
        self.out_features = out_features
        self.constraint_mode = dict(
            gtc="gather-then-concat", ctg="concat-then-gather"
        ).get(constraint_mode, constraint_mode)
        self.use_bias = use_bias
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def approximate_flops(self):
        ret = 0
        for lin in [self.inv2inv, self.equiv2inv, self.inv2equiv]:
            if lin is None:
                continue
            ret += lin.kernel.size
        if self.equiv2equiv is not None:
            kernel = self._prepare_kernel(self.equiv2equiv)
            ret += kernel.size
        return ret

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
        C = fi.equiv  # noqa: F841
        Rp = ro.dim  # noqa: F841
        Cp = fo.equiv  # noqa: F841

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

    def __call__(self, inputs: SymDecomp) -> SymDecomp:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        assert self.in_features.validate(inputs), self.in_features.validation_problems(
            inputs
        )

        to_consider = [inputs.inv, inputs.equiv]
        if self.equiv2equiv is not None:
            to_consider.append(self.equiv2equiv.value)
        if self.inv_bias is not None:
            to_consider.append(self.inv_bias.value)
        if self.equiv_bias is not None:
            to_consider.append(self.equiv_bias.value)
        dtype = nnx.nn.dtypes.canonicalize_dtype(*to_consider, dtype=self.dtype)

        xi, xf = self.promote_dtype((inputs.inv, inputs.equiv), dtype=dtype)
        batch = jnp.broadcast_shapes(xi.shape[:-1], xf.shape[:-2])

        xfa = jnp.mean(xf, axis=-2)

        of = self.out_features
        yi = [
            lin(inp)
            for lin, inp in [(self.inv2inv, xi), (self.equiv2inv, xfa)]
            if lin is not None
        ] + [
            self.promote_dtype((b,), dtype=dtype)[0]
            for b in [self.inv_bias]
            if b is not None
        ]
        yi = sum(yi) if yi else jnp.empty(batch + (of.inv,), dtype)
        yi = jnp.broadcast_to(yi, batch + (of.inv,))

        if self.equiv2equiv is not None:
            # TODO: should the preparation be applied before promotion instead?
            (kernel_base,) = self.promote_dtype((self.equiv2equiv.value,), dtype=dtype)
            kernel = self._prepare_kernel(kernel_base)

            yf = self.dot_general(
                xf,
                kernel,
                (((xf.ndim - 2, xf.ndim - 1), (0, 1)), ((), ())),
                precision=self.precision,
            )
        else:
            yf = None
        if self.inv2equiv is not None:
            yfa = self.inv2equiv(xi)[..., None, :]
        else:
            yfa = None
        yf = [v for v in [yf, yfa] if v is not None]
        if self.equiv_bias is not None:
            yf.append(self.promote_dtype((self.equiv_bias,), dtype=dtype)[0])
        yf = sum(yf) if yf else jnp.zeros(batch + (of.rep.dim, of.equiv), dtype)
        yf = jnp.broadcast_to(yf, batch + (of.rep.dim, of.equiv))

        ret = SymDecomp(inv=yi, equiv=yf, rep=of.rep)
        assert self.out_features.validate(ret), self.out_features.validation_problems(
            ret
        )
        return ret
