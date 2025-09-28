import itertools
import typing

import attrs
import jax.numpy as jnp
import jaxtyping as jt
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
from ..symmetry import D4, PermRepBase, SymOpBase
from .symrep import FlatSymDecomp, SplitSymDecomp, SymDecompBase, SymDecompDims


class SpaceSymmetricLinear(nnx.Module):
    r"""A linear operator that is equivariant under point-group operations.
    It represents an operator $A^{uvw\ldots}_{abc\ldots}$ that commutes with
    symmetry operations in the following way:
    $$
    \sum_{u'v'w'\ldots} A^{u'v'w'\ldots}_{abc\ldots} R^U_{u'u}(s) R^V_{v'v}(s) R^W_{w'w}(s)\ldots =
    \sum_{a'b'c'\ldots}  R^A_{a'a}(s) R^B_{b'b}(s) R^C_{c'c}(s)\ldots A^{uvw\ldots}_{a'b'c'\ldots}
    \quad\forall a,b,c,\, u,v,w,\, s.
    $$
    That is, it commutes with the symmetry operations as represented by
    $R^U \otimes R^V \otimes R^W \ldots$ on the input side
    and $R^A \otimes R^B \otimes R^C \ldots$, where each of these representations is a permutation representation.

    The weight format is (N,C,C'), where N enumerates the free subsets of parameters as encountered in
    the usual row-major order among the full (u,v,w,a,b,c) kernel shape, excluding the mandatory last invariant axis.
    The layer computes o(...,a,b,c,...,o) = b(o) + sum_{uvw...i} k(u,v,w,i,a,b,c,o) * i(...,u,v,w,...,i)
    """

    def __init__(
        self,
        in_reps: tuple[PermRepBase, ...],
        in_features: int,
        out_reps: tuple[PermRepBase, ...],
        out_features: int,
        *,
        # in_axes: tuple[int,...] | None = None,
        # out_axes: tuple[int,...] | None = None,
        symmetry_group: type[SymOpBase] = D4,
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
        kw = dict(
            in_reps=in_reps,
            in_features=in_features,
            out_reps=out_reps,
            out_features=out_features,
            symmetry_group=symmetry_group,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dot_general=dot_general,
            promote_dtype=promote_dtype,
        )
        for k, v in kw.items():
            setattr(self, k, v)

        permutation_index, N = self._get_permutation_index(
            in_reps, out_reps, symmetry_group
        )
        self.permutation_index = nnx_compat.static(permutation_index)
        self.bias = (
            nnx.Param(bias_init(rngs.params(), (out_features,), param_dtype))
            if use_bias
            else nnx_compat.data(None)
        )
        self.kernel_params = nnx.Param(
            kernel_init(
                rngs.params(),
                (N, in_features, out_features),
                param_dtype,
            )
        )

    def get_kernel(self, dtype: Dtype | None = None):
        params = self.kernel_params
        if dtype is not None:
            params = params.astype(dtype)
        kernel = params[self.permutation_index, :, :]
        kernel = jnp.moveaxis(kernel, -2, len(self.in_reps))
        return kernel

    _permutation_index_cache: dict[
        tuple[tuple[PermRepBase, ...], tuple[PermRepBase, ...]],
        tuple[jt.Int[np.ndarray, "..."], int],
    ] = {}

    @classmethod
    def _get_permutation_index(
        cls,
        in_reps: tuple[PermRepBase, ...],
        out_reps: tuple[PermRepBase, ...],
        symmetry_group: type[SymOpBase],
    ) -> tuple[jt.Int[np.ndarray, "..."], int]:
        if (ret := cls._permutation_index_cache.get((in_reps, out_reps))) is not None:
            return ret
        # TODO: this is a rather brute-force approach; it should be possible to do better
        reps = in_reps + out_reps
        idxmap = {basis: i for rep in set(reps) for i, basis in enumerate(rep)}
        result_shape = tuple(len(rep) for rep in reps)
        idx = np.arange(np.prod(result_shape), dtype=int).reshape(result_shape)
        for op in symmetry_group:
            if op == symmetry_group.e:
                continue
            iop = op.inverse
            # ensure commutativity by merging all parameters that need to be equal
            for ein in itertools.product(*in_reps):
                iein = tuple(idxmap[b] for b in ein)
                itin = tuple(idxmap[b.apply(iop)] for b in ein)
                for eout in itertools.product(*out_reps):
                    ieout = tuple(idxmap[b] for b in eout)
                    itout = tuple(idxmap[b.apply(op)] for b in eout)
                    # find position in `idx` for A R(s) and R(s) A
                    ia = iein + itout  # A R(s)
                    ib = itin + ieout  # R(s) A
                    if False and idx[ia] != idx[ib]:
                        print(
                            f"{op} merges [{''.join(str(b) for b in ein)},{''.join(str(b.apply(op)) for b in eout)}]{ia}"
                            f" and [{''.join(str(b.apply(iop)) for b in ein)},{''.join(str(b) for b in eout)}]{ib}"
                        )
                    # merge their trees
                    idx[ia] = idx[ib] = min(idx[ia], idx[ib])
        # now, we need to simplify all remaining trees into "bushes": each leaf is a direct child of their root;
        # at the same time, we also reindex the roots into a new consecutive sequence of tree indexes
        idx = idx.ravel()
        count = 0
        for i, v in enumerate(idx):
            assert v <= i
            if v < i:
                idx[i] = idx[v]
                continue
            # this is a root:
            idx[i] = count
            count += 1
        idx = idx.reshape(result_shape)
        ret = (idx, count)
        cls._permutation_index_cache[in_reps, out_reps] = ret
        return ret


class SymDecompLinear(nnx.Module):
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
        inf = in_features
        outf = out_features

        R = min(inf.rep.dim, outf.rep.dim)
        kw = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dot_general=dot_general,
            promote_dtype=promote_dtype,
        )
        self.bias = {
            k: (
                nnx.Param(bias_init(rngs.params(), (v,), param_dtype))
                if use_bias and v
                else nnx_compat.data(None)
            )
            for k, v in outf.representations.items()
        }
        self.invariant = {
            (ki, ko): (
                nnx.Linear(
                    vi,
                    vo,
                    use_bias=False,
                    **kw,
                    rngs=rngs,
                )
                if vi and vo
                else nnx_compat.data(None)
            )
            for ki, vi in inf.representations.items()
            for ko, vo in outf.representations.items()
            if (ki, ko)
            not in {("space", "space"), ("space", "flavour"), ("space", "flavour")}
        }
        vi, vo = inf.flavour, outf.flavour
        self.flavour_pointwise = (
            nnx.Linear(
                vi,
                vo,
                use_bias=False,
                **kw,
                rngs=rngs,
            )
            if vi and vo
            else nnx_compat.data(None)
        )
        self.space2space = (
            nnx.Param(
                kernel_init(
                    rngs.params(),
                    (in_features.equiv, R, out_features.equiv),
                    param_dtype,
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
        ret = sum(v.size for v in self.bias.values()) if self.bias is not None else 0
        for lin in self.parts.values():
            if lin is None:
                continue
            ret += lin.kernel.size
        if self.space2space is not None:
            kernel = self._prepare_kernel(self.space2space)
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

        R = ri.n_space  # noqa: F841
        C = fi.space  # noqa: F841
        Rp = ro.n_space  # noqa: F841
        Cp = fo.space  # noqa: F841

        mode = self.constraint_mode

        # this is that |o>
        ref = ri.repseq[0]

        ret = []
        for u in ro.repseq:
            # apply the representation
            ki = np.array(
                [ro.rep2idx[u.transform(v.tfo_to(ref))] for v in ri.opseq], "i4"
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

    def __call__(
        self,
        inputs: SymDecompBase,
        *,
        mode: typing.Literal["flat", "split"] | None = None,
    ) -> SymDecompBase:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        assert self.in_features.validate(inputs), self.in_features.validation_problems(
            inputs
        )

        if mode is None:
            match inputs:
                case FlatSymDecomp():
                    mode = "flat"
                case SplitSymDecomp():
                    mode = "split"
                case _:
                    raise TypeError(
                        f"Unsupported symmetry decomposition {type(inputs).__name__}"
                    )
        match mode:
            case "flat":
                ret = self._apply_flat(inputs.as_flat(dtype=self.dtype))
            case "split":
                ret = self._apply_split(inputs.as_split())
            case _:
                raise KeyError(f"Unsupported mode {mode!r}")

        assert self.out_features.validate(ret), self.out_features.validation_problems(
            ret
        )
        return ret

    def _apply_flat(self, inputs: FlatSymDecomp) -> FlatSymDecomp:
        raise NotImplementedError

    def _apply_split(self, inputs: SplitSymDecomp) -> SplitSymDecomp:
        to_consider = list(inputs.representations.values())
        if self.space2space is not None:
            to_consider.append(self.space2space.value)
        for v in self.parts.values():
            if v is not None:
                to_consider.append(v.value)
        for v in self.bias.values():
            if v is not None:
                to_consider.append(v.value)
        dtype = nnx.nn.dtypes.canonicalize_dtype(*to_consider, dtype=self.dtype)

        r = inputs.representations
        inputs = attrs.evolve(
            inputs,
            **dict(zip(r.keys(), self.promote_dtype(r.values(), dtype=dtype))),
        )
        batch = inputs.batch_shape
        xi = inputs.invariant
        xs = inputs.space
        xf = inputs.flavour
        xsa = jnp.mean(xs, axis=-2)
        xfa = jnp.mean(xf, axis=-2)

        ainp = dict(
            invariant=xi,
            space=xsa,
            flavour=xfa,
        )
        aout = {k: v for k, v in self.bias.items() if v is not None}
        for (ki, ko), lin in self.parts.items():
            if lin is None:
                continue
            y = lin(ainp[ki])
            if (prev := aout.get(ko)) is not None:
                y = prev + y
            aout[ko] = y

        out = dict(
            invariant=aout.pop(
                "invariant",
                jnp.zeros(batch + (self.in_features.invariant,), dtype=dtype),
            ),
        )

        if self.flavour2flavour is not None:
            out["flavour"] = self.flavour2flavour(xf)

        if self.space2space is not None:
            # TODO: should the preparation be applied before promotion instead?
            (kernel_base,) = self.promote_dtype((self.space2space.value,), dtype=dtype)
            kernel = self._prepare_kernel(kernel_base)

            out["space"] = self.dot_general(
                xs,
                kernel,
                (((xs.ndim - 2, xs.ndim - 1), (0, 1)), ((), ())),
                precision=self.precision,
            )

        of = self.out_features
        for k, v in dict(flavour=of.n_flavours, space=of.n_space).items():
            if (y := aout.get(k)) is not None:
                y = jnp.tile(y[..., None, :], (v, 1))
            if (n := out.get(k)) is not None:
                y = n if y is None else n + y
            if y is None:
                y = jnp.zeros(batch + (v, getattr(of, k)), dtype=dtype)
            out[k] = y

        return SplitSymDecomp(**out, rep=of.rep)
