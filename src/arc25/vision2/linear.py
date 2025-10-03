import itertools
import typing

import attrs
import jax
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
from ..lib.compat import Self
from ..symmetry import D4, PermRepBase, SymOpBase
from .symrep import FlatSymDecomp, SplitSymDecomp, SymDecompBase, SymDecompDims


@attrs.frozen
class AxisSymmetryInfo:
    rep: type[PermRepBase] = attrs.field(repr=lambda v: v.__name__)
    covariant: bool


@attrs.frozen
class SymmetryMappingSpec:
    symmetry_group: type[SymOpBase]
    axes: tuple[AxisSymmetryInfo, ...]
    permutation_index: np.ndarray = attrs.field(repr=lambda v: f"{v.shape}")
    n_free_subspaces: int
    fully_independent: bool
    fully_coupled: bool

    _cache: typing.ClassVar[dict[Self, Self]] = {}

    @classmethod
    def get(
        cls,
        axes: tuple[AxisSymmetryInfo, ...],
        symmetry_group: type[SymOpBase],
    ) -> Self:
        key = cls(
            symmetry_group=symmetry_group,
            axes=axes,
            permutation_index=None,
            n_free_subspaces=None,
            fully_coupled=None,
            fully_independent=None,
        )
        if (ret := cls._cache.get(key)) is not None:
            return ret
        # TODO: this is a rather brute-force approach; it should be possible to do better
        reps = tuple(ax.rep for ax in axes)
        cov_reps = tuple(ax.rep for ax in axes if ax.covariant)
        cont_reps = tuple(ax.rep for ax in axes if not ax.covariant)

        def combine(
            cov_idx: tuple[int, ...], cont_idx: tuple[int, ...]
        ) -> tuple[int, ...]:
            ret = []
            ncov = 0
            ncont = 0
            for ax in axes:
                if ax.covariant:
                    ret.append(cov_idx[ncov])
                    ncov += 1
                else:
                    ret.append(cont_idx[ncont])
                    ncont += 1
            assert ncov == len(cov_idx)
            assert ncont == len(cont_idx)
            return tuple(ret)

        idxmap = {basis: i for rep in set(reps) for i, basis in enumerate(rep)}
        result_shape = tuple(len(rep) for rep in reps)
        idx = np.arange(np.prod(result_shape), dtype=int).reshape(result_shape)
        for op in symmetry_group:
            if op == symmetry_group.e:
                continue
            iop = op.inverse
            # ensure commutativity by merging all parameters that need to be equal
            for ein in itertools.product(*cov_reps):
                iein = tuple(idxmap[b] for b in ein)
                itin = tuple(idxmap[b.apply(iop)] for b in ein)
                for eout in itertools.product(*cont_reps):
                    ieout = tuple(idxmap[b] for b in eout)
                    itout = tuple(idxmap[b.apply(op)] for b in eout)
                    # find position in `idx` for A R(s) and R(s) A
                    ia = combine(iein, itout)  # A R(s)
                    ib = combine(itin, ieout)  # R(s) A
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
        ret = SymmetryMappingSpec(
            symmetry_group=symmetry_group,
            axes=axes,
            permutation_index=idx,
            n_free_subspaces=count,
            fully_coupled=count == 1,
            fully_independent=count == idx.size,
        )
        cls._cache[key] = ret
        return ret


class SpaceSymmetricTensor(nnx.Module):
    r"""
    Represents a tensor living in a tensor-product space,
    where each tensor axis transforms according to a well-specified representation.
    We currently only allow "basic" permutation representations
    which have a fixed size, plus the trivial representation of arbitrary size.
    """

    def __init__(
        self,
        representations: tuple[type[PermRepBase] | int, ...],
        covariant: int | tuple[bool, ...] = 0,
        *,
        # in_axes: tuple[int,...] | None = None,
        # out_axes: tuple[int,...] | None = None,
        symmetry_group: type[SymOpBase] = D4,
        dtype: Dtype = jnp.float32,
        init: Initializer = default_kernel_init,
        rngs: rnglib.Rngs,
    ):
        if not isinstance(covariant, tuple):
            covariant = (True,) * covariant + (False,) * (
                len(representations) - covariant
            )
        kw = dict(
            representations=representations,
            covariant=covariant,
            symmetry_group=symmetry_group,
            dtype=dtype,
            init=init,
        )
        for k, v in kw.items():
            setattr(self, k, v)

        assert len(representations) == len(covariant)
        axis_info = []
        trivial_shape = []
        transpose_dims = []
        for rep, cov in zip(representations, covariant):
            if isinstance(rep, int):
                # trivial dimension
                transpose_dims.append((1, len(trivial_shape)))
                trivial_shape.append(rep)
            else:
                transpose_dims.append((0, len(axis_info)))
                axis_info.append(AxisSymmetryInfo(rep, cov))
        transpose_dims = tuple(len(axis_info) * i + j for i, j in transpose_dims)
        mapping_info = SymmetryMappingSpec.get(tuple(axis_info), symmetry_group)
        self.mapping_info = nnx_compat.static(mapping_info)
        self.transpose_dims = transpose_dims
        self.trivial_shape = tuple(trivial_shape)
        self.params = nnx.Param(
            (init if all(trivial_shape) else jax.nn.initializers.zeros)(
                rngs.params(),
                (mapping_info.n_free_subspaces,) + tuple(trivial_shape),
                dtype,
            )
        )

    def get_tensor(self, dtype: Dtype | None = None):
        info = self.mapping_info
        params = self.params
        if dtype is not None:
            params = params.astype(dtype)
        idx = info.permutation_index
        shape = idx.shape
        if info.fully_coupled:
            # instead of a constant gather, use a tile
            assert np.all(idx == 0)
            ret = jnp.tile(params[0], shape + (1,) * len(self.trivial_shape))
        elif info.fully_independent:
            # instead of a sequential gather, use a reshape
            assert np.all(idx.ravel() == np.arange(idx.size, int))
            ret = params.reshape(shape + self.trivial_shape)
        else:
            ret = params[idx, ...]
        return ret.transpose(self.transpose_dims)


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
        in_reps: tuple[type[PermRepBase], ...],
        in_features: int,
        out_reps: tuple[type[PermRepBase], ...],
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

        self.bias = (
            SpaceSymmetricTensor(
                out_reps + (out_features,),
                symmetry_group=symmetry_group,
                dtype=param_dtype,
                init=bias_init,
                rngs=rngs,
            )
            if use_bias
            else nnx_compat.data(None)
        )
        self.kernel = SpaceSymmetricTensor(
            in_reps + (in_features,) + out_reps + (out_features,),
            len(in_reps) + 1,
            symmetry_group=symmetry_group,
            dtype=param_dtype,
            init=kernel_init,
            rngs=rngs,
        )

    @property
    def approximate_flops(self):
        bias = self.bias.get_tensor().size if self.bias is not None else 0
        kernel = self.kernel.get_tensor().size
        return bias + kernel

    def __call__(self, inputs):
        ind = inputs.ndim
        incd = len(self.in_reps) + 1
        dtype = nnx.nn.dtypes.canonicalize_dtype(
            inputs, self.kernel.params, dtype=self.dtype
        )
        kernel = self.kernel.get_tensor(dtype=dtype)
        ret = self.dot_general(
            inputs,
            kernel,
            ((tuple(range(ind - incd, ind)), tuple(range(incd))), ((), ())),
            precision=self.precision,
        )
        if self.bias is not None:
            bias = self.bias.get_tensor(dtype=dtype)
            ret = ret + bias
        return ret


class SymDecompLinear(nnx.Module):
    def __init__(
        self,
        in_features: SymDecompDims,
        out_features: SymDecompDims,
        *,
        extra_in_reps: tuple[type[PermRepBase], ...] = (),
        extra_out_reps: tuple[type[PermRepBase], ...] = (),
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

        assert inf.rep.symmetry_group == outf.rep.symmetry_group

        kw = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dot_general=dot_general,
            promote_dtype=promote_dtype,
            rngs=rngs,
        )

        self.in_features = in_features
        self.extra_in_reps = extra_in_reps
        self.out_features = out_features
        self.extra_out_reps = extra_out_reps
        self.use_bias = use_bias
        for k, v in kw.items():
            setattr(self, k, v)

        self.bias = (
            nnx_compat.Dict(
                {
                    k: SpaceSymmetricTensor(
                        extra_out_reps
                        + ((outf.rep.space, v) if k == "space" else (v,)),
                        symmetry_group=outf.rep.symmetry_group,
                        dtype=param_dtype,
                        init=bias_init,
                        rngs=rngs,
                    )
                    for k, v in outf.representations.items()
                }
            )
            if use_bias
            else nnx_compat.data(None)
        )

        self.flavour_invariant = nnx_compat.Dict(
            {
                ki: nnx_compat.Dict(
                    {
                        ko: SpaceSymmetricLinear(
                            extra_in_reps + ((inf.rep.space,) if ki == "space" else ()),
                            vi,
                            extra_out_reps
                            + ((outf.rep.space,) if ko == "space" else ()),
                            vo,
                            use_bias=False,
                            **kw,
                        )
                        for ko, vo in outf.representations.items()
                        if {ki, ko} != {"space", "flavour"}
                    }
                )
                for ki, vi in inf.representations.items()
            }
        )
        if not extra_in_reps and not extra_out_reps:
            # in the usual case, we should not need any gather in these linear ops,
            # let's make sure they're not accidentally in.
            for ki, ilin in self.flavour_invariant.items():
                for ko, lin in ilin.items():
                    if ki == "space" == ko:
                        continue
                    assert (
                        lin.kernel.mapping_info.fully_coupled
                    ), f"{ki} -> {ko}: {lin.kernel.mapping_info}"
        vi, vo = inf.flavour, outf.flavour
        self.flavour_pointwise = (
            SpaceSymmetricLinear(
                extra_in_reps,
                vi,
                extra_out_reps,
                vo,
                use_bias=False,
                **kw,
            )
            if vi and vo
            else nnx_compat.data(None)
        )

    @property
    def approximate_flops(self):
        ret = (
            sum(v.get_tensor().size for v in self.bias.values())
            if self.bias is not None
            else 0
        )
        for lin in self.flavour_invariant.values():
            ret += lin.approximate_flops
        if self.flavour_pointwise is not None:
            ret += self.flavour_pointwise.kernel.size
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
        if self.extra_in_reps:
            batch = inputs.batch_shape
            assert batch[-len(self.extra_in_reps) :] == tuple(
                len(rep) for rep in self.extra_in_reps
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
        if self.extra_out_reps:
            batch = ret.batch_shape
            assert batch[-len(self.extra_out_reps) :] == tuple(
                len(rep) for rep in self.extra_out_reps
            )
        return ret

    def _apply_flat(self, inputs: FlatSymDecomp) -> FlatSymDecomp:
        raise NotImplementedError

    def _apply_split(self, inputs: SplitSymDecomp) -> SplitSymDecomp:
        to_consider = list(inputs.representations.values())
        if self.flavour_pointwise is not None:
            to_consider.append(self.flavour_pointwise.kernel.params)
        for ilin in self.flavour_invariant.values():
            for v in ilin.values():
                p = v.kernel.params
                if p.size:
                    to_consider.append(p)
        if self.bias is not None:
            for v in self.bias.values():
                p = v.params
                if p.size:
                    to_consider.append(v.params)
        dtype = nnx.nn.dtypes.canonicalize_dtype(*to_consider, dtype=self.dtype)

        r = inputs.representations
        inputs = attrs.evolve(
            inputs,
            **dict(zip(r.keys(), self.promote_dtype(r.values(), dtype=dtype))),
        )
        batch = inputs.batch_shape
        extra_in = tuple(len(rep) for rep in self.extra_in_reps)
        if extra_in:
            assert batch[-len(extra_in) :] == extra_in
            batch = batch[: -len(extra_in)]
        extra_out = tuple(len(rep) for rep in self.extra_out_reps)
        xi = inputs.invariant
        xs = inputs.space
        xf = inputs.flavour
        xfa = jnp.mean(xf, axis=-2)

        # start with flavour invariant stuff
        inp = dict(
            invariant=xi,
            space=xs,
            flavour=xfa,
        )
        out = (
            {k: v.get_tensor() for k, v in self.bias.items()}
            if self.bias is not None
            else {}
        )
        for ki, ilin in self.flavour_invariant.items():
            for ko, lin in ilin.items():
                y = lin(inp[ki])
                if (prev := out.get(ko)) is not None:
                    y = prev + y
                out[ko] = y

        # add in flavour dimension
        k = "flavour"
        if (yf := out.get(k)) is not None:
            out[k] = yf[..., None, :]

        if self.flavour_pointwise is not None:
            ko = "flavour"
            x = jnp.moveaxis(xf, -2, -2 - len(self.extra_in_reps))
            y = self.flavour_pointwise(x)
            y = jnp.moveaxis(y, -2 - len(self.extra_out_reps), -2)
            if (prev := out.get(ko)) is not None:
                y = prev + y
            out[ko] = y

        # finally, ensure we have fully broadcasted outputs
        of = self.out_features
        for k, v in dict(
            invariant=extra_out,
            flavour=extra_out + (of.n_flavours,),
            space=extra_out + (of.n_space,),
        ).items():
            y = out.get(k)
            sh = v + (getattr(of, k),)
            if y is None:
                y = jnp.zeros(batch + sh, dtype=dtype)
            else:
                assert y.ndim == len(batch) + len(sh)
                y = jnp.broadcast_to(y, y.shape[: len(batch)] + sh)
            out[k] = y

        return SplitSymDecomp(**out, rep=of.rep)
