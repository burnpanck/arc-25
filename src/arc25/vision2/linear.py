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
from ..lib.misc import first_from
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
    permutation_index: np.ndarray = attrs.field(repr=lambda v: f"{v.shape}", eq=False)
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
        init_scale: float = 1,
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
        # TODO: how do we teach a general init what is inputs and what is outputs?
        if init is default_kernel_init:
            n_in = max(
                1,
                np.prod(
                    [
                        len(r) if not isinstance(r, int) else r
                        for r, c in zip(representations, covariant)
                        if c
                    ]
                ),
            )
            init = jax.nn.initializers.truncated_normal(stddev=np.sqrt(0.1 / n_in))

        self.params = nnx.Param(
            init_scale
            * (init if all(trivial_shape) else jax.nn.initializers.zeros)(
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
        init_scale: float = 1,
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
            init_scale=init_scale,
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
        mode: typing.Literal["flat", "split"] | None = None,
        kernel_init: Initializer = default_kernel_init,
        init_scale: float = 1,
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
            promote_dtype=promote_dtype,
            dot_general=dot_general,
        )
        ikw = dict(
            kernel_init=kernel_init,
            bias_init=bias_init,
            init_scale=init_scale,
            use_bias=False,
            rngs=rngs,
        )

        self.in_features = in_features
        self.extra_in_reps = extra_in_reps
        self.out_features = out_features
        self.extra_out_reps = extra_out_reps
        self.use_bias = use_bias
        self.mode = mode
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
                            **kw,
                            **ikw,
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
                **kw,
                **ikw,
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
        for ilin in self.flavour_invariant.values():
            for lin in ilin.values():
                ret += lin.approximate_flops
        if self.flavour_pointwise is not None:
            ret += (
                self.flavour_pointwise.approximate_flops
                * self.in_features.rep.n_flavours
            )
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

        mode = first_from(mode, self.mode)
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
                dtype = nnx.nn.dtypes.canonicalize_dtype(
                    *inputs.elements.values(), dtype=self.dtype
                )
                ret = self._apply_flat(inputs.as_flat(dtype=dtype))
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

    def _build_flat_kernel(self, dtype: Dtype) -> jt.Array:
        """Build the flat kernel matrix as a block-structured matrix.

        Returns:
            Kernel with shape (*extra_in, total_in, *extra_out, total_out)
            where blocks correspond to (rep_in → rep_out) transformations.
        """
        inf = self.in_features
        outf = self.out_features

        # Feature dimensions
        ni, ns, nf = inf.invariant, inf.space, inf.flavour
        no, nos, nof = outf.invariant, outf.space, outf.flavour
        n_space_in, n_flavours = inf.rep.n_space, inf.rep.n_flavours
        n_space_out, n_flavours_out = outf.rep.n_space, outf.rep.n_flavours

        extra_in_shape = tuple(len(rep) for rep in self.extra_in_reps)
        extra_out_shape = tuple(len(rep) for rep in self.extra_out_reps)
        n_extra_in = len(extra_in_shape)
        n_extra_out = len(extra_out_shape)

        # Axis positions for concatenation
        in_feat_axis = n_extra_in  # Input features are after extra_in dims
        out_feat_axis = (
            n_extra_in + 1 + n_extra_out
        )  # Output features are after extra_in, in_feat, extra_out

        # Build blocks: blocks[ki][ko] = kernel for (ki → ko) transformation
        # Each block shape: (*extra_in, in_feat, *extra_out, out_feat)
        blocks = {ki: {} for ki in FlatSymDecomp.subrep_seq}

        # Fill blocks from flavour_invariant pathways
        for ki in FlatSymDecomp.subrep_seq:
            if ki not in self.flavour_invariant:
                continue
            ilin = self.flavour_invariant[ki]

            for ko in FlatSymDecomp.subrep_seq:
                if (lin := getattr(ilin, ko, None)) is None:
                    # No connection between ki and ko, create zero block
                    in_feat = {
                        "invariant": ni,
                        "space": ns * n_space_in,
                        "flavour": nf * n_flavours,
                    }[ki]
                    out_feat = {
                        "invariant": no,
                        "space": nos * n_space_out,
                        "flavour": nof * n_flavours_out,
                    }[ko]

                    block_shape = (
                        extra_in_shape + (in_feat,) + extra_out_shape + (out_feat,)
                    )
                    blocks[ki][ko] = jnp.zeros(block_shape, dtype=dtype)
                    continue

                k_block = lin.kernel.get_tensor(dtype=dtype)

                # Flatten space dimensions if present
                # Original shape has space dims separate: (*extra_in, [n_space_in], feat, *extra_out, [n_space_out], feat)

                if ki == "space":
                    # Shape: (*extra_in, n_space_in, ns, *extra_out, ...)
                    # Flatten to: (*extra_in, n_space_in*ns, *extra_out, ...)
                    k_block = k_block.reshape(
                        *k_block.shape[:n_extra_in],
                        n_space_in * ns,
                        *k_block.shape[n_extra_in + 2 :],
                    )

                if ko == "space":
                    # Shape: (*extra_in, in_feat, *extra_out, n_space_out, nos)
                    # Flatten to: (*extra_in, in_feat, *extra_out, n_space_out*nos)
                    k_block = k_block.reshape(
                        *k_block.shape[:out_feat_axis], n_space_out * nos
                    )

                # Handle flavour averaging on input side
                if ki == "flavour":
                    # Tile across n_flavours input flavours with averaging
                    # Current shape: (*extra_in, nf, *extra_out, out_feat)
                    # Target shape: (*extra_in, nf*n_flavours, *extra_out, out_feat)
                    tile_shape = [1] * k_block.ndim
                    tile_shape[in_feat_axis] = n_flavours
                    k_block = jnp.tile(k_block, tile_shape) / n_flavours

                # Handle flavour tiling on output side
                if ko == "flavour":
                    # Tile across n_flavours_out output flavours
                    # Current shape: (*extra_in, in_feat, *extra_out, nof)
                    # Target shape: (*extra_in, in_feat, *extra_out, nof*n_flavours_out)
                    tile_shape = [1] * k_block.ndim
                    tile_shape[out_feat_axis] = n_flavours_out
                    k_block = jnp.tile(k_block, tile_shape)

                blocks[ki][ko] = k_block

        # Add flavour_pointwise to the flavour→flavour block (block-diagonal structure)
        if self.flavour_pointwise is not None:
            k_pw = self.flavour_pointwise.kernel.get_tensor(dtype=dtype)
            # Shape: (*extra_in, nf, *extra_out, nof)

            # Expand to block-diagonal across flavours
            # Target shape: (*extra_in, nf*n_flavours, *extra_out, nof*n_flavours_out)
            n_blocks = min(n_flavours, n_flavours_out)

            # Build block-diagonal matrix
            block_diag_shape = (
                extra_in_shape
                + (nf * n_flavours,)
                + extra_out_shape
                + (nof * n_flavours_out,)
            )
            block_diag = jnp.zeros(block_diag_shape, dtype=dtype)

            for f_idx in range(n_blocks):
                # Build slice for this diagonal block
                in_slice = slice(f_idx * nf, (f_idx + 1) * nf)
                out_slice = slice(f_idx * nof, (f_idx + 1) * nof)

                # Create index tuple accounting for extra dims
                idx = (
                    *(slice(None),) * n_extra_in,
                    in_slice,
                    *(slice(None),) * n_extra_out,
                    out_slice,
                )

                block_diag = block_diag.at[idx].set(k_pw)

            # Add to the flavour→flavour block
            blocks["flavour"]["flavour"] = blocks["flavour"]["flavour"] + block_diag

        # Concatenate blocks to form the full kernel
        # First, for each output representation, concatenate all input representations
        row_blocks = []
        for ko in FlatSymDecomp.subrep_seq:
            # Concatenate blocks along the input feature axis
            row = jnp.concatenate(
                [blocks[ki][ko] for ki in FlatSymDecomp.subrep_seq], axis=in_feat_axis
            )
            row_blocks.append(row)

        # Then concatenate all rows along the output feature axis
        kernel = jnp.concatenate(row_blocks, axis=out_feat_axis)

        return kernel

    def _apply_flat(self, inputs: FlatSymDecomp) -> FlatSymDecomp:
        """Apply linear transformation using a single large matmul.

        Builds a block-structured kernel and applies it via one matmul,
        trading slightly more FLOPs for better hardware utilization.
        """
        # Canonicalize dtype
        to_consider = [inputs.data]
        if self.flavour_pointwise is not None:
            to_consider.append(self.flavour_pointwise.kernel.params)
        for ilin in self.flavour_invariant.values():
            for lin in ilin.values():
                p = lin.kernel.params
                if p.size:
                    to_consider.append(p)
        if self.bias is not None:
            for v in self.bias.values():
                p = v.params
                if p.size:
                    to_consider.append(p)

        dtype = nnx.nn.dtypes.canonicalize_dtype(*to_consider, dtype=self.dtype)
        x_data = self.promote_dtype([inputs.data], dtype=dtype)[0]

        # Build the kernel matrix
        kernel = self._build_flat_kernel(dtype)

        # Build the bias vector by concatenating blocks
        if self.bias is not None:
            outf = self.out_features
            no, nos, nof = outf.invariant, outf.space, outf.flavour  # noqa: F841
            n_space_out, n_flavours_out = outf.rep.n_space, outf.rep.n_flavours
            extra_out_shape = tuple(len(rep) for rep in self.extra_out_reps)
            n_extra_out = len(extra_out_shape)

            bias_parts = []
            for rep_name in FlatSymDecomp.subrep_seq:
                b_block = self.bias[rep_name].get_tensor(dtype=dtype)
                # Shape: (*extra_out, feat)

                if rep_name == "space":
                    # Flatten: (*extra_out, n_space_out, nos) -> (*extra_out, n_space_out*nos)
                    b_block = b_block.reshape(
                        *b_block.shape[:n_extra_out], n_space_out * nos
                    )
                elif rep_name == "flavour":
                    # Tile: (*extra_out, nof) -> (*extra_out, nof*n_flavours_out)
                    tile_shape = [1] * b_block.ndim
                    tile_shape[-1] = n_flavours_out
                    b_block = jnp.tile(b_block, tile_shape)

                bias_parts.append(b_block)

            # Concatenate along the feature axis (last axis)
            bias = jnp.concatenate(bias_parts, axis=-1)
        else:
            bias = None

        # Apply the transformation
        # x_data shape: (*batch, *extra_in, total_in)
        # kernel shape: (*extra_in, total_in, *extra_out, total_out)
        # output shape: (*batch, *extra_out, total_out)

        # Contract over (*extra_in, total_in) dimensions
        extra_in_shape = tuple(len(rep) for rep in self.extra_in_reps)
        n_extra_in = len(extra_in_shape)

        # Calculate contraction axes (must be nonnegative)
        n_contract = n_extra_in + 1  # extra_in dims + total_in dim
        n_batch = x_data.ndim - n_contract
        x_contract_axes = tuple(range(n_batch, x_data.ndim))  # Last n_contract axes
        k_contract_axes = tuple(range(n_contract))  # First n_extra_in+1 axes

        y_data = self.dot_general(
            x_data,
            kernel,
            ((x_contract_axes, k_contract_axes), ((), ())),
            precision=self.precision,
        )

        if bias is not None:
            y_data = y_data + bias

        return FlatSymDecomp(data=y_data, dim=self.out_features)

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
            {k: v.get_tensor(dtype=dtype) for k, v in self.bias.items()}
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
