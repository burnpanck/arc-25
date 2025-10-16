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
from ..lib.compat import Self
from ..lib.misc import first_from, show_dims
from ..symmetry import PermRepBase, transform_vector
from .fields import CoordinateGrid
from .linear import SpaceSymmetricTensor, SymDecompLinear
from .symrep import FlatSymDecomp, SplitSymDecomp, SymDecompBase, SymDecompDims


class AxialAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        in_features: SymDecompDims,
        out_features: SymDecompDims | None = None,
        *,
        num_groups: int | None = None,
        qk_head_width: SymDecompDims | None = None,
        v_head_width: SymDecompDims | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        # broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        activation=jax.nn.sigmoid,
        # out_kernel_init: Initializer | None = None,
        bias_init: Initializer = default_bias_init,
        # out_bias_init: Initializer | None = None,
        use_bias: bool | None = None,
        use_qk_bias: bool | None = None,
        use_v_bias: bool | None = None,
        use_out_bias: bool | None = None,
        use_chirality_rep: bool = True,
        rope_freq_scaling: typing.Literal["linear-freqs", "linear-k"] = "linear-k",
        learnable_rope_freqs: typing.Literal["per-head", "tied", "none"] | None = None,
        per_head_rope_freq: bool | None = None,
        # attention_fn: Callable[..., Array] = dot_product_attention,
        normalise_qk: bool = False,
        normalise_pre_out: bool = False,
        keep_rngs: bool = True,
        rngs: rnglib.Rngs,
    ):
        if out_features is None:
            out_features = in_features
        if qk_head_width is None:
            qk_head_width = in_features.map_representations(lambda k, v: v // num_heads)
        assert all(not v % 2 for v in qk_head_width.representations.values())
        if v_head_width is None:
            v_head_width = out_features.map_representations(lambda k, v: v // num_heads)
        assert not num_heads % num_groups
        use_qk_bias = first_from(use_qk_bias, use_bias, False)
        use_v_bias = first_from(use_v_bias, use_bias, True)
        use_out_bias = first_from(use_out_bias, use_bias, False)

        assert learnable_rope_freqs is None or per_head_rope_freq is None
        if per_head_rope_freq is not None:
            learnable_rope_freqs = "per-head" if per_head_rope_freq else "tied"
        if learnable_rope_freqs is None:
            learnable_rope_freqs = "none"

        if normalise_qk or normalise_pre_out:
            raise NotImplementedError()

        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.qk_head_width = qk_head_width
        self.v_head_width = v_head_width
        self.num_groups = num_groups
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.activation = activation
        self.normalise_qk = normalise_qk
        self.normalise_pre_out = normalise_pre_out
        if False:
            self.param_dtype = param_dtype
            self.kernel_init = kernel_init
            self.bias_init = bias_init
            self.use_qk_bias = use_qk_bias
            self.use_v_bias = use_v_bias
            self.use_out_bias = use_out_bias
            self.learnable_rope_freqs = learnable_rope_freqs
        self.use_chirality_rep = use_chirality_rep
        self.rngs = rngs if keep_rngs else None

        head_reps = (
            symmetry.AxialDirRep,
            symmetry.ChiralityRep if use_chirality_rep else symmetry.TrivialRep,
        )
        kw = dict(
            kernel_init=kernel_init,
            bias_init=bias_init,
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        # Q has extra dimensions: head_reps + (M, K) where M = num_heads // num_groups
        M = num_heads // num_groups
        K = num_groups
        self.q = SymDecompLinear(
            in_features,
            qk_head_width,
            extra_out_reps=head_reps + (M, K),
            use_bias=use_qk_bias,
            **kw,
        )
        # K and V have extra dimensions: head_reps + (1, K) for broadcasting against M
        self.k = SymDecompLinear(
            in_features,
            qk_head_width,
            extra_out_reps=head_reps + (1, K),
            use_bias=use_qk_bias,
            **kw,
        )
        self.v = SymDecompLinear(
            in_features,
            v_head_width,
            extra_out_reps=head_reps + (1, K),
            use_bias=use_v_bias,
            **kw,
        )
        # Out has extra input dimensions: head_reps + (M, K)
        self.out = SymDecompLinear(
            v_head_width,
            out_features,
            extra_in_reps=head_reps + (M, K),
            use_bias=use_out_bias,
            **kw,
        )
        self.rope_freq_params = (
            qk_head_width.map_representations(
                lambda k, v: nnx.Param(
                    initializers.truncated_normal(stddev=1, dtype=param_dtype)(
                        rngs.param(),
                        {
                            "per-head": (M, K, v // 2, 2),
                            "tied": (v // 2, 2),
                        }[learnable_rope_freqs],
                        dtype=param_dtype,
                    )
                ),
                cls=nnx_compat.Dict,
            )
            if learnable_rope_freqs != "none"
            else nnx_compat.data(None)
        )
        self.rope_freq_scaling = qk_head_width.map_representations(
            lambda k, v: np.pi
            / np.array(
                [np.linspace(1, 30, v // 2), np.linspace(0.1, 1, v // 2)], dtype=dtype
            ).T,
            cls=nnx_compat.Dict,
        )

    def __call__(
        self,
        inputs: SymDecompBase,
        grid: CoordinateGrid,
        *,
        return_attention_weights: bool = False,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        debug: str | None = None,
    ) -> SymDecompBase:

        assert self.in_features.validate(inputs), self.in_features.validation_problems(
            inputs
        )

        batch = inputs.batch_shape
        Bs = batch[:-2]
        Y, X = batch[-2:]
        nB = len(Bs)
        N = self.num_heads
        K = self.num_groups
        M = N // K  # noqa: F841

        assert (
            "".join(basis.symbol for basis in symmetry.AxialDirRep) == "↓↑→←"
        ), "The following code depends on this order"

        qkv = {}
        for m in "qkv":
            lin = getattr(self, m)
            y = lin(inputs, mode=mode)
            # After lin: Q has shape (*batch, Y, X, A*D, V, M, K, [F,] H)
            #            K, V have shape (*batch, Y, X, A*D, V, 1, K, [F,] H)
            # where A=2 (axial), D=2 (direction), V=1or2 (chirality), M, K (heads)
            # [F,] is present for space/flavour in split mode, absent for invariant/flat
            # Reshape to: (*batch, Y, X, A, D, V, M, K, F, H_pairs, 2) where F is dummy if absent
            y = dict(
                **{
                    k: v.reshape(
                        *v.shape[: nB + 2],  # (*batch, Y, X)
                        2,
                        2,  # (A*D,) -> (A, D)
                        *v.shape[nB + 3 : nB + 6],  # (V or 1, M or 1, K)
                        *(v.shape[nB + 6 : -1] or (1,)),  # [F,] or F=1
                        *((-1, 2) if m in "qk" else (-1,)),  # (H//2, 2) or (H,)
                    )
                    for k, v in y.elements.items()
                },
            )
            if debug:
                for k, v in y.items():
                    print(
                        f"[{debug}] After reshape {m}.{k}:"
                        f" {show_dims('yxadvmkfhp' if m!='v' else 'yxadvmkfh',v)},"
                        f" max={np.abs(v).max():.3e}"
                    )
            qkv[m] = y

        all_v = []
        all_aw = []
        for axis, pos in enumerate(
            [grid.ypos[..., :, None, :], grid.xpos[..., None, :, :]]
        ):
            rot = {}
            for k, v in self.rope_freq_params.items():
                # M K H 2
                freq = v * self.rope_freq_scaling[k]
                # ... Y X M K H 2
                phi = (freq * pos[..., :, :, None, None, None, :]).sum(-1)
                cs, sn = jnp.cos(phi), jnp.sin(phi)
                nsn = -sn
                # rd will have shape ... Y X M K H 3
                rd = jnp.moveaxis(jnp.array([cs, sn, nsn]), 0, -1)
                # idx will have shape D=2 P=2 Q=2
                idx = np.r_[0, 2, 1, 0, 0, 1, 2, 0].reshape(2, 2, 2)
                # r will have shape ... Y X D=2 M K H P=2 Q=2
                r = jnp.moveaxis(rd[..., idx], -3, -6)
                rot[k] = r
            if "data" in qkv["q"]:
                # This is a FlatSymDecomp; tile and concatenate rotations along H dimension
                # Flat data shape: (..., F=1, Dd) where Dd = Di + Fs*Ds + Ff*Df
                # Each rotation has shape: ...Y X D M K H P Q where H is per-representation
                # Need to tile each representation's rotation and concatenate
                rot_parts = []
                for rep_name in FlatSymDecomp.subrep_seq:
                    r = rot[rep_name]  # Shape: ...Y X D M K H P Q

                    # Determine tiling factor for this representation
                    tile_factor = dict(
                        invariant=1,
                        space=self.qk_head_width.rep.n_space,
                        flavour=self.qk_head_width.rep.n_flavours,
                    )[rep_name]

                    # Tile along the H dimension (axis -3)
                    if tile_factor > 1:
                        r = jnp.tile(r, [tile_factor] + [1] * 2)

                    rot_parts.append(r)

                # Concatenate along the H dimension (axis -3)
                rot = dict(data=jnp.concatenate(rot_parts, axis=-3))
            qk = []
            for kk in ["q", "k"]:
                qk.append(
                    {
                        k: jnp.einsum(
                            "...yxdvmkfhp,...yxdmkhpq->...yxdvmkfhq",
                            v[..., :, :, axis, :, :, :, :, :, :, :],
                            rot[k],
                        )
                        for k, v in qkv[kk].items()
                    }
                )
            Q, K = qk

            logits = sum(
                jnp.einsum(
                    [
                        "...txdvmkfhp,...sxdvmkfhp->...tsxdvmk",
                        "...ytdvmkfhp,...ysdvmkfhp->...ytsdvmk",
                    ][axis],
                    v,
                    K[k],
                )
                for k, v in Q.items()
            )
            # each element brings F*H*P (the last three dims) of coefficients
            d = sum(int(np.prod(v.shape[-3:])) for v in Q.values())
            if debug:
                print(
                    f"[{debug}] axis={axis}: {d=} ({ {k:v.shape[-3:] for k,v in Q.items()} })"
                )
            scale = 1 / np.sqrt(d)
            mask = (
                grid.mask[..., None, :, :, None, None, None, None]
                if not axis
                else grid.mask[..., :, None, :, None, None, None, None]
            )
            weight = jnp.where(mask, self.activation(scale * logits), 0)
            weight = weight / jnp.sqrt(1 + mask.sum(axis=[-6, -5][axis], keepdims=True))
            V = {
                k: jnp.einsum(
                    [
                        "...tsxdvmk,...sxdvmkfh->...txdvmkfh",
                        "...ytsdvmk,...ysdvmkfh->...ytdvmkfh",
                    ][axis],
                    weight,
                    v[..., :, :, axis, :, :, :, :, :, :],
                )[..., :, :, None, :, :, :, :, :, :]
                for k, v in qkv["v"].items()
            }
            all_v.append(V)
            all_aw.append(weight)

        rep = self.v.out_features.rep
        match set(qkv["v"].keys()):
            case SplitSymDecomp.element_names:
                sym_decomp_cls = SplitSymDecomp
                extra_v_kw = dict(rep=rep)
            case FlatSymDecomp.element_names:
                sym_decomp_cls = FlatSymDecomp
                extra_v_kw = dict(dim=self.v.out_features)
            case _:
                raise KeyError(
                    f"Cannot determine SymDecomp implementation for elements {set(qkv['v'].keys())}"
                )
        # Concatenate the two axes (horizontal and vertical attention)
        V_concat = {
            k: jnp.concatenate([V[k] for V in all_v], axis=-7) for k in qkv["v"]
        }
        if debug:
            for k, v in V_concat.items():
                print(f"[{debug}] V.{k}: {show_dims('yxadvmkfh',v)}")

        # Remove dummy F dimension if present (F=1), otherwise keep as is
        V = sym_decomp_cls(
            **{
                k: v.reshape(
                    *v.shape[: nB + 2],  # (*batch, Y, X)
                    4,  # (A,D) -> (A*D,)
                    *v.shape[-5:-2],  # (V or 1, M or 1, K)
                    *(
                        v.shape[-2:-1] if k not in {"invariant", "data"} else ()
                    ),  # F only for those which have it
                    v.shape[-1],  # H
                )
                for k, v in V_concat.items()
            },
            **extra_v_kw,
        )
        out = self.out(V, mode=mode)
        if return_attention_weights:
            return out, all_aw
        return out


class GlobalAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        *,
        in_features: SymDecompDims | None = None,
        target_features: SymDecompDims | None = None,
        source_features: SymDecompDims | None = None,
        out_features: SymDecompDims | None = None,
        qk_head_width: SymDecompDims | None = None,
        v_head_width: SymDecompDims | None = None,
        num_groups: int | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        # broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        use_softmax: bool = True,
        activation=None,
        # out_kernel_init: Initializer | None = None,
        bias_init: Initializer = default_bias_init,
        # out_bias_init: Initializer | None = None,
        use_bias: bool | None = None,
        use_qk_bias: bool | None = None,
        use_v_bias: bool | None = None,
        use_out_bias: bool | None = None,
        head_rep: type[PermRepBase] = symmetry.TrivialRep,
        # attention_fn: Callable[..., Array] = dot_product_attention,
        normalise_qk: bool = False,
        keep_rngs: bool = True,
        rngs: rnglib.Rngs,
    ):
        if num_groups is None:
            num_groups = num_heads
        if target_features is None:
            target_features = in_features
        if source_features is None:
            source_features = in_features
        if out_features is None:
            out_features = target_features
        if qk_head_width is None:
            qk_head_width = target_features.map_representations(
                lambda k, vq, vkv: min(vq, vkv) // num_heads, source_features
            )
        if v_head_width is None:
            v_head_width = out_features.map_representations(lambda k, v: v // num_heads)
        assert not num_heads % num_groups
        use_qk_bias = first_from(use_qk_bias, use_bias, False)
        use_v_bias = first_from(use_v_bias, use_bias, False)
        use_out_bias = first_from(use_out_bias, use_bias, False)

        if normalise_qk:
            raise NotImplementedError()

        self.num_heads = num_heads
        self.target_features = target_features
        self.source_features = source_features
        self.out_features = out_features
        self.qk_head_width = qk_head_width
        self.v_head_width = v_head_width
        self.num_groups = num_groups
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.use_softmax = use_softmax
        self.activation = activation
        self.normalise_qk = normalise_qk
        if False:
            self.param_dtype = param_dtype
            self.kernel_init = kernel_init
            self.bias_init = bias_init
            self.use_qk_bias = use_qk_bias
            self.use_v_bias = use_v_bias
            self.use_out_bias = use_out_bias
        self.head_rep = head_rep
        self.rngs = rngs if keep_rngs else None

        head_reps = (head_rep,)
        kw = dict(
            kernel_init=kernel_init,
            bias_init=bias_init,
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        M = num_heads // num_groups
        K = num_groups

        for mode, (inf, nh) in dict(
            q=(target_features, M),
            k=(source_features, 1),
        ).items():
            setattr(
                self,
                mode,
                SymDecompLinear(
                    inf,
                    qk_head_width,
                    extra_out_reps=head_reps + (nh, K),
                    use_bias=use_qk_bias,
                    **kw,
                ),
            )
        self.v = SymDecompLinear(
            source_features,
            v_head_width,
            extra_out_reps=head_reps + (1, K),
            use_bias=use_v_bias,
            **kw,
        )
        self.out = SymDecompLinear(
            v_head_width,
            out_features,
            extra_in_reps=head_reps + (M, K),
            use_bias=use_out_bias,
            **kw,
        )

    def __call__(
        self,
        target: SymDecompBase,
        source: SymDecompBase | None = None,
        mask: jt.Bool[jt.Array, "... T S"] | None = None,
        *,
        return_attention_weights: bool = False,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
    ) -> SymDecompBase:
        if source is None:
            source = target

        assert self.target_features.validate(
            target
        ), self.target_features.validation_problems(target)
        assert self.source_features.validate(
            source
        ), self.source_features.validation_problems(source)
        batch = np.broadcast_shapes(target.batch_shape[:-1], source.batch_shape[:-1])
        if mask is not None:
            # done early, so we get an exeption if shapes mismatch
            mask = jnp.broadcast_to(
                mask,
                batch
                + (
                    target.batch_shape[-1],
                    source.batch_shape[-1],
                ),
            )
        nB = len(batch)
        T = target.batch_shape[-1]  # noqa: F841
        S = source.batch_shape[-1]  # noqa: F841
        N = self.num_heads
        K = self.num_groups
        M = N // K  # noqa: F841

        # insert broadcasting batch dims where needed
        if n_missing := len(target.batch_shape) - 1 - len(batch):
            target = target.batch_reshape(*(1,) * n_missing, *target.batch_shape)
        if n_missing := len(source.batch_shape) - 1 - len(batch):
            source = source.batch_reshape(*(1,) * n_missing, *source.batch_shape)

        qkv = {}
        for m in "qkv":
            lin = getattr(self, m)
            y = lin(dict(q=target).get(m, source), mode=mode)
            # After lin: Q has shape (*batch, S/T, V, M, K, [F,] H)
            #            K, V have shape (*batch, S/T, V, 1, K, [F,] H)
            # where S/T are source/target tokens, V=head_rep, M, K (heads)
            # [F,] is present for space/flavour in split mode, absent for invariant/flat
            # Insert dummy F where needed,
            y = dict(
                **{
                    k: v[..., None, :] if k in {"invariant", "data"} else v
                    for k, v in y.elements.items()
                },
            )
            qkv[m] = y

        Q, K = [qkv[k] for k in ["q", "k"]]
        logits = sum(
            jnp.einsum(
                "...tvmkfh,...svmkfh->...tsvmk",
                Q[k],
                K[k],
            )
            for k in Q
        )
        d = sum(np.prod(v.shape[-2:]) for v in Q.values())
        scale = 1 / np.sqrt(d)
        logits = scale * logits
        if self.activation is not None:
            logits = self.activation(logits)
        lgmsk = mask[..., :, :, None, None, None] if mask is not None else True
        if self.use_softmax:
            weight = jax.nn.softmax(logits, axis=-4, where=lgmsk)
        else:
            weight = jnp.where(lgmsk, logits, 0)
        V = {
            k: jnp.einsum(
                "...tsvmk,...svmkfh->...tvmkfh",
                weight,
                v,
            )
            for k, v in qkv["v"].items()
        }
        match set(qkv["v"].keys()):
            case SplitSymDecomp.element_names:
                sym_decomp_cls = SplitSymDecomp
                extra_v_kw = dict(rep=self.v.out_features.rep)
            case FlatSymDecomp.element_names:
                sym_decomp_cls = FlatSymDecomp
                extra_v_kw = dict(dim=self.v.out_features)
            case _:
                raise KeyError(
                    f"Cannot determine SymDecomp implementation for elements {set(qkv['v'].keys())}"
                )
        V = sym_decomp_cls(
            **{
                # get rid of that dummy "f" dimension where it doesn't belong
                k: (
                    v.reshape(
                        *v.shape[: nB + 4],
                        v.shape[-1],
                    )
                    if k in {"invariant", "data"}
                    else v
                )
                for k, v in V.items()
            },
            **extra_v_kw,
        )
        out = self.out(V, mode=mode)
        if return_attention_weights:
            return out, weight
        return out
