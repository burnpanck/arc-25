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
        per_head_rope_freq: bool = True,
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
            self.per_head_rope_freq = per_head_rope_freq
        self.use_chirality_rep = use_chirality_rep
        self.rngs = rngs if keep_rngs else None

        head_reps = (symmetry.AxialDirRep,) + (
            (symmetry.ChiralityRep,) if use_chirality_rep else ()
        )
        kw = dict(
            kernel_init=kernel_init,
            bias_init=bias_init,
            precision=precision,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        for mode, nh in dict(
            q=num_heads,
            k=num_groups,
        ).items():
            setattr(
                self,
                mode,
                SymDecompLinear(
                    in_features,
                    qk_head_width.map_representations(
                        lambda k, v: v * nh  # noqa: B023
                    ),
                    extra_out_reps=head_reps,
                    use_bias=use_qk_bias,
                    **kw,
                ),
            )
        self.v = SymDecompLinear(
            in_features,
            v_head_width.map_representations(lambda k, v: v * num_groups),
            extra_out_reps=head_reps,
            use_bias=use_v_bias,
            **kw,
        )
        self.out = SymDecompLinear(
            v_head_width.map_representations(lambda k, v: v * num_heads),
            out_features,
            extra_in_reps=head_reps,
            use_bias=use_out_bias,
            **kw,
        )
        self.rope_freq_params = qk_head_width.map_representations(
            lambda k, v: nnx.Param(
                initializers.truncated_normal(stddev=1, dtype=param_dtype)(
                    rngs.param(),
                    (
                        (num_heads // num_groups, num_groups, v // 2, 2)
                        if per_head_rope_freq
                        else (v // 2, 2)
                    ),
                    dtype=param_dtype,
                )
            ),
            cls=nnx_compat.Dict,
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
    ) -> SymDecompBase:

        assert self.in_features.validate(inputs), self.in_features.validation_problems(
            inputs
        )

        batch = inputs.batch_shape
        # Bs = batch[:-2]
        Y, X = batch[-2:]
        N = self.num_heads
        K = self.num_groups
        M = N // K

        S = (2, 2, 2) if self.use_chirality_rep else (2, 2, 1)
        assert (
            "".join(basis.symbol for basis in symmetry.AxialDirRep) == "↓↑→←"
        ), "The following code depends on this order"

        qkv = {}
        for m in "qkv":
            lin = getattr(self, m)
            y = lin(inputs)
            y = dict(
                rep=y.rep,
                **{
                    k: v.reshape(
                        *v.shape[: len(batch)],  # batch, including X/Y
                        *S,  # (3) new extra_out_rep
                        *v.shape[len(y.batch_shape) : -1]
                        or (
                            1,
                        ),  # (1) main representation (flavour/space) if any, adding broadcasting dim otherwise
                        (M if m == "q" else 1),
                        K,  # (2) heads in groups
                        *(
                            (-1,) if m == "v" else (-1, 2)
                        ),  # (1/2) head dim (with rope pairs)
                    )
                    for k, v in y.representations.items()
                },
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
                # idx will have shape 2 2 2
                idx = np.r_[0, 2, 1, 0, 0, 1, 2, 0].reshape(2, 2, 2)
                # r will have shape ... Y X 2 M K H 2 2
                r = jnp.moveaxis(rd[..., idx], -3, -6)
                rot[k] = r
            qk = []
            for kk in ["q", "k"]:
                qk.append(
                    {
                        k: jnp.einsum(
                            "...yxdvfmkhp,...yxdmkhpq->...yxdvfmkhq",
                            qkv[kk][k][..., :, :, axis, :, :, :, :, :, :, :],
                            r,
                        )
                        for k, r in rot.items()
                    }
                )
            Q, K = qk
            #            for k in rot:
            #                print(f"Q[{k}]: {show_dims(["txdvfmkhp","ytdvfmkhp"][axis], Q[k])} (size {Q[k].size})")
            #                print(f"K[{k}]: {show_dims(["sxdvfmkhp","ysdvfmkhp"][axis], K[k])} (size {K[k].size})")

            logits = sum(
                jnp.einsum(
                    [
                        "...txdvfmkhp,...sxdvfmkhp->...tsxdvmk",
                        "...ytdvfmkhp,...ysdvfmkhp->...ytsdvmk",
                    ][axis],
                    Q[k],
                    K[k],
                )
                for k in rot
            )
            d = sum(Q[k].shape[-5] * Q[k].shape[-2] * Q[k].shape[-1] for k in rot)
            scale = 1 / np.sqrt(d)
            # print(f"weight: {show_dims(["tsxdvmk","ytsdvmk"][axis], scale)}")
            mask = (
                grid.mask[..., None, :, :, None, None, None, None]
                if not axis
                else grid.mask[..., :, None, :, None, None, None, None]
            )
            # print(f"gridmask: {show_dims(["tsxdvmk","ytsdvmk"][axis], mask)}")
            weight = jnp.where(mask, self.activation(scale * logits), 0)
            # print(f"weight: {show_dims(["tsxdvmk","ytsdvmk"][axis], weight)}")
            Vrep = qkv["v"]
            # for k in rot:
            #    vv = Vrep[k][..., :, :, axis, :, :, :, :, :, :]
            #    print(f"{k}: {show_dims(["sxdvfmkh","ysdvfmkh"][axis], vv)}")
            V = {
                k: jnp.einsum(
                    [
                        "...tsxdvmk,...sxdvfmkh->...txdvfmkh",
                        "...ytsdvmk,...ysdvfmkh->...ytdvfmkh",
                    ][axis],
                    weight,
                    Vrep[k][..., :, :, axis, :, :, :, :, :, :],
                )[..., :, :, None, :, :, :, :, :, :]
                for k in rot
            }
            all_v.append(V)
            all_aw.append(weight)

        rep = qkv["v"]["rep"]
        V = SplitSymDecomp(
            **{
                k: jnp.concatenate(
                    [V[k] for V in all_v],
                    axis=-7,
                ).reshape(
                    *batch,
                    *(4, 2) if self.use_chirality_rep else (4,),
                    *dict(
                        invariant=(),
                        flavour=(rep.n_flavours,),
                        space=(rep.n_space,),
                    )[k],
                    -1,
                )
                for k in rot
            },
            rep=rep,
        )
        out = self.out(V)
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
        for mode, (inf, nh) in dict(
            q=(target_features, num_heads),
            k=(source_features, num_groups),
        ).items():
            setattr(
                self,
                mode,
                SymDecompLinear(
                    inf,
                    qk_head_width.map_representations(
                        lambda k, v: v * nh  # noqa: B023
                    ),
                    extra_out_reps=head_reps,
                    use_bias=use_qk_bias,
                    **kw,
                ),
            )
        self.v = SymDecompLinear(
            source_features,
            v_head_width.map_representations(lambda k, v: v * num_groups),
            extra_out_reps=head_reps,
            use_bias=use_v_bias,
            **kw,
        )
        self.out = SymDecompLinear(
            v_head_width.map_representations(lambda k, v: v * num_heads),
            out_features,
            extra_in_reps=head_reps,
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
        assert target.batch_shape[:-1] == source.batch_shape[:-1]
        if mask is not None:
            # done early, so we get an exeption if shapes mismatch
            mask = jnp.broadcast_to(
                mask,
                target.batch_shape[:-1]
                + (
                    target.batch_shape[-1],
                    source.batch_shape[-1],
                ),
            )

        batch = np.broadcast_shapes(target.batch_shape[:-1], source.batch_shape[:-1])
        T = target.batch_shape[-1]  # noqa: F841
        S = source.batch_shape[-1]  # noqa: F841
        N = self.num_heads
        K = self.num_groups
        M = N // K

        qkv = {}
        qkvrep = {}
        for m in "qkv":
            lin = getattr(self, m)
            y = lin(dict(q=target).get(m, source), mode=mode)
            qkvrep[m] = y.rep
            y = dict(
                **{
                    k: v.reshape(
                        *v.shape[
                            : len(batch) + 2
                        ],  # batch, including T/S and the newly created head_rep
                        *v.shape[len(batch) + 2 : -1]
                        or (
                            1,
                        ),  # (1) main representation (flavour/space) if any, adding broadcasting dim otherwise
                        (M if m == "q" else 1),
                        K,  # (2) heads in groups
                        v.shape[-1] // (N if m == "q" else K),  # (1) head dim
                    )
                    for k, v in y.representations.items()
                },
            )
            qkv[m] = y

        Q, K = [qkv[k] for k in ["q", "k"]]
        logits = sum(
            jnp.einsum(
                "...tdfmkh,...sdfmkh->...tsdmk",
                Q[k],
                K[k],
            )
            for k in Q
        )
        d = sum(v.shape[-4] * v.shape[-1] for v in Q.values())
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
                "...tsdmk,...sdfmkh->...tdfmkh",
                weight,
                v,
            )
            for k, v in qkv["v"].items()
        }

        rep = qkvrep["v"]
        V = SplitSymDecomp(
            **{
                # get rid of that "f" dimension where it doesn't belong
                k: v.reshape(
                    *v.shape[: len(batch) + 2],  # retain the head_rep dim
                    *v.shape[-dict(invariant=3).get(k, 4) : -3],
                    np.prod(v.shape[-3:]),
                )
                for k, v in V.items()
            },
            rep=rep,
        )
        out = self.out(V, mode=mode)
        if return_attention_weights:
            return out, weight
        return out
