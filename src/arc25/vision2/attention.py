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
from ..symmetry import transform_vector
from .linear import SpaceSymmetricTensor, SymDecompLinear
from .rope import QKV, attention_RoPE_with_global
from .symrep import FlatSymDecomp, SplitSymDecomp, SymDecompBase, SymDecompDims


def first_from(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def show_dims(dimnames: str, obj) -> str:
    try:
        shape = obj.shape
    except AttributeError:
        shape = obj
    batch = shape[: -len(dimnames)]
    ret = [str(n) for n in batch] + [
        f"{k}={v}" for k, v in zip(dimnames, shape[-len(dimnames) :])
    ]
    ret = ",".join(ret)
    return f"({ret})"


class AxialAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        in_features: SymDecompDims,
        out_features: SymDecompDims | None = None,
        *,
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
        activation=jax.nn.sigmoid,
        # out_kernel_init: Initializer | None = None,
        bias_init: Initializer = default_bias_init,
        # out_bias_init: Initializer | None = None,
        use_bias: bool | None = None,
        use_qk_bias: bool | None = None,
        use_v_bias: bool | None = None,
        use_out_bias: bool | None = None,
        use_chirality_rep: bool = True,
        # attention_fn: Callable[..., Array] = dot_product_attention,
        normalise_qk: bool = False,
        normalise_pre_out: bool = False,
        keep_rngs: bool = True,
        rngs: rnglib.Rngs,
    ):
        if num_groups is None:
            num_groups = num_heads
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
                    (num_heads // num_groups, num_groups, v // 2, 2),
                    dtype=param_dtype,
                )
            ),
            cls=nnx_compat.Dict,
        )
        self.rope_freq_scaling = qk_head_width.map_representations(
            lambda k, v: np.pi
            / jnp.array(
                [np.linspace(1, 30, v // 2), np.linspace(0.1, 1, v // 2)], dtype=dtype
            ).T,
            cls=nnx_compat.Dict,
        )

    def __call__(
        self,
        inputs: SymDecompBase,
        xpos: jt.Float[jt.Array, "... X 2"],
        ypos: jt.Float[jt.Array, "... Y 2"],
    ) -> SymDecompBase:
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
            print(f"{m}: {inputs.shapes} -> {y.shapes}")
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
        for axis, pos in enumerate([ypos[:, None, :], xpos[None, :, :]]):
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
                print(f"{k} rot: ", show_dims("yxdmkhpq", r))
                rot[k] = r
            qk = []
            for kk in ["q", "k"]:
                for k in rot:
                    vv = qkv[kk][k]
                    print(f"{kk}.{k} rot: ", show_dims("yxadvfmkhp", vv))
                    print(
                        "-> ",
                        show_dims(
                            "yxdvfmkhp", vv[..., :, :, axis, :, :, :, :, :, :, :]
                        ),
                    )
                qk.append(
                    {
                        k: jnp.einsum(
                            "...yxdvfmkhp,yxdmkhpq->...yxdvfmkhq",
                            qkv[kk][k][..., :, :, axis, :, :, :, :, :, :, :],
                            v,
                        )
                        for k, v in rot.items()
                    }
                )
            Q, K = qk
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
            scale = self.activation(logits)
            Vrep = qkv["v"]
            V = {
                k: jnp.einsum(
                    [
                        "...tsxdvmk,...sxdvfmkh->...txdvfmkh",
                        "...ytsdvmk,...ysdvfmkh->...ytdvfmkh",
                    ][axis],
                    scale,
                    Vrep[k],
                )[..., :, :, None, :, :, :, :, :, :]
            }
            all_v.append(V)

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
            },
            rep=rep,
        )
        out = self.out(V)
        return out
