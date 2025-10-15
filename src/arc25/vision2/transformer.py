import typing
from types import SimpleNamespace

import attrs
import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .. import symmetry
from ..lib import nnx_compat
from ..symmetry import PermRepBase
from .attention import AxialAttention, GlobalAttention
from .fields import Field, FieldDims
from .layernorm import SymDecompLayerNorm
from .linear import SymDecompDims
from .swiglu import SwiGLU


class FieldTransformer(nnx.Module):
    def __init__(
        self,
        hidden_size: FieldDims,
        *,
        qk_head_width: SymDecompDims | None = None,
        v_head_width: SymDecompDims | None = None,
        swiglu_width_factor: float | None = None,
        num_heads: int,
        num_groups: int | None = None,
        dtype: Dtype | None = None,
        use_chirality_rep: bool = True,
        head_rep: type[PermRepBase] = symmetry.TrivialRep,
        per_head_rope_freq: bool = True,
        # attention_dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        dropout_rate: float = 0.0,
        normalise_qk: bool = False,  # True to stabilise learning in ViT-22B; see paper http://arxiv.org/abs/2302.05442
        keep_rngs: bool = True,
        style: typing.Literal["co-attention", "perceiver", "decoder"] = "co-attention",
        norm: typing.Literal["nnx", "custom"] = "nnx",
        rngs: nnx.Rngs,
    ) -> None:
        match style:
            case "co-attention":
                active_towers = {"context", "cells"}
            case "perceiver":
                active_towers = {"context"}
            case "decoder":
                active_towers = {"cells"}
            case _:
                raise KeyError(style)
        active_towers = frozenset(active_towers)
        self.active_towers = active_towers
        self.norm_style = norm

        def norms(features: FieldDims = hidden_size, **kw):
            match norm:
                case "custom":
                    # New behavior: one SymDecompLayerNorm per projection
                    return features.map_projections(
                        lambda k, v: (
                            SymDecompLayerNorm(
                                v, dtype=dtype, param_dtype=param_dtype, rngs=rngs, **kw
                            )
                            if k in active_towers
                            else None
                        ),
                        cls=nnx_compat.Dict,
                    )
                case "nnx":
                    # Old behavior: one nnx.LayerNorm per representation per projection
                    return features.map_representations(
                        lambda k, kk, v: (
                            nnx.LayerNorm(
                                v, dtype=dtype, param_dtype=param_dtype, rngs=rngs, **kw
                            )
                            if k in active_towers
                            else None
                        ),
                        cls=nnx_compat.Dict,
                    )
                case _:
                    raise KeyError(f"Unknown norm style: {norm!r}")

        global_attn_kw = dict(
            num_heads=num_heads,
            num_groups=num_groups,
            dtype=dtype,
            param_dtype=param_dtype,
            # attention_dtype=attention_dtype,
            precision=precision,
            dropout_rate=dropout_rate,
            head_rep=head_rep,
            # broadcast_dropout=False,
            deterministic=False,
            normalise_qk=normalise_qk,
            keep_rngs=keep_rngs,
            rngs=rngs,
        )

        self.norm1 = norms()
        self.self_attn = nnx_compat.Dict(
            cells=(
                AxialAttention(
                    num_heads=num_heads,
                    num_groups=num_groups,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    # attention_dtype=attention_dtype,
                    precision=precision,
                    in_features=hidden_size.cells,
                    qk_head_width=qk_head_width,
                    v_head_width=v_head_width,
                    dropout_rate=dropout_rate,
                    use_chirality_rep=use_chirality_rep,
                    per_head_rope_freq=per_head_rope_freq,
                    # broadcast_dropout=False,
                    deterministic=False,
                    normalise_qk=normalise_qk,
                    keep_rngs=keep_rngs,
                    rngs=rngs,
                )
                if "cells" in active_towers
                else None
            ),
            context=(
                GlobalAttention(
                    in_features=hidden_size.context,
                    **global_attn_kw,
                )
                if "context" in active_towers
                else None
            ),
        )

        self.norm2 = norms()
        self.cross_attn = nnx_compat.Dict(
            cells2context=(
                GlobalAttention(
                    target_features=hidden_size.cells,
                    source_features=hidden_size.context,
                    **global_attn_kw,
                )
                if "cells" in active_towers
                else None
            ),
            context2cells=(
                GlobalAttention(
                    target_features=hidden_size.context,
                    source_features=hidden_size.cells,
                    **global_attn_kw,
                )
                if "context" in active_towers
                else None
            ),
        )

        self.norm3 = norms()
        self.swiglu = nnx_compat.Dict(
            {
                k: (
                    SwiGLU(
                        v,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        width_factor=swiglu_width_factor,
                        dropout_rate=dropout_rate,
                        keep_rngs=keep_rngs,
                        rngs=rngs,
                    )
                    if k in active_towers
                    else None
                )
                for k, v in hidden_size.projections.items()
            }
        )

    def __call__(
        self,
        x: Field,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        with_residuals: bool = False,
    ) -> Field:
        active_towers = self.active_towers

        def apply(inp, fun, *other):
            return fun(inp, *other) if fun is not None else inp

        def apply_norms(x: Field, norms) -> Field:
            match self.norm_style:
                case "custom":
                    # New behavior: apply per-projection norms
                    return attrs.evolve(
                        x,
                        **{
                            k: apply(getattr(x, k), norm, mode=mode)
                            for k, norm in norms.items()
                            if k in active_towers
                        },
                    )
                case "nnx":
                    # Old behavior: apply per-representation norms
                    return x.map_representations(apply, norms)
                case _:
                    raise KeyError(f"Unknown norm style: {self.norm_style!r}")

        def add_resid(x, ax):
            return attrs.evolve(
                x,
                **{
                    k: v.map_elementwise(lambda a, b: a + b, getattr(ax, k))
                    for k, v in x.projections.items()
                    if k in active_towers
                },
            )

        # step 1: self-attention
        ax = apply_norms(x, self.norm1)
        sa_inp = ax
        sa_res = ax = attrs.evolve(
            ax,
            cells=(
                self.self_attn.cells(
                    ax.cells,
                    grid=ax.grid,
                    rngs=rngs,
                    deterministic=deterministic,
                    mode=mode,
                )
                if "cells" in active_towers
                else ax.cells
            ),
            context=(
                self.self_attn.context(
                    ax.context,
                    mask=None,
                    rngs=rngs,
                    deterministic=deterministic,
                    mode=mode,
                )
                if "context" in active_towers
                else ax.context
            ),
        ).as_split()
        sa_out = x = add_resid(x, ax)

        # step 2: cross-attention
        ax = apply_norms(x, self.norm2)
        ca_inp = ax
        cells_flat = ax.cells.batch_reshape(*ax.cells.batch_shape[:-2], -1)
        mask = ax.grid.mask
        ca_res = ax = attrs.evolve(
            ax,
            cells=(
                self.cross_attn.cells2context(
                    target=cells_flat,
                    source=ax.context,
                    mask=mask.reshape(*mask.shape[:-2], -1, 1),
                    rngs=rngs,
                    deterministic=deterministic,
                    mode=mode,
                ).batch_reshape(*ax.cells.batch_shape)
                if "cells" in active_towers
                else ax.cells
            ),
            context=(
                self.cross_attn.context2cells(
                    target=ax.context,
                    source=cells_flat,
                    mask=mask.reshape(*mask.shape[:-2], 1, -1),
                    rngs=rngs,
                    deterministic=deterministic,
                    mode=mode,
                )
                if "context" in active_towers
                else ax.context
            ),
        ).as_split()
        ca_out = x = add_resid(x, ax)

        # step 3: swiglu
        ax = apply_norms(x, self.norm3)
        loc_inp = ax
        loc_res = ax = ax.map_projections(
            lambda v, swiglu: (
                swiglu(v, rngs=rngs, deterministic=deterministic, mode=mode)
                if swiglu is not None
                else v
            ),
            self.swiglu,
        ).as_split()
        x = add_resid(x, ax)

        if with_residuals:
            return x, dict(
                sa_inp=sa_inp,
                sa_res=sa_res,
                sa_out=sa_out,
                ca_inp=ca_inp,
                ca_res=ca_res,
                ca_out=ca_out,
                loc_inp=loc_inp,
                loc_res=loc_res,
            )

        return x
