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


def make_norms(
    features: FieldDims,
    *,
    norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
    active_towers: set[str] = frozenset({"context", "cells"}),
    **kw,
):
    if norm_per == "basis-nnx":
        # Old behavior: one nnx.LayerNorm per representation per projection
        return features.map_representations(
            lambda k, kk, v: (nnx.LayerNorm(v, **kw) if k in active_towers else None),
            cls=nnx_compat.Dict,
        )
    else:
        # New behavior: one SymDecompLayerNorm per projection
        return features.map_projections(
            lambda k, v: (
                SymDecompLayerNorm(
                    v,
                    norm_per=norm_per,
                    **kw,
                )
                if k in active_towers
                else None
            ),
            cls=nnx_compat.Dict,
        )


def apply_norms(
    x: Field,
    norms,
    *,
    norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
    mode: typing.Literal["flat", "split"] | None = None,
    active_towers: set[str] = frozenset({"context", "cells"}),
) -> Field:
    if norm_per == "basis-nnx":
        # Old behavior: apply per-representation norms
        return x.map_representations(
            lambda rep, norm: norm(rep) if norm is not None else rep, norms
        )

    # New behavior: apply per-projection norms
    return attrs.evolve(
        x,
        **{
            k: norm(getattr(x, k), mode=mode)
            for k, norm in norms.items()
            if k in active_towers
        },
    )


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
        rope_freq_scaling: typing.Literal["linear-freqs", "linear-k"] = "linear-freqs",
        learnable_rope_freqs: typing.Literal["per-head", "tied", "none"] | None = None,
        per_head_rope_freq: bool | None = None,
        # attention_dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        dropout_rate: float = 0.0,
        normalise_qk: bool = False,  # True to stabilise learning in ViT-22B; see paper http://arxiv.org/abs/2302.05442
        keep_rngs: bool = True,
        active_context_tokens: int | None = None,
        style: typing.Literal[
            "co-attention", "perceiver", "decoder", "active-decoder"
        ] = "co-attention",
        norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
        rngs: nnx.Rngs,
    ) -> None:
        active_sa = None
        active_ca = None
        match style:
            case "co-attention" | "active-decoder":
                active_towers = {"context", "cells"}
                active_ca = {"cells"}
            case "perceiver":
                active_towers = {"context"}
            case "decoder":
                active_towers = {"cells"}
            case _:
                raise KeyError(style)
        active_towers = frozenset(active_towers)
        active_sa = active_towers if active_sa is None else frozenset(active_sa)
        active_ca = active_towers if active_ca is None else frozenset(active_ca)
        self.active_towers = active_towers
        self.active_sa = active_sa
        self.active_ca = active_ca
        self.active_context_tokens = active_context_tokens
        self.norm_per = norm_per

        def norms(features: FieldDims = hidden_size, **kw):
            return make_norms(
                features,
                norm_per=norm_per,
                active_towers=active_towers,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                **kw,
            )

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
                    rope_freq_scaling=rope_freq_scaling,
                    learnable_rope_freqs=learnable_rope_freqs,
                    per_head_rope_freq=per_head_rope_freq,
                    # broadcast_dropout=False,
                    deterministic=False,
                    normalise_qk=normalise_qk,
                    keep_rngs=keep_rngs,
                    rngs=rngs,
                )
                if "cells" in active_sa
                else None
            ),
            context=(
                GlobalAttention(
                    in_features=hidden_size.context,
                    **global_attn_kw,
                )
                if "context" in active_sa
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
                if "cells" in active_ca
                else None
            ),
            context2cells=(
                GlobalAttention(
                    target_features=hidden_size.context,
                    source_features=hidden_size.cells,
                    **global_attn_kw,
                )
                if "context" in active_ca
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
        with_attention_maps: bool = False,
    ) -> Field:
        batch_shape = x.batch_shape

        active_towers = self.active_towers
        active_sa = self.active_sa
        active_ca = self.active_ca
        nactx = self.active_context_tokens

        def add_resid(x, ax, active):
            return attrs.evolve(
                x,
                **{
                    k: v.map_elementwise(lambda a, b: a + b, getattr(ax, k))
                    for k, v in x.projections.items()
                    if k in active
                },
            )

        # step 1: self-attention
        ax = apply_norms(
            x,
            self.norm1,
            norm_per=self.norm_per,
            active_towers=active_sa,
            mode=mode,
        )
        sa_inp = ax
        sa_maps = {}
        if "cells" in active_sa:
            res = self.self_attn.cells(
                ax.cells,
                grid=ax.grid,
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
                return_attention_weights=with_attention_maps,
            )
            if with_attention_maps:
                cells_out, attn = res
                # cells_attn_weights is a list of 2 tensors [vertical, horizontal]
                # Each has shape ...tsxdvmk or ...ytsdvmk
                # Aggregate over t (query), m, k (heads) and stack along d to get ...{sx,ys}dv
                # Then, stack the two axes along d dimension to get full AxialDirRep (d=4: ↓↑→←)
                sa_maps["cells"] = jnp.concatenate(
                    [w.mean(axis=(axis - 7, -2, -1)) for axis, w in enumerate(attn)],
                    axis=-2,
                )
            else:
                cells_out = res
        else:
            cells_out = ax.cells

        if "context" in active_sa:
            if nactx is not None:
                n = len(batch_shape)
                tgt = ax.context.map_elementwise(
                    lambda v: v[(slice(None),) * n + (slice(None, nactx),)]
                )
            else:
                tgt = ax.context
            res = self.self_attn.context(
                tgt,
                ax.context,
                mask=None,
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
                return_attention_weights=with_attention_maps,
            )
            if with_attention_maps:
                context_out, attn = res
                # context_attn_weight has shape ...tsvmk
                # Aggregate over t (query), m, k (heads) to get ...sv
                sa_maps["context"] = attn.mean(axis=(-5, -2, -1))
            else:
                context_out = res
        else:
            context_out = ax.context

        sa_res = ax = attrs.evolve(ax, cells=cells_out, context=context_out)

        # step 1X: split off static context if needed
        if nactx is not None:
            n = len(batch_shape)
            static_ctxt = x.context.map_elementwise(
                lambda v: v[(slice(None),) * n + (slice(nactx, None),)]
            )
            x = attrs.evolve(
                x,
                context=x.context.map_elementwise(
                    lambda v: v[(slice(None),) * n + (slice(None, nactx),)]
                ),
            )

        sa_out = x = add_resid(x, ax, active_sa)

        # step 2: cross-attention
        ax = apply_norms(
            x,
            self.norm2,
            norm_per=self.norm_per,
            active_towers=active_ca,
            mode=mode,
        )
        ca_inp = ax
        ca_maps = {}
        cells_flat = ax.cells.batch_reshape(*ax.cells.batch_shape[:-2], -1)
        mask = ax.grid.mask

        if "cells" in active_ca:
            res = self.cross_attn.cells2context(
                target=cells_flat,
                source=ax.context,
                mask=mask.reshape(*mask.shape[:-2], -1, 1),
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
                return_attention_weights=with_attention_maps,
            )
            if with_attention_maps:
                cells_out, attn = res
                # cells2context_attn_weight has shape ...tsvmk
                # where t=cell tokens, s=context tokens
                # Aggregate over t (cell query tokens), m, k (heads) to get ...sv
                ca_maps["context"] = attn.mean(axis=(-5, -2, -1))
            else:
                cells_out = res
            cells_out = cells_out.batch_reshape(*ax.cells.batch_shape)
        else:
            cells_out = ax.cells

        if "context" in active_ca:
            res = self.cross_attn.context2cells(
                target=ax.context,
                source=cells_flat,
                mask=mask.reshape(*mask.shape[:-2], 1, -1),
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
                return_attention_weights=with_attention_maps,
            )
            if with_attention_maps:
                context_out, attn = res
                # context2cells_attn_weight has shape ...tsvmk
                # where t=context tokens, s=cell tokens
                # Aggregate over  m, k (heads) and transpose to get ...stv
                attn = jnp.moveaxis(attn.mean(axis=(-2, -1)), -3, -2)
                # then, unflatten `s` to get back the cell array
                ca_maps["cells"] = attn.reshape(*ax.cells.batch_shape, *attn.shape[-2:])
            else:
                context_out = res
        else:
            context_out = ax.context

        ca_res = ax = attrs.evolve(ax, cells=cells_out, context=context_out)
        ca_out = x = add_resid(x, ax, active_ca)

        # step 3: swiglu
        ax = apply_norms(
            x,
            self.norm3,
            norm_per=self.norm_per,
            active_towers=active_towers,
            mode=mode,
        )
        loc_inp = ax
        loc_res = ax = ax.map_projections(
            lambda v, swiglu: (
                swiglu(v, rngs=rngs, deterministic=deterministic, mode=mode)
                if swiglu is not None
                else v
            ),
            self.swiglu,
        )
        x = add_resid(x, ax, active_towers)

        # step 3f: add static context if needed
        if nactx is not None:
            n = len(batch_shape)
            x = attrs.evolve(
                x,
                context=x.context.map_elementwise(
                    lambda a, b: jnp.concatenate([a, b], axis=n),
                    static_ctxt,
                ),
            )

        intermediates = dict()

        if with_residuals:
            intermediates.update(
                sa_inp=sa_inp,
                sa_res=sa_res,
                sa_out=sa_out,
                ca_inp=ca_inp,
                ca_res=ca_res,
                ca_out=ca_out,
                loc_inp=loc_inp,
                loc_res=loc_res,
            )

        if with_attention_maps:
            intermediates.update(
                sa_maps=sa_maps,
                ca_maps=ca_maps,
            )

        if intermediates:
            return x, intermediates

        return x
