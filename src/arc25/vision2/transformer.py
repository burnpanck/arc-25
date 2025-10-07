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
        rngs: nnx.Rngs,
    ) -> None:
        def norms(features: FieldDims = hidden_size, **kw):
            return features.map_representations(
                lambda k, kk, v: nnx.LayerNorm(
                    v, dtype=dtype, param_dtype=param_dtype, rngs=rngs, **kw
                ),
                cls=nnx_compat.Dict,
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
            cells=AxialAttention(
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
            ),
            context=GlobalAttention(
                in_features=hidden_size.context,
                **global_attn_kw,
            ),
        )

        self.norm2 = norms()
        self.cross_attn = nnx_compat.Dict(
            cells2context=GlobalAttention(
                target_features=hidden_size.cells,
                source_features=hidden_size.context,
                **global_attn_kw,
            ),
            context2cells=GlobalAttention(
                target_features=hidden_size.context,
                source_features=hidden_size.cells,
                **global_attn_kw,
            ),
        )

        self.norm3 = norms()
        self.swiglu = nnx_compat.Dict(
            {
                k: SwiGLU(
                    v,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    width_factor=swiglu_width_factor,
                    dropout_rate=dropout_rate,
                    keep_rngs=keep_rngs,
                    rngs=rngs,
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
    ) -> Field:
        def apply(inp, fun, *other):
            return fun(inp, *other)

        # step 1: self-attention
        ax = x.map_representations(apply, self.norm1)
        ax = attrs.evolve(
            ax,
            cells=self.self_attn.cells(
                ax.cells,
                grid=ax.grid,
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
            ),
            context=self.self_attn.context(
                ax.context, mask=None, rngs=rngs, deterministic=deterministic, mode=mode
            ),
        )
        x = x.map_representations(lambda a, b: a + b, ax)

        # step 2: cross-attention
        ax = x.map_representations(apply, self.norm2)
        cells_flat = ax.cells.batch_reshape(*ax.cells.batch_shape[:-2], -1)
        mask = ax.grid.mask
        ax = attrs.evolve(
            ax,
            cells=self.cross_attn.cells2context(
                target=cells_flat,
                source=ax.context,
                mask=mask.reshape(*mask.shape[:-2], -1, 1),
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
            ).batch_reshape(*ax.cells.batch_shape),
            context=self.cross_attn.context2cells(
                target=ax.context,
                source=cells_flat,
                mask=mask.reshape(*mask.shape[:-2], 1, -1),
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
            ),
        )
        x = x.map_representations(lambda a, b: a + b, ax)

        # step 3: swiglu
        ax = x.map_representations(apply, self.norm3)
        ax = x.map_projections(
            lambda v, swiglu: swiglu(
                v, rngs=rngs, deterministic=deterministic, mode=mode
            ),
            self.swiglu,
        )
        x = x.map_representations(lambda a, b: a + b, ax)

        return x
