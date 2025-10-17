import typing
from types import MappingProxyType, SimpleNamespace

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import nnx
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .. import symmetry
from ..lib import nnx_compat
from .attention import GlobalAttention
from .encoder import ARCEncoder, LayerStack
from .fields import CoordinateGrid, Field, FieldDims
from .layernorm import SymDecompLayerNorm
from .linear import SymDecompLinear
from .symrep import RepSpec, SplitSymDecomp, SymDecompBase, SymDecompDims, standard_rep
from .transformer import FieldTransformer


class MaskedAutoencoder(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: FieldDims,
        num_heads: int,
        num_groups: int | None = None,
        dropout_rate: float = 0.0,
        head_rep: type[symmetry.PermRepBase] = symmetry.TrivialRep,
        num_decoder_layers: int = 4,
        decoder_cell_width: SymDecompDims | None = None,
        mlp_features_per_flavour: int = 4,
        decoder_cell_infusion: typing.Literal[
            "legacy", "backbone+semantic"
        ] = "backbone+semantic",
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
        with_final_norm: bool = True,
        keep_rngs: bool = True,
        rngs: nnx.Rngs,
        **kw,
    ):
        self.dtype = dtype
        self.hidden_size = hidden_size

        base_kw = dict(
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        attn_kw = dict(
            num_heads=num_heads,
            num_groups=num_groups,
            head_rep=head_rep,
            dropout_rate=dropout_rate,
            keep_rngs=keep_rngs,
            **base_kw,
        )

        self.encoder = ARCEncoder(
            num_heads=num_heads,
            num_groups=num_groups,
            head_rep=head_rep,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            norm_per=norm_per,
            **kw,
            **base_kw,
        )
        self.config = dict(self.encoder.config)
        self.config.update(
            num_decoder_layers=num_decoder_layers,
            mlp_features_per_flavour=mlp_features_per_flavour,
            decoder_cell_infusion=decoder_cell_infusion,
            with_final_norm=with_final_norm,
            decoder_cell_width=decoder_cell_width,
        )

        stack_kw = dict(self.encoder.config)
        for k in [
            "num_layers",
            "hidden_size",
            "num_perceiver_layers",
            "num_perceiver_tokens",
        ]:
            stack_kw.pop(k)

        ehs = self.encoder.hidden_size
        phs = self.encoder.perceiver_hidden_size

        n_flavours = phs.rep.n_flavours

        dhs = (
            phs
            if decoder_cell_width is None
            else attrs.evolve(
                phs,
                cells=decoder_cell_width,
            )
        )

        pos_embedding_dims = SymDecompDims(
            invariant=0,
            space=2,
            flavour=0,
            rep=RepSpec(symmetry.AxialDirRep, n_flavours),
        )
        token_init = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.decoder_cell_infusion = decoder_cell_infusion
        match decoder_cell_infusion:
            case "legacy":
                self.mask_embedding = SymDecompLinear(
                    attrs.evolve(pos_embedding_dims, invariant=1),
                    dhs.cells,
                    **base_kw,
                )
            case "backbone+semantic":
                self.pos_embedding = SymDecompLinear(
                    pos_embedding_dims,
                    ehs.cells,
                    **base_kw,
                )
                self.mask_token = nnx.Param(
                    token_init(
                        rngs.params(),
                        (ehs.cells.invariant,),
                        param_dtype,
                    )
                )
                self.decoder_cell_prep = SymDecompLinear(
                    ehs.cells,
                    dhs.cells,
                    **base_kw,
                )
                self.decoder_infusion_query_norm = SymDecompLayerNorm(
                    ehs.cells,
                    norm_per=norm_per,
                    **{k: v for k, v in base_kw.items() if k != "precision"},
                )
                self.decoder_semantic_infusion = GlobalAttention(
                    target_features=ehs.cells,
                    source_features=phs.context,
                    out_features=dhs.cells,
                    **attn_kw,
                )
            case _:
                raise KeyError(
                    f"Uknown value for `decoder_cell_infusion`:{decoder_cell_infusion!r}"
                )

        self.decoder_hidden_size = dhs
        n_flavours = dhs.rep.n_flavours

        self.decoder = LayerStack(
            num_layers=num_decoder_layers,
            hidden_size=dhs,
            style="decoder",
            keep_rngs=keep_rngs,
            rngs=rngs,
            **stack_kw,
        )

        mlp_hidden_dims = SymDecompDims(
            flavour=mlp_features_per_flavour,
            invariant=0,
            space=0,
            rep=RepSpec(symmetry.TrivialRep, n_flavours=n_flavours),
        )

        final_dtype = (
            jnp.promote_types(dtype, jnp.float32) if dtype is not None else jnp.float32
        )

        self.final_norm = (
            SymDecompLayerNorm(
                dhs.cells,
                norm_per=norm_per,
                **{k: v for k, v in base_kw.items() if k != "precision"},
            )
            if with_final_norm
            else None
        )

        self.reduction = SymDecompLinear(
            dhs.cells,
            mlp_hidden_dims,
            dtype=final_dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.classifier_activation = nnx.gelu
        self.classifier = nnx.Linear(
            mlp_features_per_flavour,
            1,
            dtype=final_dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        mask: jt.Bool[jt.Array, "... Y X"],
        *,
        with_stats: bool = False,
        with_attention_maps: bool = False,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        **kw,
    ) -> jt.Float[jt.Array, "... F"]:
        batch = x.shape[:-2]
        Y, X = x.shape[-2:]

        lin_kw = dict(
            mode=mode,
        )
        enc_kw = dict(
            deterministic=deterministic,
            unroll=unroll,
            remat=remat,
            rngs=rngs,
            **lin_kw,
            **kw,
        )
        res = self.encoder(
            x,
            size,
            mask=mask,
            **enc_kw,
            with_stats=with_stats,
            with_attention_maps=with_attention_maps,
        )

        if with_stats or with_attention_maps:
            encoded, encoder_stats = res
        else:
            encoded = res
        grid = encoded.grid
        mask = grid.mask if mask is None else grid.mask & mask

        ecrep = encoded.cells.rep
        dtype = encoded.cells.invariant.dtype
        n_flavours = ecrep.n_flavours

        # TODO: we could probably share this one with the original call within self.encoder above.
        # the only difference is in the mask.
        pos_enc, pos_enc_rep = self.encoder.encode_positions(size, grid)
        pos_enc = SplitSymDecomp(
            invariant=jnp.zeros(batch + (Y, X, 0), dtype),
            flavour=jnp.zeros(batch + (Y, X, n_flavours, 0), dtype),
            space=pos_enc.astype(dtype),
            rep=RepSpec(pos_enc_rep, n_flavours),
        )

        match self.decoder_cell_infusion:
            case "legacy":
                mask_enc = attrs.evolve(
                    pos_enc,
                    invariant=(grid.mask & ~mask).astype(dtype)[..., None],
                )
                y_cells = self.mask_embedding(mask_enc, **lin_kw)
            case "backbone+semantic":
                in_field = attrs.evolve(
                    encoded.cells,
                    invariant=jnp.where(
                        mask[..., None],
                        self.mask_token.astype(dtype),
                        encoded.cells.invariant,
                    ),
                    flavour=jnp.where(
                        mask[..., None, None],
                        0.0,
                        encoded.cells.flavour,
                    ),
                    space=jnp.where(
                        mask[..., None, None],
                        0.0,
                        encoded.cells.space,
                    ),
                )
                pos_emb = self.pos_embedding(pos_enc, **lin_kw)
                in_field = in_field.map_elementwise(lambda a, b: a + b, pos_emb)
                res_path = self.decoder_cell_prep(
                    in_field,
                    **lin_kw,
                )
                norm_in_field = self.decoder_infusion_query_norm(in_field, **lin_kw)
                ca_path = self.decoder_semantic_infusion(
                    target=norm_in_field.batch_reshape(
                        *norm_in_field.batch_shape[:-2], -1
                    ),
                    source=encoded.context,
                    rngs=rngs,
                    deterministic=deterministic,
                    mode=mode,
                ).batch_reshape(*norm_in_field.batch_shape)
                y_cells = res_path.map_elementwise(lambda a, b: a + b, ca_path)
            case _:
                raise KeyError(
                    f"Uknown value for `decoder_cell_infusion`:{self.decoder_cell_infusion!r}"
                )

        y = Field(
            context=encoded.context,
            cells=y_cells,
            grid=grid,
        ).as_split()

        res = self.decoder(
            y, **enc_kw, with_stats=with_stats, with_attention_maps=with_attention_maps
        )
        if with_stats or with_attention_maps:
            decoded, decoder_stats = res
        else:
            decoded = res

        z = decoded.cells
        if self.final_norm is not None:
            z = self.final_norm(z, **lin_kw)

        z = self.reduction(z, **lin_kw).as_split().flavour

        z = self.classifier_activation(z)
        z = self.classifier(z)

        logits = z.reshape(z.shape[:-1])

        if with_stats or with_attention_maps:
            stats = dict(encoder=encoder_stats, decoder=decoder_stats)
            return logits, stats
        return logits


_modern_config_base = dict(
    norm_per="all",
    rope_freq_scaling="linear-freqs",
    learnable_rope_freqs="none",
    decoder_cell_infusion="backbone+semantic",
    use_chirality_rep=False,
    head_rep=symmetry.TrivialRep,
    with_final_norm=True,
)


def decomp_32a(n):
    """n*32 total width, split as 8×(1×n) + 10×(1×n) + 14×n."""
    return SymDecompDims(
        space=1 * n,
        flavour=1 * n,
        invariant=14 * n,
    )


def decomp_48a(n):
    """n*48 total width, split as 8×(2×n) + 10×(1×n) + 22×n."""
    return SymDecompDims(
        space=2 * n,
        flavour=1 * n,
        invariant=22 * n,
    )


def decomp_64a(n):
    """n*64 total width, split as 8×(3×n) + 10×(2×n) + 20×n."""
    return SymDecompDims(
        space=3 * n,
        flavour=2 * n,
        invariant=20 * n,
    )


def decomp_qk_32a_chiral(n):
    """n*32 total width, split as 2×(3×n) + 10×(2×n) + 6×n."""
    return SymDecompDims(
        space=4 * n,
        flavour=1 * n,
        invariant=14 * n,
        rep=RepSpec(symmetry.ChiralityRep, 10),
    )


configs = MappingProxyType(
    dict(
        legacy=dict(
            num_heads=8,
            num_groups=2,
            num_layers=8,
            num_perceiver_layers=3,
            num_perceiver_tokens=8,
            num_decoder_layers=4,
            hidden_size=FieldDims(
                context=SymDecompDims(
                    space=2 * 16,  # 8x2 = 16
                    flavour=1 * 16,  # 10x1 = 10
                    invariant=14 * 16,  # 1x14 -> 40*16
                ),
                cells=SymDecompDims(
                    space=2 * 8,
                    flavour=1 * 8,
                    invariant=22 * 8,  # -> 48*8
                ),
                context_tokens=2,
            ),
            swiglu_width_factor=8 / 3,
            qk_head_width=SymDecompDims(
                space=3 * 8,  # 1x3x8
                flavour=1 * 4,  # 10x1x4 = 5x1x8
                invariant=4 * 8,  # 1*4*8 -> 12x8
                rep=RepSpec(symmetry.ChiralityRep, 10),
            ),
            v_head_width=SymDecompDims(
                space=2 * 4,
                flavour=1 * 4,
                invariant=14 * 4,
            ),
            norm_per="basis-nnx",
            rope_freq_scaling="linear-k",
            learnable_rope_freqs="tied",
            decoder_cell_infusion="legacy",
            use_chirality_rep=False,
            head_rep=symmetry.TrivialRep,
            with_final_norm=False,
        ),
        tiny=dict(
            num_heads=4,
            num_groups=1,
            num_layers=8,
            hidden_size=FieldDims(
                context=decomp_64a(16),  # 64x8 = 512
                cells=decomp_48a(8),  # 48x8 = 384
                context_tokens=2,
            ),
            qk_head_width=decomp_qk_32a_chiral(8),  # 32x8 = 256
            v_head_width=decomp_64a(4),  # 64x4 = 256
            num_perceiver_layers=3,
            num_perceiver_tokens=8,
            num_decoder_layers=3,
            decoder_cell_width=decomp_64a(4),  # 64x4 = 256
            swiglu_width_factor=2,
            **_modern_config_base,
        ),
        small=dict(
            num_heads=8,
            num_groups=2,
            num_layers=12,
            hidden_size=FieldDims(
                context=decomp_64a(12),  # 64x12 = 768
                cells=decomp_64a(8),  # 64x8 = 512
                context_tokens=2,
            ),
            qk_head_width=decomp_qk_32a_chiral(16),  # 32x16 = 512
            v_head_width=decomp_64a(8),  # 64x8 = 512
            num_perceiver_layers=4,
            num_perceiver_tokens=16,
            num_decoder_layers=4,
            decoder_cell_width=decomp_64a(8),  # 64x8 = 512
            swiglu_width_factor=2.5,
            **_modern_config_base,
        ),
    )
)
