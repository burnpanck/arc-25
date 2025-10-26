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
from . import mae
from .attention import GlobalAttention
from .encoder import ARCEncoder, LayerStack
from .fields import CoordinateGrid, Field, FieldDims
from .layernorm import SymDecompLayerNorm
from .linear import SymDecompLinear
from .symrep import (
    FlatSymDecomp,
    RepSpec,
    SplitSymDecomp,
    SymDecompBase,
    SymDecompDims,
    standard_rep,
)
from .transformer import FieldTransformer


class ARCSolver(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: FieldDims,
        num_heads: int,
        num_groups: int | None = None,
        dropout_rate: float = 0.0,
        head_rep: type[symmetry.PermRepBase] = symmetry.TrivialRep,
        decoder_cell_width: SymDecompDims | None = None,
        num_decoder_layers: int = 4,
        num_program_tokens: int | None = None,
        mlp_features_per_flavour: int = 4,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
        with_final_norm: bool = True,
        num_latent_programs: int | None = None,
        keep_rngs: bool = True,
        rngs: nnx.Rngs,
        **kw,
    ):
        self.dtype = dtype
        self.hidden_size = hidden_size

        if num_program_tokens is None:
            num_program_tokens = hidden_size.context_tokens

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
            with_final_norm=with_final_norm,
            decoder_cell_width=decoder_cell_width,
            num_program_tokens=num_program_tokens,
            num_latent_programs=num_latent_programs,
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

        decoder_cell_width = None
        dhs = attrs.evolve(
            phs,
            cells=decoder_cell_width if decoder_cell_width is not None else phs.cells,
            context_tokens=phs.context_tokens + num_program_tokens,
        )

        pos_embedding_dims = SymDecompDims(
            invariant=0,
            space=2,
            flavour=0,
            rep=RepSpec(symmetry.AxialDirRep, n_flavours),
        )
        token_init = jax.nn.initializers.truncated_normal(stddev=0.02)

        self.pos_embedding = SymDecompLinear(
            pos_embedding_dims,
            ehs.cells,
            **base_kw,
        )
        self.latent_program_embeddings = (
            nnx.Param(
                token_init(
                    rngs.params(),
                    (
                        num_latent_programs,
                        num_program_tokens,
                        ehs.context.total_channels,
                    ),
                    param_dtype,
                )
            )
            if num_latent_programs is not None
            else nnx_compat.data(None)
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

        self.decoder_hidden_size = dhs
        n_flavours = dhs.rep.n_flavours

        self.decoder = LayerStack(
            num_layers=num_decoder_layers,
            hidden_size=dhs,
            style="active-decoder",
            active_context_tokens=num_program_tokens,
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
        *,
        latent_program_idx: jt.Int[jt.Array, "..."] | None = None,
        latent_program: jt.Float[jt.Array, "... N C"] | None = None,
        with_stats: bool = False,
        with_attention_maps: bool = False,
        **kw,
    ) -> jt.Float[jt.Array, "... F"]:
        skw = dict(
            with_stats=with_stats,
            with_attention_maps=with_attention_maps,
            **kw,
        )
        res = self.encoder(
            x,
            size,
            **skw,
        )

        if with_stats or with_attention_maps:
            encoded, encoder_stats = res
        else:
            encoded = res

        encoded = encoded.map_elementwise(lambda v: jax.lax.stop_gradient(v))

        res = self.decode(
            encoded,
            output_size=size,
            latent_program_idx=latent_program_idx,
            latent_program=latent_program,
            **skw,
        )

        if with_stats or with_attention_maps:
            logits, decoder_stats = res
        else:
            logits = res

        if with_stats or with_attention_maps:
            stats = dict(encoder=encoder_stats, decoder=decoder_stats)
            return logits, stats

        return logits

    def decode(
        self,
        embeddings: Field,
        output_size: jt.Int[jt.Array, "... 2"],
        output_grid: CoordinateGrid | None = None,
        *,
        latent_program_idx: jt.Int[jt.Array, "..."] | None = None,
        latent_program: jt.Float[jt.Array, "... N C"] | None = None,
        with_stats: bool = False,
        with_attention_maps: bool = False,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        **kw,
    ) -> jt.Float[jt.Array, "... F"]:
        assert (latent_program_idx is None) != (latent_program is None)
        cells_shape = embeddings.cells.batch_shape
        batch, (Y, X) = cells_shape[:-2], cells_shape[-2:]

        lin_kw = dict(
            mode=mode,
        )
        enc_kw = dict(
            unroll=unroll,
            remat=remat,
            rngs=rngs,
            **lin_kw,
            **kw,
        )

        if latent_program is None:
            # explicit sharding help for the new explicit sharding mode in JAX
            # (though other parts of the model are still not compatible)
            latent_program = self.latent_program_embeddings.at[latent_program_idx].get(
                out_sharding=jax.typeof(latent_program_idx).sharding
            )
        if output_grid is None:
            output_grid = embeddings.grid

        ecrep = embeddings.cells.rep
        match embeddings.cells:
            case FlatSymDecomp():
                dtype = embeddings.cells.data.dtype
            case SplitSymDecomp():
                dtype = embeddings.cells.invariant.dtype
            case _:
                raise TypeError(type(embeddings.cells).__name__)
        n_flavours = ecrep.n_flavours

        in_context = embeddings.context.as_flat()
        in_context = attrs.evolve(
            in_context,
            data=jnp.concatenate(
                [
                    latent_program.astype(in_context.data.dtype),
                    in_context.data,
                ],
                axis=-2,
            ),
        )

        # TODO: we could probably share this one with the original call within self.encoder above.
        pos_enc, pos_enc_rep = self.encoder.encode_positions(output_size, output_grid)
        pos_enc = SplitSymDecomp(
            invariant=jnp.zeros(batch + (Y, X, 0), dtype),
            flavour=jnp.zeros(batch + (Y, X, n_flavours, 0), dtype),
            space=pos_enc.astype(dtype),
            rep=RepSpec(pos_enc_rep, n_flavours),
        )

        pos_emb = self.pos_embedding(pos_enc, **lin_kw)
        in_cells = embeddings.cells.map_elementwise(lambda a, b: a + b, pos_emb)
        in_context = in_context.coerce_to(in_cells)
        res_path = self.decoder_cell_prep(
            in_cells,
            **lin_kw,
        )
        norm_in_cells = self.decoder_infusion_query_norm(in_cells, **lin_kw)
        ca_path = self.decoder_semantic_infusion(
            target=norm_in_cells.batch_reshape(*norm_in_cells.batch_shape[:-2], -1),
            source=in_context,
            rngs=rngs,
            deterministic=deterministic,
            mode=mode,
        ).batch_reshape(*norm_in_cells.batch_shape)
        y_cells = res_path.map_elementwise(lambda a, b: a + b, ca_path)

        y = Field(
            context=in_context,
            cells=y_cells,
            grid=output_grid,
        )

        res = self.decoder(
            y,
            deterministic=deterministic,
            **enc_kw,
            with_stats=with_stats,
            with_attention_maps=with_attention_maps,
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
            return logits, decoder_stats
        return logits


configs = dict(
    tiny=dict(
        num_program_tokens=2,
        num_decoder_layers=4,
    ),
    small=dict(
        num_program_tokens=8,
        num_decoder_layers=6,
    ),
)

configs = MappingProxyType(
    {
        config_name: MappingProxyType(
            {
                k: v
                for k, v in mae.configs[config_name].items()
                if k not in {"decoder_cell_infusion", "num_decoder_layers"}
            }
            | decoder_config
        )
        for config_name, decoder_config in configs.items()
    }
)
