import typing
from types import SimpleNamespace

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import nnx
from flax.nnx.nn.linear import default_bias_init
from flax.typing import (
    Dtype,
    PrecisionLike,
)

from .. import symmetry
from ..lib import nnx_compat
from ..lib.misc import first_from
from .attention import GlobalAttention
from .fields import CoordinateGrid, Field, FieldDims
from .linear import SymDecompLinear
from .symrep import RepSpec, SplitSymDecomp, SymDecompBase, SymDecompDims, standard_rep
from .transformer import FieldTransformer, apply_norms, make_norms


class LayerStack(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: FieldDims,
        num_heads: int,
        num_groups: int | None = None,
        num_layers: int = 12,
        style: typing.Literal["co-attention", "perceiver", "decoder"] = "co-attention",
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        swiglu_width_factor: float | None = None,
        dropout_rate: float = 0.1,
        keep_rngs: bool = True,
        remat: bool | None = True,
        unroll: int | bool | None = 1,
        rngs: nnx.Rngs,
        head_rep: type[symmetry.PermRepBase] = symmetry.TrivialRep,
        # the following ones do not apply to perceiver_only stacks
        use_chirality_rep: bool = True,
        per_head_rope_freq: bool = True,
        norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
        qk_head_width: SymDecompDims | None = None,
        v_head_width: SymDecompDims | None = None,
    ):
        # As Block contains dropout op, we prefer
        # to split RNG into num_layers of RNGs
        # using @nnx.split_rngs decorator.
        # Next, nnx.vmap creates a vectorized version of Block.
        # in_axes and out_axes define vectorization axis
        # of the input splitted rngs and the output Block instance.
        # Both axes should be 0.
        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_block(rngs: nnx.Rngs):
            return FieldTransformer(
                hidden_size=hidden_size,
                qk_head_width=qk_head_width,
                v_head_width=v_head_width,
                swiglu_width_factor=swiglu_width_factor,
                use_chirality_rep=use_chirality_rep,
                per_head_rope_freq=per_head_rope_freq,
                num_heads=num_heads,
                num_groups=num_groups,
                dropout_rate=dropout_rate,
                dtype=dtype,
                param_dtype=param_dtype,
                # attention_dtype=attention_dtype,
                precision=precision,
                style=style,
                head_rep=head_rep,
                norm_per=norm_per,
                # we can't hold on to the Rngs, as we want to carry it through the scan later
                keep_rngs=False,
                rngs=rngs,
            )

        self.num_layers = num_layers
        self.dtype = dtype
        self.blocks = create_block(rngs)
        self.hidden_size = hidden_size
        self.remat = remat
        self.unroll = unroll
        self.rngs = rngs if keep_rngs else nnx_compat.data(None)

    def __call__(
        self,
        x: Field,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        with_stats: bool = False,
        with_attention_maps: bool = False,
    ) -> Field | tuple[Field, dict]:
        assert self.hidden_size.validate(x), self.hidden_size.validation_problems(x)

        # Transformer encoder blocks.
        # Process the embedded patches through the transformer encoder layers.

        # We use nnx.scan to apply sequentially the blocks
        # on the input, for example with num_layers=3
        # output = block[0](x)
        # output = block[1](output)
        # output = block[2](output)
        #
        # In `forward` function defined below:
        # - x represents the loop carry value
        # - model is the data to scan along the leading axis
        # nnx.scan args:
        # - in_axes marks the inputs: x is marked as carry
        # and the model is to scan along the axis 0
        # - out_axes marks the output as carry

        def update_intermediates(intermediates, i, x, layer_attn_maps=None):
            all_batch = (slice(None),) * len(batch_shape)
            if with_stats:
                stats = intermediates["stats"]
                for pk, proj in x.projections.items():
                    for rk, rep in proj.representations.items():
                        if not rep.size:
                            continue
                        for k, v in dict(
                            min=rep.min(),
                            mean=rep.mean(),
                            std=rep.std(),
                            max=rep.max(),
                        ).items():
                            stats[pk][rk][k] = (
                                stats[pk][rk][k].at[all_batch + (i,)].set(v)
                            )
            if with_attention_maps and layer_attn_maps is not None:
                attn_maps = intermediates["attention_maps"]
                for k, v in layer_attn_maps.items():
                    for kk, vv in v.items():
                        attn_maps[k][kk] = (
                            attn_maps[k][kk].at[all_batch + (i - 1,)].set(vv)
                        )
            return intermediates

        def forward(
            carry: tuple[int, Field, dict, nnx.Rngs], block: FieldTransformer
        ) -> tuple[int, Field, dict, nnx.Rngs]:
            i, x, intermediates, rngs = carry
            res = block(
                x,
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
                with_attention_maps=with_attention_maps,
            )
            layer_attn_maps = None
            if with_attention_maps:
                x, layer_attn_maps = res
            else:
                x = res
            i += 1
            intermediates = update_intermediates(intermediates, i, x, layer_attn_maps)
            carry = i, x, intermediates, rngs
            return carry

        remat = nnx.module.first_from(
            remat,
            self.remat,
            error_msg="LayerStack needs `remat`, either in constructor or call",
        )
        if remat is not False:
            forward = nnx.remat(
                forward,
                prevent_cse=False,
                policy=(
                    jax.checkpoint_policies.checkpoint_dots if remat is True else remat
                ),
            )
        unroll = nnx.module.first_from(
            unroll,
            self.unroll,
            error_msg="LayerStack needs `unroll`, either in constructor or call",
        )
        forward = nnx.scan(
            forward, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry, unroll=unroll
        )

        rngs = nnx.module.first_from(
            rngs,
            self.rngs,
            error_msg="""No `rngs` argument was provided to LayerStack
as either a __call__ argument or class attribute""",
        )

        batch_shape = np.broadcast_shapes(
            x.cells.batch_shape[:-2],
            x.context.batch_shape[:-1],
        )

        intermediates = {}
        if with_stats:
            stats = {}
            for pk, proj in x.projections.items():
                for rk, rep in proj.representations.items():
                    if not rep.size:
                        continue
                    s = stats.setdefault(pk, {}).setdefault(rk, {})
                    for k in ["min", "mean", "std", "max"]:
                        s[k] = jnp.zeros(
                            batch_shape + (self.num_layers + 1,), self.dtype
                        )
            intermediates["stats"] = stats

        if with_attention_maps:
            # Run first block to determine attention map shapes
            first_block = jax.tree.map(lambda x: x[0], self.blocks)
            _, first_attn_maps = first_block(
                x,
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
                with_attention_maps=True,
            )
            # Pre-allocate arrays with shape (*batch, num_layers, ...)
            attn_maps = {
                k: {
                    kk: jnp.zeros(
                        batch_shape + (self.num_layers,) + vv.shape[len(batch_shape) :],
                        vv.dtype,
                    )
                    for kk, vv in v.items()
                }
                for k, v in first_attn_maps.items()
            }
            intermediates["attention_maps"] = attn_maps

        intermediates = update_intermediates(intermediates, 0, x)
        carry = (0, x, intermediates, rngs)
        carry = forward(carry, self.blocks)
        i, y, intermediates, rngs = carry

        if with_stats or with_attention_maps:
            return y, intermediates
        return y


class ARCEncoder(nnx.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_groups: int | None = None,
        hidden_size: FieldDims,
        num_layers: int = 12,
        num_perceiver_layers: int = 4,
        num_perceiver_tokens: int = 24,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        qk_head_width: SymDecompDims | None = None,
        v_head_width: SymDecompDims | None = None,
        swiglu_width_factor: float | None = None,
        use_chirality_rep: bool = True,
        per_head_rope_freq: bool = True,
        head_rep: type[symmetry.PermRepBase] = symmetry.TrivialRep,
        norm_per: typing.Literal["basis-nnx", "all", "rep", "basis"] = "all",
        dropout_rate: float = 0.1,
        keep_rngs: bool = True,
        remat: bool | None = True,
        unroll: int | bool | None = 1,
        rngs: nnx.Rngs,
    ):
        self.config = dict(
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_perceiver_layers=num_perceiver_layers,
            num_perceiver_tokens=num_perceiver_tokens,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            qk_head_width=qk_head_width,
            v_head_width=v_head_width,
            swiglu_width_factor=swiglu_width_factor,
            use_chirality_rep=use_chirality_rep,
            per_head_rope_freq=per_head_rope_freq,
            head_rep=head_rep,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            remat=remat,
            unroll=unroll,
        )

        self.num_layers = num_layers
        self.norm_per = norm_per = norm_per

        self.hidden_size = hidden_size
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.num_perceiver_layers = num_perceiver_layers
        self.perceiver_hidden_size = perceiver_hidden_size = attrs.evolve(
            hidden_size, context_tokens=num_perceiver_tokens
        )
        self.dtype = dtype
        self.rngs = rngs if keep_rngs else nnx_compat.data(None)
        if False:
            self.param_dtype = param_dtype
            self.precision = precision

        embedding_dims = dict(
            context=SymDecompDims(
                invariant=0,
                space=1,
                flavour=2,
                rep=RepSpec(symmetry.AxisRep, hidden_size.context.n_flavours),
            ),
            cells=SymDecompDims(
                invariant=2,
                space=2,
                flavour=1,
                rep=RepSpec(symmetry.AxialDirRep, hidden_size.cells.n_flavours),
            ),
        )
        self.embedding = hidden_size.map_projections(
            lambda k, v: SymDecompLinear(
                embedding_dims[k],
                v,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                use_bias=k != "context",
            ),
            cls=nnx_compat.Dict,
        )
        self.context_tokens = nnx.Param(
            default_bias_init(
                rngs.params(),
                (hidden_size.context_tokens, hidden_size.context.invariant),
                param_dtype,
            )
        )
        attn_kw = dict(
            num_heads=num_heads,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            head_rep=head_rep,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            keep_rngs=keep_rngs,
            rngs=rngs,
        )

        stack_kw = dict(
            remat=remat,
            unroll=unroll,
            **attn_kw,
        )

        self.main_stack = LayerStack(
            num_layers=num_layers,
            hidden_size=hidden_size,
            qk_head_width=qk_head_width,
            v_head_width=v_head_width,
            swiglu_width_factor=swiglu_width_factor,
            use_chirality_rep=use_chirality_rep,
            per_head_rope_freq=per_head_rope_freq,
            style="co-attention",
            norm_per=norm_per,
            **stack_kw,
        )

        self.main_norm = make_norms(
            hidden_size,
            norm_per=norm_per,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.perceiver_queries = nnx.Param(
            default_bias_init(
                rngs.params(),
                (
                    max(
                        0,
                        perceiver_hidden_size.context_tokens
                        - hidden_size.context_tokens,
                    ),
                    perceiver_hidden_size.context.invariant,
                ),
                param_dtype,
            )
        )
        self.perceiver_init_attn = (
            GlobalAttention(
                target_features=attrs.evolve(
                    perceiver_hidden_size.context, space=0, flavour=0
                ),
                source_features=hidden_size.context,
                out_features=perceiver_hidden_size.context,
                **attn_kw,
            )
            if self.perceiver_queries.size
            else nnx_compat.data(None)
        )

        self.perceiver_stack = LayerStack(
            num_layers=num_perceiver_layers,
            hidden_size=perceiver_hidden_size,
            style="perceiver",
            norm_per=norm_per,
            **stack_kw,
        )

        self.final_norm = make_norms(
            perceiver_hidden_size,
            norm_per=norm_per,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            active_towers={"context"},
        )["context"]

    def __call__(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        mask: jt.Bool[jt.Array, "... Y X"] | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
        unroll: int | bool | None = None,
        remat: bool | None = None,
        mode: typing.Literal["flat", "split"] | None = None,
        with_stats: bool = False,
        with_attention_maps: bool = False,
    ) -> Field:
        """
        Encodes an ARC-style input grid into a `SymDecomp` of dimensions `(..., F, R?, C)`;
        Here, `F = num_colours+1`, where the first flavour is colour-indepenent, and
        the others map to the input colours.
        """
        pre_embedding = self.encode(x, size, mask=mask)

        # print(f"{pre_embedding.shapes=}")
        embedding = pre_embedding.map_projections(lambda v, f: f(v), self.embedding)

        dtype = nnx.nn.dtypes.canonicalize_dtype(
            embedding.context.invariant, self.context_tokens, dtype=self.dtype
        )
        context_tokens = self.context_tokens.value
        if dtype is not None:
            context_tokens = context_tokens.astype(dtype)

        # print(f"{embedding.shapes=}")
        embedding = attrs.evolve(
            embedding,
            context=attrs.evolve(
                embedding.context,
                invariant=embedding.context.invariant + context_tokens,
            ),
        )

        # Apply the dropout layer to embedded patches.
        embedding = embedding.map_representations(
            self.dropout, rngs=rngs, deterministic=deterministic
        )

        intermediates = {}
        res = self.main_stack(
            embedding,
            rngs=rngs,
            deterministic=deterministic,
            mode=mode,
            with_stats=with_stats,
            with_attention_maps=with_attention_maps,
        )
        if with_stats or with_attention_maps:
            x, inter = res
            intermediates["main"] = inter
        else:
            x = res

        # Apply layer normalization
        y = apply_norms(
            x,
            self.main_norm,
            norm_per=self.norm_per,
            active_towers={"context", "cells"},
            mode=mode,
        )

        if self.perceiver_init_attn is not None:
            n = self.perceiver_queries.shape[0]
            dtype = self.dtype
            rep = self.perceiver_hidden_size.context.rep
            target = SplitSymDecomp(
                invariant=self.perceiver_queries.astype(dtype),
                space=jnp.zeros((n, rep.n_space, 0), dtype),
                flavour=jnp.zeros((n, rep.n_flavours, 0), dtype),
                rep=rep,
            )
            source = y.context
            ptok = self.perceiver_init_attn(
                target=target,
                source=source,
                rngs=rngs,
                deterministic=deterministic,
                mode=mode,
            )
            nB = len(y.context.batch_shape) - 1
            context = y.context.map_elementwise(
                lambda a, b: jnp.concatenate([a, b], axis=nB), ptok.as_split()
            )
            y = attrs.evolve(
                y,
                context=context,
            )
        else:
            context = y.context.map_elementwise(
                lambda a: a[
                    (slice(None),) * (len(y.context.batch_shape) - 1)
                    + (slice(self.perceiver_hidden_size.context_tokens),)
                ]
            )
            y = attrs.evolve(
                y,
                context=context,
            )

        res = self.perceiver_stack(
            y,
            rngs=rngs,
            deterministic=deterministic,
            mode=mode,
            with_stats=with_stats,
            with_attention_maps=with_attention_maps,
        )
        if with_stats or with_attention_maps:
            y, inter = res
            intermediates["perceiver"] = inter
        else:
            y = res

        # final normalisation
        y = apply_norms(
            y,
            (
                SimpleNamespace(
                    cells=SimpleNamespace(invariant=None, space=None, flavour=None),
                    context=self.final_norm,
                )
                if self.norm_per == "basis-nnx"
                else dict(context=self.final_norm)
            ),
            norm_per=self.norm_per,
            active_towers={"context"},
            mode=mode,
        )

        if intermediates:
            return y, intermediates
        return y

    @classmethod
    def encode_positions(
        cls, size: jt.Int[jt.Array, "... 2"], grid: CoordinateGrid
    ) -> tuple[jt.Float[jt.Array, "... Y X R "], symmetry.PermRepBase]:
        def encd(arr, abso=0, relo=0):
            # input is (absolute, relative)
            return jnp.concatenate(
                [
                    1 / jnp.maximum(1, 1 + arr[..., :1] + abso),
                    jnp.clip(arr[..., 1:] + relo, 0, 1),
                ],
                axis=-1,
            )

        batch = np.broadcast_shapes(
            size.shape[:-1],
            grid.xpos.shape[:-2],
            grid.ypos.shape[:-2],
            grid.mask.shape[:-2],
        )
        Y, X = grid.mask.shape[-2:]

        rep = symmetry.AxialDirRep
        encoded_dir_info = {
            rep.d: encd(grid.ypos[..., :, None, :]),
            rep.u: encd(-grid.ypos[..., :, None, :], size[..., None, None, :1] - 1, 1),
            rep.r: encd(grid.xpos[..., None, :, :]),
            rep.l: encd(-grid.xpos[..., None, :, :], size[..., None, None, 1:] - 1, 1),
        }
        cells_space = jnp.where(
            grid.mask[..., None, None],
            jnp.concatenate(
                [
                    jnp.broadcast_to(encoded_dir_info[basis], batch + (Y, X, 2))[
                        ..., :, :, None, :
                    ]
                    for basis in rep
                ],
                axis=-2,
            ),
            0,
        )
        return cells_space, rep

    def encode(
        self,
        x: jt.Int[jt.Array, "... Y X"],
        size: jt.Int[jt.Array, "... 2"],
        mask: jt.Bool[jt.Array, "... Y X"] | None = None,
    ) -> Field:
        batch = x.shape[:-2]
        Y, X = shape = x.shape[-2:]  # noqa: F841
        hs = self.hidden_size
        T = hs.context_tokens

        grid = CoordinateGrid.for_batch(Y, X, size)

        mask = grid.mask if mask is None else grid.mask & mask

        dtype = self.dtype or jnp.float32

        x = x[..., :, :, None]
        # print(f"{x.shape=} {grid.mask.shape=}")

        colour_idx = np.r_[: hs.rep.n_flavours]
        presence = (x == colour_idx) & mask[..., None]
        count = presence.sum((-3, -2)).astype(dtype)
        prevalence = count / (1 + mask.astype(dtype).sum((-2, -1))[..., None]).astype(
            dtype
        )
        intensity = 1 / (1 + count)
        # print(
        #    f"{presence.shape=} {count.shape=} {grid.mask.shape=} {mask.shape=} {prevalence.shape=} {intensity.shape=}"
        # )
        # size comes in as (height, width)
        axl = [symmetry.AxisRep.v, symmetry.AxisRep.h]
        context = SplitSymDecomp(
            invariant=jnp.empty(batch + (T, 0), dtype),
            flavour=jnp.tile(
                jnp.concatenate(
                    [prevalence[..., None], intensity[..., None]], dtype=dtype, axis=-1
                )[..., None, :, :],
                (T, 1, 1),
            ),
            space=jnp.tile(
                size.astype(dtype)[
                    ..., None, [axl.index(basis) for basis in symmetry.AxisRep], None
                ],
                (T, 1, 1),
            ),
            rep=RepSpec(symmetry.AxisRep, hs.context.n_flavours),
        )

        cells_flavour = presence[..., :, :, :, None].astype(dtype)
        cells_invariant = jnp.where(
            mask[..., None],
            jnp.concatenate(
                [
                    jnp.take_along_axis(prevalence[..., None, None, :], x, axis=-1),
                    jnp.take_along_axis(intensity[..., None, None, :], x, axis=-1),
                ],
                dtype=dtype,
                axis=-1,
            ),
            0,
        )

        cells_space, cells_space_rep = self.encode_positions(size, grid)

        cells = SplitSymDecomp(
            invariant=cells_invariant,
            flavour=cells_flavour,
            space=cells_space,
            rep=RepSpec(cells_space_rep, hs.cells.n_flavours),
        )
        # print(f"{context.shapes=} {cells.shapes=}")
        return Field(
            context=context,
            cells=cells,
            grid=grid,
        )
