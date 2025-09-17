from types import SimpleNamespace

import attrs
from flax import nnx

from .attention import FieldAttention
from .fields import Field, FieldDims
from .linear import SymmetricLinear


class FieldMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: FieldDims,
        *,
        mlp_width_factor: float,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ):
        def make_linear(in_feat: FieldDims, out_feat: FieldDims, **kw):
            return in_feat.map_projections(
                lambda k, v, o: SymmetricLinear(v, o, rngs=rngs),
                out_feat,
            )

        mlp_dim = attrs.evolve(
            hidden_size,
            **{
                k: attrs.evolve(
                    v,
                    **{
                        kk: int(round(vv * mlp_width_factor))
                        for kk, vv in v.representations.items()
                    },
                )
                for k, v in hidden_size.projections.items()
            },
        )

        self.widen = make_linear(hidden_size, mlp_dim)
        self.activation = nnx.gelu
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.narrow = make_linear(mlp_dim, hidden_size)

    def __call__(self, x: Field) -> Field:
        def apply(inp, fun, *other):
            return fun(inp, *other)

        x = x.map_projections(apply, self.widen)
        x = x.map_representations(self.activation)
        x = x.map_representations(self.dropout)
        x = x.map_projections(apply, self.narrow)
        x = x.map_representations(self.dropout)
        return x


class FieldTransformer(nnx.Module):
    def __init__(
        self,
        hidden_size: FieldDims,
        mha_features: int,
        *,
        mlp_width_factor: float,
        num_heads: int,
        num_groups: int | None = None,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ) -> None:
        def norms(features: FieldDims = hidden_size, **kw):
            return features.map_representations(
                lambda k, kk, v: nnx.LayerNorm(v, rngs=rngs, **kw)
            )

        # First layer normalization using `flax.nnx.LayerNorm`
        # before we apply Multi-Head Attentn.
        self.norm1 = norms()
        # The Multi-Head Attention layer
        self.attn = FieldAttention(
            num_heads=num_heads,
            num_groups=num_groups,
            in_features=hidden_size,
            qkv_features=mha_features,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            deterministic=False,
            normalize_qk=False,  # True to stabilise learning in ViT-22B; see paper http://arxiv.org/abs/2302.05442
            rngs=rngs,
        )
        # Second layer normalization using `flax.nnx.LayerNorm`.
        self.norm2 = norms()

        # The MLP for point-wise feedforward (using `flax.nnx.Sequential`, `flax.nnx.Linear, flax.nnx.Dropout`)
        # with the GeLU activation function (`flax.nnx.gelu`).
        self.mlp = FieldMLP(
            hidden_size=hidden_size,
            mlp_width_factor=mlp_width_factor,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

    def __call__(self, x: Field) -> Field:
        def apply(inp, fun, *other):
            return fun(inp, *other)

        # The Multi-Head Attention layer with layer normalization.
        ax = x.map_representations(apply, self.norm1)
        ax = self.attn(ax)
        x = x.map_representations(lambda a, b: a + b, ax)

        # The feed-forward network with layer normalization.
        ax = x.map_representations(apply, self.norm2)
        ax = self.mlp(ax)
        x = x.map_representations(lambda a, b: a + b, ax)
        return x
