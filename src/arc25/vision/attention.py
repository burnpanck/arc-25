r"""
## Symmetry-preserving axial attention
Applying any point-symmetry to the input field and representation
should be equivalent to applying the point symmetry to the output field and representation

Formally, axial attention computes:
$$
Y(t) = \sum_r |r\rangle \int_\delta p_r[t,\delta] \langle r| V(t+\delta),
\quad\text{where}\quad
p_r[\delta] = w_r[\delta] \cdot \mathrm{softmax}_\delta \left[ \langle r| Q(t)) \cdot \phi_r(\delta) \langle r| K(t+\delta) \right]
$$

Furthermore, applying the operation $s$ to an input field $X(p)$ has the following effect:
$$
X'(p) = R_s X(R_s^{-1} p) = \sum_{uv} |u\rangle \langle u| R_s |v\rangle \langle v| X(R_s^{-1} p)
= \sum_v |s \cdot v\rangle \langle v| X(R_s^{-1} p),
$$
where the last equality holds for representation labels matching symmetry operations.

Thus, we require the following to hold:
$$
\begin{align}
Y'(t) &= \sum_r |r\rangle \int_\delta p'_r[t,\delta] \langle r| V'(t+\delta) \\
&= \sum_r |r\rangle \int_\delta p'_r[t,\delta] \langle s^{-1} r| V(R_s^{-1} (t+\delta)) \\
&= \sum_{r'} |s\cdot r'\rangle \int_\delta p'_{s\cdot r'}[t,\delta] \langle r'| V(R_s^{-1} (t+\delta)) \\
&= \sum_{r'} |s\cdot r'\rangle \int_{\delta'} p'_{s\cdot r'}[t,R_s \delta'] \langle r'| V(R_s^{-1} t+\delta') \\
&= R_s \sum_{r'} |r'\rangle \int_{\delta'} p'_{s\cdot r'}[t,R_s \delta'] \langle r'| V(R_s^{-1} t+\delta') \\
&= R_s Y(R_s^{-1} t).
\end{align}
$$

A sufficient condition for the last equality is $p'_{s\cdot r'}[t,R_s \delta'] = p_{r'}[R_s^{-1} t,\delta']$ everywhere.
Starting from the definition, we have
$$
\begin{align}
p'_{s\cdot r'}[t, R_s \delta'] &= w_{s\cdot r'}[R_s \delta'] \cdot
\mathrm{softmax}_\delta \left[ \langle s\cdot r'| Q'(t))
\cdot \phi_{s\cdot r'}(R_s \delta') \langle s\cdot r'| K'(t+R_s \delta') \right] \\
&= w_{s\cdot r'}[R_s \delta'] \cdot
\mathrm{softmax}_\delta \left[ \langle r'| Q(R_s^{-1} t))
\cdot \phi_{s\cdot r'}(R_s \delta') \langle r'| K(R_s^{-1} t + \delta') \right] \\
\end{align}
$$
thus, again, a sufficient condition are $w_{s\cdot r'}[R_s \delta'] = w_{r'}[\delta']$
and $\phi_{s\cdot r'}(R_s \delta')=\phi_{r'}(\delta')$.
This can be implemented by setting $\phi_r(\delta) = \vec \delta \cdot R_r \hat u$, and suitable rotation of $w$.
"""

from types import SimpleNamespace

import attrs
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn.linear import default_bias_init, default_kernel_init, initializers
from flax.typing import (
    Dtype,
    Initializer,
    PrecisionLike,
)

from ..dsl.types import Vector
from ..symmetry import transform_vector
from .fields import Field, FieldDims
from .linear import SymmetricLinear
from .rope import QKV, attention_RoPE_with_global
from .symrep import SymRep


class FieldAttention(nnx.Module):
    """
    This module performs axial attention.

    For the "e" component of the representation,
    we have chosen an arbitrary axis along which to perform the attention;
    it determines the axis of attention for all other components.
    With trainable frequencies (allowing negative ones),
    revesing the direction would be equivalent, but rotations by 90° arent.
    Thus, with this choice, we break the symmetry within the
    representation. This is fine, if we do this only in one place.
    Otherwise, we'd have to augment this with a second attention axis.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: FieldDims,
        qkv_features: int,
        out_features: FieldDims | None = None,
        *,
        global_mix_reduction: int = 4,
        num_groups: int | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        attention_dtype: Dtype | None = None,
        # broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        # out_kernel_init: Initializer | None = None,
        bias_init: Initializer = default_bias_init,
        # out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        hdrs_attend: bool = False,
        # attention_fn: Callable[..., Array] = dot_product_attention,
        normalize_qk: bool = False,
        keep_rngs: bool = True,
        rngs: rnglib.Rngs,
    ):
        if num_groups is None:
            num_groups = num_heads

        if out_features is None:
            out_features = in_features
        assert not qkv_features % (2 * num_heads)
        assert not num_heads % num_groups
        self.n_features = n_features = qkv_features // (2 * num_heads)
        self.global_mix_reduction = global_mix_reduction
        self.in_features = in_features
        self.qkv_features = qkv_features
        self.out_features = out_features
        self.hdrs_attend = hdrs_attend
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.dtype = dtype
        self.attention_dtype = attention_dtype
        self.param_dtype = param_dtype
        self.precision = precision

        # frequency is both per group, per features, and linear in both absolute and relative
        kernel_key = rngs.params()
        freq_init = initializers.normal(1)
        self.freqs = nnx.Param(
            freq_init(kernel_key, (num_groups, n_features, 2), param_dtype)
        )

        def make_linear(inf, outf, *, cls=SymmetricLinear):
            # TODO: do we have all relevant parameters?
            return cls(
                inf,
                outf,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                use_bias=use_bias,
                precision=precision,
                rngs=rngs,
            )

        nqkv = (num_heads + 2 * num_groups) * n_features * 2

        self.cell_global_mix = make_linear(
            in_features.context.inv,
            in_features.cells.inv // global_mix_reduction,
            cls=nnx.Linear,
        )
        mixmap = dict(
            cells=self.cell_global_mix.out_features,
        )

        if hdrs_attend:
            (hdr_inv,) = {in_features.rows.inv, in_features.cols.inv}
            hdr_mix = hdr_inv // global_mix_reduction
            self.hdr_global_mix = make_linear(
                in_features.context.inv,
                hdr_mix,
                cls=nnx.Linear,
            )
            mixmap.update(rows=hdr_mix, cols=hdr_mix)
            skip_inv = []
        else:
            self.hdr_global_mix = nnx.data(None)
            skip_inv = ["rows", "cols"]
        self.qkv = in_features.map_projections(
            lambda k, v: make_linear(
                attrs.evolve(v, inv=v.inv + mixmap.get(k, 0)),
                attrs.evolve(v, inv=nqkv if k not in skip_inv else 0, equiv=nqkv),
            ),
            cls=dict,
        )
        nv = num_heads * n_features * 2
        self.out = {
            k: make_linear(
                attrs.evolve(
                    v.out_features,
                    inv=dict(context=2 * nv, cells=nv).get(k, nv if hdrs_attend else 0),
                    equiv=nv,
                ),
                getattr(out_features, k),
            )
            for k, v in self.qkv.items()
        }

    def __call__(
        self,
        features: Field,
        *,
        rngs: nnx.Rngs | None = None,
        deterministic: bool | None = None,
    ) -> Field:
        assert self.in_features.validate(
            features
        ), self.in_features.validation_problems(features)
        assert features.context.rep == features.cells.rep

        R = features.cells.rep.dim
        batch = features.cells.equiv.shape[:-5]
        B = int(np.prod(batch))
        Y, X, F = features.cells.equiv.shape[-5:-2]
        H = self.n_features
        D = 2 * H
        K = self.num_groups
        N = self.num_heads

        # print(f"{batch=} {B=} {Y=} {X=} {F=} {R=} {N=} {K=} {H=} {D=}")

        inputs = [
            r for p in features.projections.values() for r in p.representations.values()
        ]
        dtype = nnx.nn.dtypes.canonicalize_dtype(*inputs, dtype=self.dtype)
        attention_dtype = nnx.nn.dtypes.canonicalize_dtype(
            *inputs, dtype=self.attention_dtype
        )
        attention_dtype = jnp.promote_types(attention_dtype, jnp.float32)

        freqs = (self.freqs).astype(dtype)
        precision = self.precision

        # `o` dimension: singular dimension to add as broadcast for "other" spatial axis
        xphi = jnp.einsum(
            "...oxa,kha -> ...oxkh",
            features.xpos[..., None, :, :].astype(dtype),
            freqs,
            precision=precision,
        )
        yphi = jnp.einsum(
            "...oya,kha -> ...oykh",
            features.ypos[..., None, :, :].astype(dtype),
            freqs,
            precision=precision,
        )
        phi = [yphi, xphi]

        # first; linear projection into QKV for each of the features separtely
        cell_mix = self.cell_global_mix(features.context.inv)[..., None, None, :, :]
        mixmap = dict(
            cells=jnp.tile(cell_mix, (Y, X, 1, 1)),
        )
        if self.hdrs_attend:
            hdr_mix = self.hdr_global_mix(features.context.inv)[..., None, :, :]
            mixmap.update(
                rows=jnp.tile(hdr_mix, (Y, 1, 1)),
                cols=jnp.tile(hdr_mix, (X, 1, 1)),
            )
        qkv = {}
        qkvi = {}
        for k, v in self.qkv.items():
            inp = getattr(features, k)
            # print(f"{k}: {inp.shapes} {v.in_features=} {v.out_features=}")
            mix = mixmap.get(k)
            if mix is not None:
                inp = attrs.evolve(inp, inv=jnp.concatenate([mix, inp.inv], axis=-1))
            out = v(inp)
            rep = out.rep
            equiv = out.equiv
            inv = out.inv

            di = {}
            d = {}
            for kk, n in dict(Q=N * H * 2, K=K * H * 2, V=K * D).items():
                if kk in "QK":
                    maybe_cast = lambda v: v.astype(attention_dtype)
                else:
                    maybe_cast = lambda v: v
                d[kk] = maybe_cast(equiv[..., :n])
                # print(f"equiv {k}.{kk}.shape = {d[kk].shape}")
                equiv = equiv[..., n:]
                if not self.hdrs_attend and k in {"rows", "cols"}:
                    continue
                di[kk] = maybe_cast(inv[..., :n])
                # print(f"inv {k}.{kk}.shape = {di[kk].shape}")
                inv = inv[..., n:]
            assert not inv.size, f"{k}: {inv.shape=}"
            assert not equiv.size
            qkvi[k] = SimpleNamespace(**di)
            qkv[k] = SimpleNamespace(**d, rep=rep)
        qkv = SimpleNamespace(**qkv)
        qkvi = SimpleNamespace(**qkvi)

        # second; axial attention
        gres = []
        ares = []
        orep = []
        for axis in range(2):
            # careful: performing attention along axis 0, column headers are global, row index acts as position
            # so in this case X is a batch dimension, and Y acts as source/target
            # context shape: ... F R hd
            # hdr shape: ... Y/X F R hd
            # axial shape .... Y X F R hd
            hdr = [qkv.cols, qkv.rows][axis]
            hmsk, ohmsk = [features.cmsk, features.rmsk][:: (1, -1)[axis]]
            oS = hdr.Q.shape[-4]
            Pi = np.array([qkv.cells.rep.op2idx[o] for o in hdr.rep.opseq])
            P = Pi.size
            orep.extend(hdr.rep.opseq)
            polarisation = np.array(
                [
                    transform_vector(o, Vector.DOWN.as_array())[axis]
                    for o in hdr.rep.opseq
                ]
            )
            assert np.all(abs(polarisation) == 1)
            polarisation = (polarisation + 1) // 2
            # first, reshape stuff into "... tB S/T tF P hd" style; this way we have fixed axis positions
            tB, tF = [(1, oS * F), (oS, F)][axis]
            gK = qkv.context.K[..., Pi, :].reshape(*batch, 1, 1, F, P, K * H * 2)
            gV = qkv.context.V[..., Pi, :].reshape(*batch, 1, 1, F, P, K * D)
            gK, gV = [
                jnp.tile(
                    v,
                    [
                        (1, 1, oS, 1, 1),
                        (oS, 1, 1, 1, 1),
                    ][axis],
                )
                for v in (gK, gV)
            ]
            hQ = hdr.Q.reshape(*batch, tB, 1, tF, P, N * H * 2)
            hK = hdr.K.reshape(*batch, tB, 1, tF, P, K * H * 2)
            hV = hdr.V.reshape(*batch, tB, 1, tF, P, K * D)
            # now, we can concatenate along axis 2
            ghK = jnp.concatenate([gK, hK], axis=-4)
            ghV = jnp.concatenate([gV, hV], axis=-4)

            def make_qkv(q, k, v, *, mask, S, T):
                # unravel hd -> (N H 2) / (K H 2) / (K D)
                return QKV(
                    query=q.reshape(*batch, tB, T, tF, P, N, H, 2),  # noqa: B023
                    key=k.reshape(*batch, tB, S, tF, P, K, H, 2),  # noqa: B023
                    value=v.reshape(*batch, tB, S, tF, P, K, D),  # noqa: B023
                    mask=mask,
                )

            S = T = ohmsk.shape[-1]
            res = attention_RoPE_with_global(
                context=make_qkv(
                    hQ,
                    ghK,
                    ghV,
                    mask=None,  # np.ones(2,bool),
                    T=1,
                    S=2,
                ),
                axial=make_qkv(
                    **{
                        k.lower(): v[..., Pi, :]
                        for k, v in vars(qkv.cells).items()
                        if k != "rep"
                    },
                    T=T,
                    S=S,
                    mask=jnp.tile(features.mask[..., None], (F,)).reshape(
                        *batch, tB, S, tF
                    ),
                ),
                pQ=phi[axis],
                polarisation=polarisation,
                precision=precision,
            )
            ohdr, oax = (v.reshape(*v.shape[:-2], N * D) for v in res)
            # ohdr now has dimensions tB 1 tF P C
            assert ohdr.shape[-4] == 1
            # oax now has dimensions tB S tF P C

            # TODO: global attention to axis headers?
            gres.append(ohdr[..., :, 0, :, :, :].reshape(*batch, oS, F, P, N * D))
            ares.append(oax.reshape(*batch, Y, X, F, P, N * D))

        cells = jnp.concatenate(ares, axis=-2)
        orep = SymRep.from_seq(orep)

        efc = {}
        for k in "QKV":
            g = getattr(qkvi.context, k)[..., None, :, :]  # ... F C
            c = getattr(qkvi.cells, k).reshape(*batch, Y * X, F, -1)  # ... Y X F C
            # print(f"g: {show_dims("sfc", g)}")
            # print(f"c: {show_dims("sfc", c)}")
            v = jnp.concatenate([g, c], axis=-3)
            v = v.reshape(
                *batch, Y * X + 1, F, *dict(Q=(N, 2 * H), K=(K, 2 * H), V=(K, D))[k]
            )
            # unfortunately, `jax.nn.dot_product_attention` doesn't do mixed precision
            efc[k] = v.astype(attention_dtype)
        efc = SimpleNamespace(**efc)

        # third; pointwise self-attention across flavours
        pwatt = jax.nn.dot_product_attention(
            query=efc.Q.reshape(B * (Y * X + 1), F, N, 2 * H),
            key=efc.K.reshape(B * (Y * X + 1), F, K, 2 * H),
            value=efc.V.reshape(B * (Y * X + 1), F, K, D),
            # mask = features.mask[...,None],
        ).astype(dtype)
        pwatt = pwatt.reshape(*batch, Y * X + 1, F, N * D)
        context_self = pwatt[..., 0, :, :]
        cells_inv = pwatt[..., 1:, :, :].reshape(*batch, Y, X, F, -1)

        if self.hdrs_attend:
            raise NotImplementedError(
                "We'd need to implement cross-flavour attention for headers here"
            )

        # fourth; global attention
        glatt = jax.nn.dot_product_attention(
            query=jnp.swapaxes(efc.Q[:, :1, :, :, :], -3, -4).reshape(
                B * F, 1, N, 2 * H
            ),
            key=jnp.swapaxes(efc.K, -3, -4).reshape(B * F, Y * X + 1, K, 2 * H),
            value=jnp.swapaxes(efc.V, -3, -4).reshape(B * F, Y * X + 1, K, D),
            mask=jnp.concatenate(
                [
                    # TODO: should we self-attend here?
                    jnp.zeros((B * F, 1, 1, 1), bool),
                    jnp.tile(features.mask, (F, 1)).reshape(-1, 1, 1, Y * X),
                ],
                axis=-1,
            ),
        ).astype(dtype)
        assert glatt.shape[-3] == 1
        glatt = glatt.reshape(*batch, F, N * D)
        context2cellinv = glatt

        # fifth; global dihedral attention
        # attention to cells
        assert qkv.context.rep == qkv.cells.rep
        context2cell = jax.nn.dot_product_attention(
            # merge F & R directly into batch dimensions left of it
            query=qkv.context.Q.reshape(-1, 1, N, 2 * H),
            # we first need to move F&R across X and Y before we can merge
            key=jnp.moveaxis(qkv.cells.K, (-3, -2), (-5, -4)).reshape(
                -1, Y * X, K, 2 * H
            ),
            # we first need to move F&R across X and Y before we can merge
            value=jnp.moveaxis(qkv.cells.V, (-3, -2), (-5, -4))
            .reshape(-1, Y * X, K, D)
            .astype(attention_dtype),
            mask=jnp.tile(features.mask, (F * R, 1)).reshape(-1, 1, 1, Y * X),
        ).astype(dtype)
        assert context2cell.shape[-3] == 1
        context2cell = context2cell.reshape(*batch, F, R, N * D)

        # print(f"{context2cellinv.shape=}")
        # print(f"{cells_inv.shape=}")
        # print(f"{context_self.shape=}")
        tmp = dict(
            context=attrs.evolve(
                features.context,
                inv=jnp.concatenate([context_self, context2cellinv], -1),
                equiv=context2cell,
                rep=qkv.context.rep,
            ),
            cols=attrs.evolve(
                features.cols,
                inv=jnp.empty(batch + (X, F, 0), dtype),
                equiv=gres[0],
                rep=qkv.cols.rep,
            ),
            rows=attrs.evolve(
                features.rows,
                inv=jnp.empty(batch + (Y, F, 0), dtype),
                equiv=gres[1],
                rep=qkv.rows.rep,
            ),
            cells=attrs.evolve(features.cells, inv=cells_inv, equiv=cells, rep=orep),
        )

        if False:
            for k, v in tmp.items():
                print(f"{k}: {v.inv.shape=} {v.equiv.shape=}")

        # finally; output projection
        output = attrs.evolve(features, **{k: self.out[k](v) for k, v in tmp.items()})
        assert self.out_features.validate(
            output
        ), self.out_features.validation_problems(output)
        return output
