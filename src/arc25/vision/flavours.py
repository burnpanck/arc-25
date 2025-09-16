import typing
from types import SimpleNamespace
from typing import Self

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.nnx.nn.linear import default_bias_init, default_kernel_init, initializers
from flax.typing import (
    DotGeneralT,
    Dtype,
    Initializer,
    PrecisionLike,
    PromoteDtypeFn,
)
from jax import lax

from ..dsl.types import Vector
from ..lib.attrs import AttrsModel
from ..symmetry import SymOp, transform_vector
from .linear import SymmetricLinear
from .rope import QKV, attention_RoPE_with_global, show_dims
from .symrep import Embedding, EmbeddingDims, SymRep


class Features(AttrsModel):
    globl: Embedding  # dimensions (... F R? C); full representation
    rows: Embedding  # dimensions (... Y F R? C); representation (t,l,r,d)
    cols: Embedding  # dimensions (... X F R? C); representation (e,x,y,i)
    cells: Embedding  # dimensions (... Y X F R? C); full representation
    ypos: jt.Float[jt.Array, "... Y 2"]  # (absolute positions, relative positions)
    xpos: jt.Float[jt.Array, "... X 2"]  # (absolute positions, relative positions)
    rmsk: jt.Bool[jt.Array, "... Y"]
    cmsk: jt.Bool[jt.Array, "... X"]
    mask: jt.Bool[jt.Array, "... Y X"]

    @property
    def embeddings(self) -> dict[str, Embedding]:
        return {
            f.name: getattr(self, f.name)
            for f in attrs.fields(type(self))
            if f.type is Embedding
        }

    def map_embeddings(
        self, fun: typing.Callable[[Embedding], Embedding], *other: Self
    ) -> Self:
        return attrs.evolve(
            self,
            **{
                k: fun(v, *[getattr(o, k) for o in other])
                for k, v in self.embeddings.items()
            },
        )

    def map_features(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self
    ) -> Self:
        return attrs.evolve(
            self,
            **{
                k: v.map_features(fun, *[getattr(o, k) for o in other])
                for k, v in self.embeddings.items()
            },
        )

    @property
    def shapes(self):
        return SimpleNamespace(
            {
                k: v.shapes if isinstance(v, Embedding) else v.shape
                for k, v in attrs.asdict(self, recurse=False).items()
            }
        )


@attrs.frozen
class FeatureDim:
    globl: EmbeddingDims
    rows: EmbeddingDims
    cols: EmbeddingDims
    cells: EmbeddingDims
    # these are in fact optional, as we don't need them for any weight calculation
    flavours: int | None = None
    shape: tuple[int, int] | None = None

    @classmethod
    def make(cls, *, iso=2, globl=None, hdrs=None, cells, **kw) -> Self:
        if globl is None:
            globl = cells
        if hdrs is None:
            hdrs = cells
        return cls(
            globl=EmbeddingDims(iso=iso * globl, full=globl),
            rows=EmbeddingDims(
                iso=iso * hdrs,
                full=hdrs,
                rep=SymRep.from_seq((SymOp.t, SymOp.l, SymOp.r, SymOp.d)),
            ),
            cols=EmbeddingDims(
                iso=iso * hdrs,
                full=hdrs,
                rep=SymRep.from_seq((SymOp.e, SymOp.x, SymOp.y, SymOp.i)),
            ),
            cells=EmbeddingDims(iso=iso * cells, full=cells),
            **kw,
        )

    @property
    def embeddings(self) -> dict[str, EmbeddingDims]:
        return {
            f.name: getattr(self, f.name)
            for f in attrs.fields(type(self))
            if f.type is EmbeddingDims
        }

    def map_embeddings(
        self,
        fun: typing.Callable[[str, Embedding], typing.Any],
        *other: Self,
        cls: type = SimpleNamespace,
    ) -> typing.Any:
        return cls(
            **{
                k: fun(k, v, *[getattr(o, k) for o in other])
                for k, v in self.embeddings.items()
            }
        )

    def map_features(
        self,
        fun: typing.Callable[[str, str, jt.Float], jt.Float],
        *other: Self,
        cls: type = SimpleNamespace,
        ecls: type = SimpleNamespace,
    ) -> typing.Any:
        return cls(
            **{
                k: v.map_features(
                    lambda kk, *vv: fun(k, kk, *vv),  # noqa: B023
                    *[getattr(o, k) for o in other],
                    cls=ecls,
                )
                for k, v in self.embeddings.items()
            }
        )

    def validity_problem(self):
        if not (
            self.globl.rep.is_valid()
            and self.rows.rep.is_valid()
            and self.cols.rep.is_valid()
            and self.cells.rep.is_valid()
        ):
            return "invalid rep"
        if (
            not set(self.globl.rep.opseq)
            == set(self.rows.rep.opseq) | set(self.cols.rep.opseq)
            == set(self.cells.rep.opseq)
        ):
            return "rep mismatch"
        if set(self.rows.rep.opseq) & set(self.cols.rep.opseq):
            return "rep overlap"
        for k in ["iso", "full"]:
            if getattr(self.rows, k) != getattr(self.cols, k):
                return f"row/col mismatch on {k}"

    def is_valid(self):
        return not self.validity_problem()

    def validation_problem(self, f: Features):
        ret = self.validity_problem()
        if ret:
            return ret
        if not self.globl.validate(f.globl):
            return f"globl {self.globl.dims} != {f.globl.shapes}"
        if not self.rows.validate(f.rows):
            return f"rows {self.rows.dims} != {f.rows.shapes}"
        if not self.cols.validate(f.cols):
            return f"cols {self.cols.dims} != {f.cols.shapes}"
        if not self.cells.validate(f.cells):
            return f"cells {self.cells.dims} != {f.cells.shapes}"
        if self.flavours is None:
            F = f.globls.iso.shape[-2]
        else:
            F = self.flavours
        if self.shape is None:
            Y, X = f.cells.full.shape[-4:-2]
        else:
            Y, X = self.shape
        shi = f"[{Y},{X},{F}]"
        if f.rows.full.shape[-4:-2] != (Y, F):
            return f"rows {shi} <> {f.rows.shapes}"
        if f.cols.full.shape[-4:-2] != (X, F):
            return f"cols {shi} <> {f.cols.shapes}"
        if f.cells.full.shape[-5:-2] != (Y, X, F):
            return f"cols {shi} <> {f.cells.shapes}"
        if f.ypos.shape[-2:] != (Y, 2):
            return f"ypos {shi} <> {f.ypos.shape}"
        if f.xpos.shape[-2:] != (X, 2):
            return f"xpos {shi} <> {f.xpos.shape}"
        if f.rmsk.shape[-1] != Y:
            return f"rmsk {shi} <> {f.rmsk.shape}"
        if f.cmsk.shape[-1] != X:
            return f"cmsk {shi} <> {f.cmsk.shape}"
        if f.mask.shape[-2:] != (Y, X):
            return f"mask {shi} <> {f.mask.shape}"
        try:
            np.broadcast_shapes(
                f.globl.iso.shape[:-2],
                f.rows.iso.shape[:-3],
                f.cols.iso.shape[:-3],
                f.cells.iso.shape[:-4],
                f.globl.full.shape[:-3],
                f.rows.full.shape[:-4],
                f.cols.full.shape[:-4],
                f.cells.full.shape[:-5],
                f.ypos.shape[:-2],
                f.xpos.shape[:-2],
                f.rmsk.shape[:-1],
                f.cmsk.shape[:-1],
                f.mask.shape[:-2],
            )

        except ValueError:
            return f"batch {f.shapes}"

    def validate(self, f: Features):
        return not self.validation_problem(f)

    def make_empty(
        self,
        batch: tuple[int, ...] = (),
        *,
        shape: tuple[int, int] | None = None,
        flavours: int | None = None,
    ) -> Features:
        if shape is None:
            shape = self.shape
            assert shape is not None
        else:
            assert self.shape is None or shape == self.shape
        if flavours is None:
            flavours = self.flavours
            assert flavours is not None
        else:
            assert self.n_flavours is None or flavours == self.flavours
        Y, X = shape
        F = flavours
        ret = Features(
            globl=self.globl.make_empty(batch + (F,)),
            rows=self.rows.make_empty(batch + (Y, F)),
            cols=self.cols.make_empty(batch + (X, F)),
            cells=self.cells.make_empty(batch + shape + (F,)),
            ypos=np.empty(batch + (Y, 2)),
            xpos=np.empty(batch + (X, 2)),
            rmsk=np.empty(batch + (Y,), bool),
            cmsk=np.empty(batch + (X,), bool),
            mask=np.empty(batch + (Y, X), bool),
        )
        assert self.validate(ret), self.validation_problem(ret)
        return ret


class SymAttention(nnx.Module):
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
        in_features: FeatureDim,
        qkv_features: int,
        out_features: FeatureDim | None = None,
        *,
        global_mix_reduction: int = 4,
        num_groups: int | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        # out_kernel_init: Initializer | None = None,
        bias_init: Initializer = default_bias_init,
        # out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        hdrs_attend: bool = False,
        # attention_fn: Callable[..., Array] = dot_product_attention,
        normalize_qk: bool = False,
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
        self.param_dtype = param_dtype

        # frequency is both per group, per features, and linear in both absolute and relative
        kernel_key = rngs.params()
        freq_init = initializers.normal(1)
        self.freqs = nnx.Param(
            freq_init(kernel_key, (num_groups, n_features, 2), param_dtype)
        )

        def make_linear(inf, outf, *, cls=SymmetricLinear):
            # TODO: do we have them all?
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
            in_features.globl.iso,
            in_features.cells.iso // global_mix_reduction,
            cls=nnx.Linear,
        )
        mixmap = dict(
            cells=self.cell_global_mix.out_features,
        )

        if hdrs_attend:
            (hdr_iso,) = {in_features.rows.iso, in_features.cols.iso}
            hdr_mix = hdr_iso // global_mix_reduction
            self.hdr_global_mix = make_linear(
                in_features.globl.iso,
                hdr_mix,
                cls=nnx.Linear,
            )
            mixmap.update(rows=hdr_mix, cols=hdr_mix)
            skip_iso = []
        else:
            self.hdr_global_mix = nnx.data(None)
            skip_iso = ["rows", "cols"]
        self.qkv = {
            k: make_linear(
                attrs.evolve(v, iso=v.iso + mixmap.get(k, 0)),
                attrs.evolve(v, iso=nqkv if k not in skip_iso else 0, full=nqkv),
            )
            for k, v in {
                k: getattr(in_features, k) for k in ["globl", "rows", "cols", "cells"]
            }.items()
        }
        nv = num_heads * n_features * 2
        self.out = {
            k: make_linear(
                attrs.evolve(
                    v.out_features,
                    iso=dict(globl=2 * nv, cells=nv).get(k, nv if hdrs_attend else 0),
                    full=nv,
                ),
                getattr(out_features, k),
            )
            for k, v in self.qkv.items()
        }

    def __call__(self, features: Features) -> Features:
        assert self.in_features.validate(
            features
        ), self.in_features.validation_problems(features)
        assert features.globl.rep == features.cells.rep

        R = features.cells.rep.dim
        batch = features.cells.full.shape[:-5]
        B = int(np.prod(batch))
        Y, X, F = features.cells.full.shape[-5:-2]
        H = self.n_features
        D = 2 * H
        K = self.num_groups
        N = self.num_heads

        # print(f"{batch=} {B=} {Y=} {X=} {F=} {R=} {N=} {K=} {H=} {D=}")

        # `o` dimension: singular dimension to add as broadcast for "other" spatial axis
        xphi = jnp.einsum(
            "...oxa,kha -> ...oxkh", features.xpos[..., None, :, :], self.freqs
        )
        yphi = jnp.einsum(
            "...oya,kha -> ...oykh", features.ypos[..., None, :, :], self.freqs
        )
        phi = [yphi, xphi]

        # first; linear projection into QKV for each of the features separtely
        cell_mix = self.cell_global_mix(features.globl.iso)[..., None, None, :, :]
        mixmap = dict(
            cells=jnp.tile(cell_mix, (Y, X, 1, 1)),
        )
        if self.hdrs_attend:
            hdr_mix = self.hdr_global_mix(features.globl.iso)[..., None, :, :]
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
                inp = attrs.evolve(inp, iso=jnp.concatenate([mix, inp.iso], axis=-1))
            out = v(inp)
            rep = out.rep
            full = out.full
            iso = out.iso

            di = {}
            d = {}
            for kk, n in dict(Q=N * H * 2, K=K * H * 2, V=K * D).items():
                d[kk] = full[..., :n]
                # print(f"full {k}.{kk}.shape = {d[kk].shape}")
                full = full[..., n:]
                if not self.hdrs_attend and k in {"rows", "cols"}:
                    continue
                di[kk] = iso[..., :n]
                # print(f"iso {k}.{kk}.shape = {di[kk].shape}")
                iso = iso[..., n:]
            assert not iso.size, f"{k}: {iso.shape=}"
            assert not full.size
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
            # globl shape: ... F R hd
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
            gK = qkv.globl.K[..., Pi, :].reshape(*batch, 1, 1, F, P, K * H * 2)
            gV = qkv.globl.V[..., Pi, :].reshape(*batch, 1, 1, F, P, K * D)
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
                globl=make_qkv(
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
            g = getattr(qkvi.globl, k)[..., None, :, :]  # ... F C
            c = getattr(qkvi.cells, k).reshape(*batch, Y * X, F, -1)  # ... Y X F C
            # print(f"g: {show_dims("sfc", g)}")
            # print(f"c: {show_dims("sfc", c)}")
            v = jnp.concatenate([g, c], axis=-3)
            v = v.reshape(
                *batch, Y * X + 1, F, *dict(Q=(N, 2 * H), K=(K, 2 * H), V=(K, D))[k]
            )
            efc[k] = v
        efc = SimpleNamespace(**efc)

        # third; pointwise self-attention across flavours
        pwatt = jax.nn.dot_product_attention(
            query=efc.Q.reshape(B * (Y * X + 1), F, N, 2 * H),
            key=efc.K.reshape(B * (Y * X + 1), F, K, 2 * H),
            value=efc.V.reshape(B * (Y * X + 1), F, K, D),
            # mask = features.mask[...,None],
        )
        pwatt = pwatt.reshape(*batch, Y * X + 1, F, N * D)
        globl_self = pwatt[..., 0, :, :]
        cells_iso = pwatt[..., 1:, :, :].reshape(*batch, Y, X, F, -1)

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
        )
        assert glatt.shape[-3] == 1
        glatt = glatt.reshape(*batch, F, N * D)
        globl2celliso = glatt

        # fifth; global dihedral attention
        # attention to cells
        assert qkv.globl.rep == qkv.cells.rep
        globl2cell = jax.nn.dot_product_attention(
            # merge F & R directly into batch dimensions left of it
            query=qkv.globl.Q.reshape(-1, 1, N, 2 * H),
            # we first need to move F&R across X and Y before we can merge
            key=jnp.moveaxis(qkv.cells.K, (-3, -2), (-5, -4)).reshape(
                -1, Y * X, K, 2 * H
            ),
            # we first need to move F&R across X and Y before we can merge
            value=jnp.moveaxis(qkv.cells.V, (-3, -2), (-5, -4)).reshape(
                -1, Y * X, K, D
            ),
            mask=jnp.tile(features.mask, (F * R, 1)).reshape(-1, 1, 1, Y * X),
        )
        assert globl2cell.shape[-3] == 1
        globl2cell = globl2cell.reshape(*batch, F, R, N * D)

        # print(f"{globl2celliso.shape=}")
        # print(f"{cells_iso.shape=}")
        # print(f"{globl_self.shape=}")
        tmp = dict(
            globl=attrs.evolve(
                features.globl,
                iso=jnp.concatenate([globl_self, globl2celliso], -1),
                full=globl2cell,
                rep=qkv.globl.rep,
            ),
            cols=attrs.evolve(
                features.cols,
                iso=jnp.empty(batch + (X, F, 0), self.dtype),
                full=gres[0],
                rep=qkv.cols.rep,
            ),
            rows=attrs.evolve(
                features.rows,
                iso=jnp.empty(batch + (Y, F, 0), self.dtype),
                full=gres[1],
                rep=qkv.rows.rep,
            ),
            cells=attrs.evolve(features.cells, iso=cells_iso, full=cells, rep=orep),
        )

        if False:
            for k, v in tmp.items():
                print(f"{k}: {v.iso.shape=} {v.full.shape=}")

        # finally; output projection
        output = attrs.evolve(features, **{k: self.out[k](v) for k, v in tmp.items()})
        assert self.out_features.validate(
            output
        ), self.out_features.validation_problems(output)
        return output
