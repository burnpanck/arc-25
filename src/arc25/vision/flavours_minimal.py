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

from ..dsl.types import Dir4, Vector
from ..symmetry import transform_vector
from .linear import SymmetricLinear
from .rope import QKV, attention_RoPE_with_global
from .symrep import Embedding, EmbeddingDims, SymRep


@attrs.frozen
class Features:
    globl: Embedding  # dimensions (... R? C); full representation
    rows: Embedding  # dimensions (... Y R? C); representation (t,l,r,d)
    cols: Embedding  # dimensions (... X R? C); representation (e,x,y,i)
    cells: Embedding  # dimensions (... Y X R? C); full representation
    fcells: jt.Float[jt.Array, "... Y X F C"]
    flavours: jt.Float[jt.Array, "... F C"]
    ypos: jt.Float[jt.Array, "... Y 2"]  # (absolute positions, relative positions)
    xpos: jt.Float[jt.Array, "... X 2"]  # (absolute positions, relative positions)
    rmsk: jt.Bool[jt.Array, "... Y"]
    cmsk: jt.Bool[jt.Array, "... X"]
    mask: jt.Bool[jt.Array, "... Y X"]

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
    fcells: int
    flavours: int
    # these are in fact optional, as we don't need them for any weight calculation
    n_flavours: int | None = None
    shape: tuple[int, int] | None = None

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
        if self.n_flavours is None:
            F = f.flavours.shape[-2]
        else:
            F = self.n_flavours
        if self.shape is None:
            Y, X = f.cells.full.shape[-4:-2]
        else:
            Y, X = self.shape
        if f.rows.full.shape[-3] != Y:
            return f"rows [{Y},{X}] <> {f.rows.shapes}"
        if f.cols.full.shape[-3] != X:
            return f"cols [{Y},{X}] <> {f.cols.shapes}"
        if f.cells.full.shape[-4:-2] != (Y, X):
            return f"cols [{Y},{X}] <> {f.cells.shapes}"
        if f.fcells.shape[-4:] != (Y, X, F, self.fcells):
            return f"fcells [{Y},{X},{F},{self.fcells}] <> {f.fcells.shape}"
        if f.flavours.shape[-2:] != (F, self.flavours):
            return f"flavours [{F},{self.flavours}] <> {f.flavours.shape}"
        if f.ypos.shape[-2:] != (Y, 2):
            return f"ypos [{Y},{X}] <> {f.ypos.shape}"
        if f.xpos.shape[-2:] != (X, 2):
            return f"xpos [{Y},{X}] <> {f.xpos.shape}"
        if f.rmsk.shape[-1] != Y:
            return f"rmsk [{Y},{X}] <> {f.rmsk.shape}"
        if f.cmsk.shape[-1] != X:
            return f"cmsk [{Y},{X}] <> {f.cmsk.shape}"
        if f.mask.shape[-2:] != (Y, X):
            return f"mask [{Y},{X}] <> {f.mask.shape}"
        try:
            np.broadcast_shapes(
                f.globl.full.shape[:-2],
                f.rows.full.shape[:-3],
                f.cols.full.shape[:-3],
                f.cells.full.shape[:-4],
                f.fcells.shape[:-4],
                f.flavours.shape[:-2],
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
        n_flavours: int | None = None,
    ) -> Features:
        if shape is None:
            shape = self.shape
            assert shape is not None
        else:
            assert self.shape is None or shape == self.shape
        if n_flavours is None:
            n_flavours = self.n_flavours
            assert n_flavours is not None
        else:
            assert self.n_flavours is None or n_flavours == self.n_flavours
        Y, X = shape
        F = n_flavours
        ret = Features(
            globl=self.globl.make_empty(batch),
            rows=self.rows.make_empty(batch + (Y,)),
            cols=self.cols.make_empty(batch + (X,)),
            cells=self.cells.make_empty(batch + shape),
            fcells=np.empty(batch + (Y, X, F, self.fcells)),
            flavours=np.empty(batch + (F, self.flavours)),
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
        self.in_features = in_features
        self.qkv_features = qkv_features
        self.out_features = out_features
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
            # TODO dtypes, biases, etc...
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

        self.qkv = {
            k: make_linear(
                v,
                attrs.evolve(
                    v, iso=nqkv if k not in {"rows", "cols"} else 0, full=nqkv
                ),
            )
            for k, v in {
                k: getattr(in_features, k) for k in ["globl", "rows", "cols", "cells"]
            }.items()
        }
        self.qkv["fcells"] = make_linear(
            in_features.fcells + in_features.flavours,
            nqkv,
            cls=nnx.Linear,
        )
        self.qkv["flavours"] = make_linear(
            in_features.flavours,
            nqkv,
            cls=nnx.Linear,
        )
        nv = num_heads * n_features * 2
        self.out = {
            k: make_linear(
                (
                    attrs.evolve(
                        v.out_features,
                        iso=dict(globl=2 * nv, cells=nv).get(k, 0),
                        full=nv,
                    )
                    if isinstance(v, SymmetricLinear)
                    else dict(flavours=2 * nv, fcells=nv)[k]
                ),
                getattr(out_features, k),
                cls=type(v),
            )
            for k, v in self.qkv.items()
        }

    def __call__(self, features: Features) -> Features:
        assert self.in_features.validate(
            features
        ), self.in_features.validation_problems(features)
        assert features.globl.rep == features.cells.rep

        R = features.cells.rep.dim
        batch = features.cells.full.shape[:-4]
        B = int(np.prod(batch))
        Y, X = features.cells.full.shape[-4:-2]
        F = features.flavours.shape[-2]
        H = self.n_features
        D = 2 * H
        K = self.num_groups
        N = self.num_heads

        print(f"{batch=} {B=} {Y=} {X=} {F=} {R=} {N=} {K=} {H=} {D=}")

        # `o` dimension: broadcast "other" spacial axis
        xphi = jnp.einsum(
            "...oxa,kha -> ...oxkh", features.xpos[..., None, :, :], self.freqs
        )
        yphi = jnp.einsum(
            "...oya,kha -> ...oykh", features.ypos[..., None, :, :], self.freqs
        )
        phi = [yphi, xphi]

        # first; linear projection into QKV for each of the features separtely
        qkv = {}
        qkvi = {}
        for k, v in self.qkv.items():
            inp = getattr(features, k)
            print(
                f"{k}: {inp.shape if not isinstance(inp, Embedding) else inp.shapes} {v.in_features=} {v.out_features=}"
            )
            if k == "fcells":
                inp = jnp.concatenate(
                    [
                        jnp.tile(
                            features.flavours[..., None, None, :, :], (Y, X, 1, 1)
                        ),
                        inp,
                    ],
                    axis=-1,
                )
            out = v(inp)
            di = {}
            if isinstance(v, SymmetricLinear):
                rep = out.rep
                full = out.full
                iso = out.iso
                d = {}
            else:
                iso = out
                full = None
                rep = None
                d = None
            for kk, n in dict(Q=N * H * 2, K=K * H * 2, V=K * D).items():
                if full is not None:
                    d[kk] = dd = full[..., :n]
                    print(f"full {k}.{kk}.shape = {dd.shape}")
                    full = full[..., n:]
                if kk in {"rows", "cols"}:
                    continue
                di[kk] = dd = iso[..., :n]
                print(f"iso {k}.{kk}.shape = {dd.shape}")
                iso = iso[..., n:]
            assert not iso.size
            if full is not None:
                assert not full.size
            qkvi[k] = SimpleNamespace(**di)
            if full is not None:
                qkv[k] = SimpleNamespace(**d, rep=rep)
        qkv = SimpleNamespace(**qkv)
        qkvi = SimpleNamespace(**qkvi)

        # second; axial attention
        gres = []
        ares = []
        orep = []
        for axis in range(2):
            match axis:
                case 0:
                    maybe_swap = lambda a, i, j: jnp.swapaxes(a, i, j)
                case 1:
                    maybe_swap = lambda a, i, j: a
                case _:
                    raise RuntimeError
            # careful: performing attention along axis 0, column headers are global, row index acts as position
            # so in this case X is a batch dimension, and Y acts as source/target
            # globl shape: ... R hd
            # hdr shape: ... oS R hd
            # axial shape
            #  - before `maybe_swap`: .... Y X R hd
            #  - after  `maybe_swap`: .... oS L R hd
            hdr = [qkv.cols, qkv.rows][axis]
            hmsk = [features.cmsk, features.rmsk][axis]
            oS = hdr.Q.shape[-3]
            Pi = np.array([qkv.cells.rep.op2idx[o] for o in hdr.rep.opseq])
            orep.extend(hdr.rep.opseq)
            polarisation = np.array(
                [
                    transform_vector(o, Vector.DOWN.as_array())[axis]
                    for o in hdr.rep.opseq
                ]
            )
            assert np.all(abs(polarisation) == 1)
            polarisation = (polarisation + 1) // 2
            # we concatenate global and axis headers for the KVs
            # but there will only be axis headers in the Qs
            # concatenation is along S/T, output shape is ... oS S/T P hd,
            gQ = hdr.Q[..., :, None, :, :]
            gK = jnp.concatenate(
                [
                    jnp.tile(qkv.globl.K[..., None, None, Pi, :], (oS, 1, 1, 1)),
                    hdr.K[..., :, None, :, :],
                ],
                axis=-3,
            )
            gV = jnp.concatenate(
                [
                    jnp.tile(qkv.globl.V[..., None, None, Pi, :], (oS, 1, 1, 1)),
                    hdr.V[..., :, None, :, :],
                ],
                axis=-3,
            )
            mask = jnp.tile(hmsk[..., :, None], (1, gK.shape[-3]))

            def make_qkv(q, k, v, mask):
                # unravel hd -> (N H 2) / (K H 2) / (K D)
                print(f"v: {v.reshape(*v.shape[:-1], K, D).shape} {v.shape=} {K=} {D=}")
                return QKV(
                    query=q.reshape(*q.shape[:-1], N, H, 2),
                    key=k.reshape(*k.shape[:-1], K, H, 2),
                    value=v.reshape(*v.shape[:-1], K, D),
                    mask=mask,
                )

            res = attention_RoPE_with_global(
                globl=make_qkv(
                    gQ,
                    gK,
                    gV,
                    mask=mask,
                ),
                axial=make_qkv(
                    **{
                        k.lower(): maybe_swap(v[..., :, :, Pi, :], -4, -3)
                        for k, v in vars(qkv.cells).items()
                        if k != "rep"
                    },
                    mask=maybe_swap(features.mask, -2, -1),
                ),
                pQ=phi[axis],
                polarisation=polarisation,
            )
            ohdr, oax = (v.reshape(*v.shape[:-2], N * D) for v in res)
            # ohdr now has dimensions ... B 1 P F
            # oax now has dimensions ... B S P F

            # TODO: global attention to axis headers

            assert ohdr.shape[-3] == 1
            gres.append(ohdr[..., :, 0, :, :])
            ares.append(maybe_swap(oax, -4, -3))
        cells = jnp.concatenate(ares, axis=-2)
        orep = SymRep.from_seq(orep)

        efc = {}
        for k in "QKV":
            g = getattr(qkvi.globl, k)[..., None, None, :]  # ... C
            f = getattr(qkvi.flavours, k)[..., None, :, :]  # ... F C
            c = getattr(qkvi.cells, k).reshape(*batch, Y * X, 1, -1)  # ... Y X C
            fc = getattr(qkvi.fcells, k).reshape(*batch, Y * X, F, -1)  # ... Y X F C
            v = jnp.concatenate(
                [jnp.concatenate(p, axis=-2) for p in [(g, f), (c, fc)]], axis=-3
            )
            v = v.reshape(
                *batch, Y * X + 1, F + 1, *dict(Q=(N, 2 * H), K=(K, 2 * H), V=(K, D))[k]
            )
            efc[k] = v
        efc = SimpleNamespace(**efc)

        # third; pointwise flavour attention
        pwatt = jax.nn.dot_product_attention(
            query=efc.Q.reshape(B * (Y * X + 1), F + 1, N, 2 * H),
            key=efc.K.reshape(B * (Y * X + 1), F + 1, K, 2 * H),
            value=efc.V.reshape(B * (Y * X + 1), F + 1, K, D),
            # mask = features.mask[...,None],
        )
        pwatt = pwatt.reshape(*batch, Y * X + 1, F + 1, N * D)
        globl2flavour = pwatt[..., 0, 0, :]
        flavours_self = pwatt[..., 0, 1:, :]
        cells_iso = pwatt[..., 1:, 0, :].reshape(*batch, Y, X, -1)
        fcells = pwatt[..., 1:, 1:, :].reshape(*batch, Y, X, F, -1)

        # fourth; global iso attention
        glatt = jax.nn.dot_product_attention(
            query=jnp.swapaxes(efc.Q[:, :1, :, :, :], -3, -4).reshape(
                B * (F + 1), 1, N, 2 * H
            ),
            key=jnp.swapaxes(efc.K, -3, -4).reshape(B * (F + 1), Y * X + 1, K, 2 * H),
            value=jnp.swapaxes(efc.V, -3, -4).reshape(B * (F + 1), Y * X + 1, K, D),
            mask=jnp.concatenate(
                [
                    # TODO: should we self-attend here?
                    jnp.zeros((B * (F + 1), 1, 1, 1), bool),
                    jnp.tile(features.mask, (F + 1, 1)).reshape(-1, 1, 1, Y * X),
                ],
                axis=-1,
            ),
        )
        assert glatt.shape[-3] == 1
        glatt = glatt.reshape(*batch, F + 1, N * D)
        globl2celliso = glatt[..., 0, :]
        flavours = glatt[..., 1:, :]

        # fifth; global dihedral attention
        # attention to cells
        assert qkv.globl.rep == qkv.cells.rep
        globl2cell = jax.nn.dot_product_attention(
            # merge R directly into batch dimensions left of it
            query=qkv.globl.Q.reshape(-1, 1, N, 2 * H),
            # we first need to move R across X and Y before we can merge
            key=jnp.moveaxis(qkv.cells.K, -2, -4).reshape(-1, Y * X, K, 2 * H),
            # we first need to move R across X and Y before we can merge
            value=jnp.moveaxis(qkv.cells.V, -2, -4).reshape(-1, Y * X, K, D),
            mask=jnp.tile(features.mask, (R, 1)).reshape(-1, 1, 1, Y * X),
        )
        assert globl2cell.shape[-3] == 1
        globl2cell = globl2cell.reshape(*batch, R, N * D)

        print(f"{globl2flavour.shape=} {globl2celliso.shape=}")
        print(f"{cells_iso.shape=}")
        print(f"{flavours_self.shape=} {flavours.shape=}")
        tmp = dict(
            globl=attrs.evolve(
                features.globl,
                iso=jnp.concatenate([globl2flavour, globl2celliso], -1),
                full=globl2cell,
                rep=qkv.globl.rep,
            ),
            cols=attrs.evolve(
                features.cols,
                iso=jnp.empty((X, 0), self.dtype),
                full=gres[0],
                rep=qkv.cols.rep,
            ),
            rows=attrs.evolve(
                features.rows,
                iso=jnp.empty((Y, 0), self.dtype),
                full=gres[1],
                rep=qkv.rows.rep,
            ),
            cells=attrs.evolve(features.cells, iso=cells_iso, full=cells, rep=orep),
            flavours=jnp.concatenate([flavours_self, flavours], -1),
            fcells=fcells,
        )

        # finally; output projection
        output = attrs.evolve(features, **{k: self.out[k](v) for k, v in tmp.items()})
        assert self.out_features.validate(
            output
        ), self.out_features.validation_problems(output)
        return output
