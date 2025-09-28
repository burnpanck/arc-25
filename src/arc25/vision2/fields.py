import typing
from types import SimpleNamespace

import attrs
import jaxtyping as jt
import numpy as np

from ..lib.attrs import AttrsModel
from ..lib.compat import Self
from ..symmetry import D4
from .symrep import SymDecompBase, SymDecompDims, SymRep, standard_rep


# TODO: latest flax has `nnx.PyTree` that seems to work with `dataclass`
class Field(AttrsModel):
    context: SymDecompBase  # dimensions (... T R? C); full representation
    # rows: SymDecompBase  # dimensions (... Y R? C); representation (t,l,r,d)
    # cols: SymDecompBase  # dimensions (... X R? C); representation (e,x,y,i)
    cells: SymDecompBase  # dimensions (... Y X R? C); full representation
    ypos: jt.Float[jt.Array, "... Y 2"]  # (absolute positions, relative positions)
    xpos: jt.Float[jt.Array, "... X 2"]  # (absolute positions, relative positions)
    # rmsk: jt.Bool[jt.Array, "... Y"]
    # cmsk: jt.Bool[jt.Array, "... X"]
    mask: jt.Bool[jt.Array, "... Y X"]

    @property
    def projections(self) -> dict[str, SymDecompBase]:
        return {
            f.name: getattr(self, f.name)
            for f in attrs.fields(type(self))
            if f.type is SymDecompBase
        }

    def map_projections(
        self, fun: typing.Callable[[SymDecompBase], SymDecompBase], *other: Self, **kw
    ) -> Self:
        return attrs.evolve(
            self,
            **{
                k: fun(v, *[getattr(o, k) for o in other], **kw)
                for k, v in self.projections.items()
            },
        )

    def map_representations(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self, **kw
    ) -> Self:
        return attrs.evolve(
            self,
            **{
                k: v.map_representations(fun, *[getattr(o, k) for o in other], **kw)
                for k, v in self.projections.items()
            },
        )

    @property
    def shapes(self):
        return SimpleNamespace(
            {
                k: v.shapes if isinstance(v, SymDecompBase) else v.shape
                for k, v in attrs.asdict(self, recurse=False).items()
            }
        )

    @property
    def batch_shape(self):
        np.broadcast_shapes(
            self.context.batch_shape[:-1],
            #                self.rows.batch_shape[:-1],
            #                self.cols.batch_shape[:-1],
            self.cells.batch_shape[:-2],
            self.ypos.shape[:-2],
            self.xpos.shape[:-2],
            #                self.rmsk.shape[:-1],
            #                self.cmsk.shape[:-1],
            self.mask.shape[:-2],
        )


@attrs.frozen
class FieldDims:
    context: SymDecompDims
    # rows: SymDecompDims
    # cols: SymDecompDims
    cells: SymDecompDims
    # these are in fact optional, as we don't need them for any weight calculation
    context_tokens: int | None = None
    shape: tuple[int, int] | None = None

    @classmethod
    def make(
        cls,
        cells,
        *,
        invariant=22,
        flavours=1,
        space=2,
        context=None,
        rep=standard_rep,
        **kw,
    ) -> Self:
        if context is None:
            context = cells
        return cls(
            context=SymDecompDims(
                invariant=invariant * context,
                flavour=flavours * context,
                space=space * context,
                rep=rep,
            ),
            cells=SymDecompDims(
                invariant=invariant * cells,
                flavour=flavours * cells,
                space=space * cells,
                rep=rep,
            ),
            **kw,
        )

    @property
    def projections(self) -> dict[str, SymDecompDims]:
        return {
            f.name: getattr(self, f.name)
            for f in attrs.fields(type(self))
            if f.type is SymDecompDims
        }

    def map_projections(
        self,
        fun: typing.Callable[[str, SymDecompBase], typing.Any],
        *other: Self,
        cls: type = SimpleNamespace,
    ) -> typing.Any:
        return cls(
            **{
                k: fun(k, v, *[getattr(o, k) for o in other])
                for k, v in self.projections.items()
            }
        )

    def map_representations(
        self,
        fun: typing.Callable[[str, str, jt.Float], jt.Float],
        *other: Self,
        cls: type = SimpleNamespace,
        field_cls: type | None = None,
        proj_cls: type | None = None,
    ) -> typing.Any:
        field_cls = field_cls or cls
        proj_cls = proj_cls or cls
        return field_cls(
            **{
                k: v.map_representations(
                    lambda kk, *vv: fun(k, kk, *vv),  # noqa: B023
                    *[getattr(o, k) for o in other],
                    cls=proj_cls,
                )
                for k, v in self.projections.items()
            }
        )

    def validity_problems(self):
        if not (
            self.context.rep.is_valid()
            # and self.rows.rep.is_valid()
            # and self.cols.rep.is_valid()
            and self.cells.rep.is_valid()
        ):
            return "invalid rep"
        if not (
            set(self.context.rep.repseq)
            # == (set(self.rows.rep.repseq) | set(self.cols.rep.repseq))
            == set(self.cells.rep.repseq)
        ):
            return "rep mismatch"
        # if set(self.rows.rep.opseq) & set(self.cols.rep.opseq):
        #    return "rep overlap"
        # for k in ["inv", "equiv"]:
        #     if getattr(self.rows, k) != getattr(self.cols, k):
        #        return f"row/col mismatch on {k}"

    def is_valid(self):
        return not self.validity_problems()

    def validation_problems(self, f: Field):
        ret = self.validity_problems()
        if ret:
            return ret
        if not self.context.validate(f.context):
            return f"context {self.context.dims} != {f.context.shapes}"
        # if not self.rows.validate(f.rows):
        #     return f"rows {self.rows.dims} != {f.rows.shapes}"
        # if not self.cols.validate(f.cols):
        #     return f"cols {self.cols.dims} != {f.cols.shapes}"
        if not self.cells.validate(f.cells):
            return f"cells {self.cells.dims} != {f.cells.shapes}"
        if self.shape is None:
            Y, X = f.cells.batch_shape[-2:]
        else:
            Y, X = self.shape
        shi = f"[{Y},{X},{self.context_tokens}]"
        if f.context.batch_shape[-1:] != (self.context_tokens,):
            return f"context tokens {shi} <> {f.context.shapes}"
        # if f.rows.equiv.shape[-4:-2] != (Y, F):
        #     return f"rows {shi} <> {f.rows.shapes}"
        # if f.cols.equiv.shape[-4:-2] != (X, F):
        #     return f"cols {shi} <> {f.cols.shapes}"
        if f.cells.batch_shape[-2:] != (Y, X):
            return f"cols {shi} <> {f.cells.shapes}"
        if f.ypos.shape[-2:] != (Y, 2):
            return f"ypos {shi} <> {f.ypos.shape}"
        if f.xpos.shape[-2:] != (X, 2):
            return f"xpos {shi} <> {f.xpos.shape}"
        # if f.rmsk.shape[-1] != Y:
        #     return f"rmsk {shi} <> {f.rmsk.shape}"
        # if f.cmsk.shape[-1] != X:
        #     return f"cmsk {shi} <> {f.cmsk.shape}"
        if f.mask.shape[-2:] != (Y, X):
            return f"mask {shi} <> {f.mask.shape}"
        try:
            f.batch_shape
        except ValueError:
            return f"batch {f.shapes}"

    def validate(self, f: Field):
        return not self.validation_problems(f)

    def make_empty(
        self,
        batch: tuple[int, ...] = (),
        *,
        shape: tuple[int, int] | None = None,
    ) -> Field:
        if shape is None:
            shape = self.shape
            assert shape is not None
        else:
            assert self.shape is None or shape == self.shape
        Y, X = shape
        ret = Field(
            context=self.context.make_empty(batch + (self.context_tokens,)),
            #            rows=self.rows.make_empty(batch + (Y, F)),
            #            cols=self.cols.make_empty(batch + (X, F)),
            cells=self.cells.make_empty(batch + shape),
            ypos=np.empty(batch + (Y, 2)),
            xpos=np.empty(batch + (X, 2)),
            rmsk=np.empty(batch + (Y,), bool),
            cmsk=np.empty(batch + (X,), bool),
            mask=np.empty(batch + (Y, X), bool),
        )
        assert self.validate(ret), self.validation_problems(ret)
        return ret
