import typing
from types import MappingProxyType, SimpleNamespace

import attrs
import jaxtyping as jt
import numpy as np

from ..lib.attrs import AttrsModel
from ..lib.compat import Self
from ..symmetry import D4

# we could have other symmetries
AnySymOp: typing.TypeAlias = D4


@attrs.frozen
class SymRep:
    # these are not actually operations of the symmetry!
    # these are just labels, and symmetry operations connect these labels
    opseq: tuple[AnySymOp, ...] = attrs.field(
        repr=lambda seq: f"({','.join(o.name for o in seq)})",
    )
    op2idx: typing.Mapping[AnySymOp, int] = attrs.field(
        default=attrs.Factory(
            lambda self: MappingProxyType({v: k for k, v in enumerate(self.opseq)}),
            takes_self=True,
        ),
        repr=False,
    )

    @classmethod
    def from_seq(cls, opseq: typing.Iterable[AnySymOp]) -> Self:
        ret = cls(tuple(opseq))
        assert ret.is_valid()
        return ret

    def is_valid(self):
        # ensure inverse map is correct
        if self.op2idx != {v: k for k, v in enumerate(self.opseq)}:
            return False
        # ensure group is closed
        operations = set(self.opseq[0].inverse.combine(o) for o in self.opseq)
        completion = set(o.inverse for o in operations) | set(
            a.combine(b) for a in operations for b in operations
        )
        return completion == operations

    @property
    def dim(self) -> int:
        return len(self.opseq)


standard_rep = SymRep.from_seq(D4)


class SymDecomp(AttrsModel):
    inv: jt.Float[jt.Array, "... Ci"]
    equiv: jt.Float[jt.Array, "... R Cf"]
    rep: SymRep = attrs.field(default=standard_rep, metadata=dict(static=True))

    @property
    def representations(self) -> dict[str, jt.Float]:
        return {k: getattr(self, k) for k in ["inv", "equiv"]}

    def map_representations(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self, **kw
    ) -> Self:
        return attrs.evolve(
            self,
            **{
                k: fun(v, *[getattr(o, k) for o in other], **kw)
                for k, v in self.representations.items()
            },
        )

    @property
    def shapes(self):
        return SimpleNamespace(
            inv=self.inv.shape,
            equiv=self.equiv.shape,
            rep=self.rep.dim,
        )


@attrs.frozen
class SymDecompDims:
    inv: int  # invariant under symmetry operations
    equiv: int  # equivariant under symmetry operations
    rep: SymRep = standard_rep

    @property
    def representations(self) -> dict[str, jt.Float]:
        return {k: getattr(self, k) for k in ["inv", "equiv"]}

    def map_representations(
        self,
        fun: typing.Callable[[str, int], typing.Any],
        *other: Self,
        cls: type = SimpleNamespace,
    ) -> typing.Any:
        return cls(
            **{
                k: fun(k, v, *[getattr(o, k) for o in other])
                for k, v in self.representations.items()
            }
        )

    @property
    def dims(self):
        return SimpleNamespace(
            inv=self.inv,
            equiv=self.equiv,
            rep=self.rep.dim,
        )

    def validation_problems(self, embedding: SymDecomp) -> bool:
        if not self.rep.is_valid():
            return "representation"
        try:
            np.broadcast_shapes(
                embedding.inv.shape[:-1],
                embedding.equiv.shape[:-2],
            )
        except ValueError:
            return "batch mismatch"
        if self.rep != embedding.rep:
            return "rep mismatch"
        if self.inv != embedding.inv.shape[-1]:
            return "inv dim mismatch"
        if (self.rep.dim, self.equiv) != embedding.equiv.shape[-2:]:
            return f"equiv dim mismatch {(self.rep.dim, self.equiv)} <> {embedding.equiv.shape}"

    def validate(self, embedding: SymDecomp) -> bool:
        return not self.validation_problems(embedding)

    def make_empty(self, batch: tuple[int, ...] = ()) -> SymDecomp:
        ret = SymDecomp(
            inv=np.empty(batch + (self.inv,)),
            equiv=np.empty(batch + (self.rep.dim, self.equiv)),
            rep=self.rep,
        )
        assert self.validate(ret)
        return ret
