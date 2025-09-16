import typing
from types import MappingProxyType, SimpleNamespace
from typing import Self

import attrs
import jaxtyping as jt
import numpy as np

from ..lib.attrs import AttrsModel
from ..symmetry import SymOp

# we could have other symmetries
AnySymOp: typing.TypeAlias = SymOp


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
    def from_seq(cls, opseq: typing.Iterable[AnySymOp]) -> typing.Self:
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


standard_rep = SymRep.from_seq(SymOp)


class Embedding(AttrsModel):
    iso: jt.Float[jt.Array, "... Ci"]
    full: jt.Float[jt.Array, "... R Cf"]
    rep: SymRep = attrs.field(default=standard_rep, metadata=dict(static=True))

    @property
    def features(self) -> dict[str, jt.Float]:
        return {
            f.name: getattr(self, f.name)
            for f in attrs.fields(type(self))
            if f.type is Embedding
        }

    def map_features(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self
    ) -> Self:
        return attrs.evolve(
            self,
            **{
                k: fun(v, *[getattr(o, k) for o in other])
                for k, v in self.features.items()
            },
        )

    @property
    def shapes(self):
        return SimpleNamespace(
            iso=self.iso.shape,
            full=self.full.shape,
            rep=self.rep.dim,
        )


@attrs.frozen
class EmbeddingDims:
    iso: int  # isotropic values/trivial representation
    full: int  # full-dimensional representation
    rep: SymRep = standard_rep

    @property
    def features(self) -> dict[str, jt.Float]:
        return {k: getattr(self, k) for k in ["iso", "full"]}

    def map_features(
        self,
        fun: typing.Callable[[str, int], typing.Any],
        *other: Self,
        cls: type = SimpleNamespace,
    ) -> typing.Any:
        return cls(
            **{
                k: fun(k, v, *[getattr(o, k) for o in other])
                for k, v in self.features.items()
            }
        )

    @property
    def dims(self):
        return SimpleNamespace(
            iso=self.iso,
            full=self.full,
            rep=self.rep.dim,
        )

    def validation_problems(self, embedding: Embedding) -> bool:
        if not self.rep.is_valid():
            return "representation"
        try:
            np.broadcast_shapes(
                embedding.iso.shape[:-1],
                embedding.full.shape[:-2],
            )
        except ValueError:
            return "batch mismatch"
        if self.rep != embedding.rep:
            return "rep mismatch"
        if self.iso != embedding.iso.shape[-1]:
            return "iso dim mismatch"
        if (self.rep.dim, self.full) != embedding.full.shape[-2:]:
            return f"full dim mismatch {(self.rep.dim, self.full)} <> {embedding.full.shape}"

    def validate(self, embedding: Embedding) -> bool:
        return not self.validation_problems(embedding)

    def make_empty(self, batch: tuple[int, ...] = ()) -> Embedding:
        ret = Embedding(
            iso=np.empty(batch + (self.iso,)),
            full=np.empty(batch + (self.rep.dim, self.full)),
            rep=self.rep,
        )
        assert self.validate(ret)
        return ret
