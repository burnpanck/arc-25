import abc
import typing
from types import MappingProxyType, SimpleNamespace

import attrs
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax.typing import Dtype

from ..lib.attrs import AttrsModel
from ..lib.compat import Self
from ..serialisation import serialisable
from ..symmetry import D4, FullRep, PermRepBase, SymOpBase


@serialisable
@attrs.frozen
class RepSpec:
    space: type[PermRepBase]
    n_flavours: int
    symmetry_group: type[SymOpBase] = D4
    basis2idx: typing.Mapping[PermRepBase, int] = attrs.field(
        default=attrs.Factory(
            lambda self: MappingProxyType({v: k for k, v in enumerate(self.space)}),
            takes_self=True,
        ),
        repr=False,
        metadata=dict(is_cache=True),
    )

    def __attrs_post_init__(self):
        assert self.is_valid()

    def is_valid(self):
        # ensure inverse map is correct
        if self.basis2idx != {v: k for k, v in enumerate(self.space)}:
            return False
        # ensure subgroup is closed
        ref = list(self.space)[0]
        operations = frozenset.union(*[ref.mapping_to(o) for o in self.space])
        completion = set(o.inverse for o in operations) | set(
            a.combine(b) for a in operations for b in operations
        )
        return operations.issubset(self.symmetry_group) and completion == operations

    @property
    def n_space(self) -> int:
        return len(self.space)


standard_rep = RepSpec(FullRep, n_flavours=10)


class SymDecompBase(abc.ABC, AttrsModel):
    @property
    @abc.abstractmethod
    def batch_shape(self) -> tuple[int, ...]: ...

    def batch_reshape(self, *new_batch_shape) -> Self:
        cur_batch_shape = self.batch_shape
        n_elements = int(np.prod(cur_batch_shape))
        if n_unspec := sum(v < 0 for v in new_batch_shape):
            if n_unspec > 1:
                raise ValueError(f"Got more than one ({n_unspec}) -1 values")
            n_rem = int(np.prod([v for v in new_batch_shape if v > 0]))
            assert n_elements == n_rem or (n_rem > 0 and not n_elements % n_rem)
            new_batch_shape = tuple(
                v if v >= 0 else n_elements // n_rem if n_rem > 0 else 0
                for v in new_batch_shape
            )
        n = len(cur_batch_shape)
        return self.map_elementwise(lambda v: v.reshape(*new_batch_shape, *v.shape[n:]))

    def coerce_to(self, cls: Self | type[Self]) -> Self:
        if not isinstance(cls, type):
            cls = type(cls)
        if isinstance(self, cls):
            return self
        return {
            SplitSymDecomp: self.as_split,
            FlatSymDecomp: self.as_flat,
        }[cls]()


@serialisable
@attrs.frozen
class SymDecompDims:
    invariant: int  # invariant under symmetry operations
    space: int  # equivariant under spatial symmetry operations
    flavour: int  # equivariant under symmetry operations of flavours
    rep: RepSpec = standard_rep

    @property
    def n_space(self):
        return self.rep.n_space

    @property
    def n_flavours(self):
        return self.rep.n_flavours

    @property
    def representations(self) -> dict[str, jt.Float]:
        return {k: getattr(self, k) for k in ["invariant", "space", "flavour"]}

    def map_representations(
        self,
        fun: typing.Callable[[str, int], typing.Any],
        *other: Self,
        cls: type | None = None,
    ) -> typing.Any:
        kw = {
            k: fun(k, v, *[getattr(o, k) for o in other])
            for k, v in self.representations.items()
        }
        if cls is None:
            return attrs.evolve(self, **kw)
        return cls(**kw)

    @property
    def dims(self):
        return SimpleNamespace(
            invariant=self.invariant,
            space=(self.n_space, self.space),
            flavour=(self.n_flavours, self.flavour),
        )

    @property
    def total_channels(self) -> int:
        return (
            self.n_space * self.space + self.invariant + self.n_flavours * self.flavour
        )

    def validation_problems(self, embedding: SymDecompBase) -> str | None:
        if not self.rep.is_valid():
            return "representation"
        return embedding.validation_problems(self)

    def validate(self, embedding: SymDecompBase) -> bool:
        return not self.validation_problems(embedding)


class SplitSymDecomp(SymDecompBase):
    invariant: jt.Float[jt.Array, "... Ci"]
    space: jt.Float[jt.Array, "... R Cs"]
    flavour: jt.Float[jt.Array, "... F Cf"]
    rep: RepSpec = attrs.field(default=standard_rep, metadata=dict(static=True))

    element_names: typing.ClassVar[frozenset[str]] = frozenset(
        {"invariant", "space", "flavour"}
    )

    @classmethod
    def empty(
        cls, dims: SymDecompDims, batch: tuple[int, ...] = (), *, dtype=jnp.float32
    ) -> Self:
        ret = cls(
            invariant=jnp.empty(batch + (dims.invariant,), dtype=dtype),
            space=jnp.empty(batch + (dims.n_space, dims.space), dtype=dtype),
            flavour=jnp.empty(batch + (dims.n_flavours, dims.flavour), dtype=dtype),
            rep=dims.rep,
        )
        assert dims.validate(ret), dims.validation_problems(ret)
        return ret

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return np.broadcast_shapes(
            self.invariant.shape[:-1],
            self.space.shape[:-2],
            self.flavour.shape[:-2],
        )

    def as_split(self) -> Self:
        return self

    def as_flat(self, *, dtype: Dtype | None = None) -> "FlatSymDecomp":
        batch = self.batch_shape
        n_feat_dims = dict(space=2, invariant=1, flavour=2)
        reps = self.representations
        assert set(reps) == set(FlatSymDecomp.subrep_seq)
        # Build SymDecompDims from the actual feature dimensions
        dim = SymDecompDims(
            **{k: v.shape[-1] for k, v in reps.items()},
            rep=self.rep,
        )
        ret = FlatSymDecomp(
            data=jnp.concatenate(
                [
                    jnp.broadcast_to(vv, batch + vv.shape[-1:])
                    for vv in [
                        reps[k].reshape(*reps[k].shape[: -n_feat_dims[k]], -1)
                        for k in FlatSymDecomp.subrep_seq
                    ]
                ],
                axis=-1,
                dtype=dtype,
            ),
            dim=dim,
        )
        return ret

    def validation_problems(self, dims: SymDecompDims) -> str | None:
        try:
            self.batch_shape
        except ValueError:
            return f"batch mismatch: {self.shapes}"
        if self.rep != dims.rep:
            return "rep mismatch"
        if self.invariant.shape[-1] != dims.invariant:
            return "invariant dim mismatch"
        if self.flavour.shape[-2:] != (dims.n_flavours, dims.flavour):
            return f"flavour dim mismatch {self.flavour.shape} <> {(dims.n_flavours, dims.flavour)}"
        if self.space.shape[-2:] != (dims.n_space, dims.space):
            return (
                f"space dim mismatch {self.space.shape} <> {(dims.n_space, dims.space)}"
            )

    @property
    def representations(self) -> dict[str, jt.Float]:
        return {k: getattr(self, k) for k in ["invariant", "space", "flavour"]}

    def map_representations(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self, **kw
    ) -> Self:
        other = [o.coerce_to(self) for o in other]
        return attrs.evolve(
            self,
            **{
                k: fun(v, *[getattr(o, k) for o in other], **kw)
                for k, v in self.representations.items()
            },
        )

    @property
    def elements(self) -> dict[str, jt.Float]:
        return self.representations

    def map_elementwise(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self, **kw
    ) -> Self:
        return self.map_representations(fun, *other, **kw)

    @property
    def shapes(self):
        return SimpleNamespace(
            invariant=self.invariant.shape,
            space=self.space.shape,
            flavour=self.flavour.shape,
        )


class FlatSymDecomp(SymDecompBase):
    data: jt.Float[jt.Array, "... C"]
    dim: SymDecompDims = attrs.field(metadata=dict(static=True))

    subrep_seq: typing.ClassVar[tuple[str, ...]] = ("invariant", "space", "flavour")
    element_names: typing.ClassVar[frozenset[str]] = frozenset({"data"})

    @property
    def rep(self) -> RepSpec:
        return self.dim.rep

    @classmethod
    def empty(
        cls, dims: SymDecompDims, batch: tuple[int, ...] = (), *, dtype=np.float32
    ) -> Self:
        ret = cls(
            data=np.empty(batch + (dims.total_channels,), dtype=dtype),
            dim=dims,
        )
        assert dims.validate(ret), dims.validation_problems(ret)
        return ret

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.data.shape[:-1]

    def as_split(self) -> SplitSymDecomp:
        kw = {}
        d = self.data
        batch = d.shape[:-1]
        feat_shape = dict(
            space=(self.dim.n_space, self.dim.space),
            invariant=(self.dim.invariant,),
            flavour=(self.dim.n_flavours, self.dim.flavour),
        )
        for k in self.subrep_seq:
            v = feat_shape[k]
            n = np.prod(v)
            kw[k] = d[..., :n].reshape(*batch, *v)
            d = d[..., n:]
        assert not d.size
        return SplitSymDecomp(
            **kw,
            rep=self.dim.rep,
        )

    def as_flat(self, *, dtype: Dtype | None = None) -> Self:
        if dtype is not None:
            return attrs.evolve(self, data=self.data.astype(dtype))
        return self

    def validation_problems(self, dims: SymDecompDims) -> str | None:
        if self.dim != dims:
            return f"dim mismatch: {self.dim} != {dims}"
        if self.data.shape[-1] != dims.total_channels:
            return "total channel mismatch"

    @property
    def representations(self) -> dict[str, jt.Float]:
        return self.as_split().representations

    def map_representations(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self, **kw
    ) -> Self:
        return self.as_split().map_representations(fun, *other, **kw)

    @property
    def elements(self) -> dict[str, jt.Float]:
        return dict(data=self.data)

    def map_elementwise(
        self, fun: typing.Callable[[jt.Float], jt.Float], *other: Self, **kw
    ) -> Self:
        other = [o.coerce_to(self) for o in other]
        return attrs.evolve(
            self,
            data=fun(self.data, *[o.data for o in other], **kw),
        )

    @property
    def shapes(self):
        batch = self.batch_shape
        dim = self.dim
        return SimpleNamespace(
            invariant=batch + (dim.invariant,),
            space=batch + (dim.n_space, dim.space),
            flavour=batch + (dim.n_flavours, dim.flavour),
        )
