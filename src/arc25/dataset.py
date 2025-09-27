import contextlib
import dataclasses
import enum
import functools
import itertools
import json
import logging
import lzma
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal

import anyio
import attrs
import cbor2
import numpy as np

from .dsl.types import Axis8, Color, Dir8, Image, IOPair
from .lib.compat import Self, StrEnum
from .serialisation import deserialise, serialisable, serialise
from .symmetry import D4

logger = logging.getLogger(__name__)


@serialisable
@attrs.frozen
class IAETriple:
    input: Image
    actual: Image | None = None
    expected: Image | None = None

    @classmethod
    def compare_output(cls, example: IOPair, actual: Image) -> Self:
        return cls(input=example.input, actual=actual, expected=example.output)


@serialisable
@attrs.frozen(kw_only=True)
@functools.total_ordering
class Challenge:
    id: str
    train: tuple[IOPair, ...]
    test: tuple[IOPair | Image, ...]

    def get_empty_eval_triples(
        self, subset: Literal["train", "test", "all"] = "all"
    ) -> Iterable[IAETriple]:
        for k in dict(all=["train", "test"]).get(subset, [subset]):
            for io in getattr(self, k):
                yield IAETriple.compare_output(io, None)

    def canonicalise(self) -> Self:
        return attrs.evolve(
            self,
            train=tuple(sorted(self.train, key=self._key_fn)),
            test=tuple(
                sorted(
                    self.test,
                    key=lambda v: self._key_fn(v.input if isinstance(v, IOPair) else v),
                )
            ),
        )

    def remove_test_output(self) -> Self:
        return attrs.evolve(
            self,
            test=tuple(
                sorted(
                    (v.input if isinstance(v, IOPair) else v for v in self.test),
                    key=self.key_fn,
                )
            ),
        )

    def __eq__(self, other: object) -> bool:
        if self.__class__ is other.__class__:
            return self._key() == other._key()
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if self.__class__ is other.__class__:
            return self._key() < other._key()
        return NotImplemented

    def __hash__(self):
        return hash(self._key())

    def _key(self):
        return self._key_fn(self)

    @classmethod
    def _key_fn(cls, obj):
        doit = cls._key_fn
        match obj:
            case Challenge():
                test = []
                for v in obj.test:
                    if isinstance(v, IOPair):
                        i = doit(v.input)
                        o = doit(v.output)
                        k = i
                        v = (i, o)
                    else:
                        k = v = doit(v)
                    test.append((k, v))
                return (
                    tuple(sorted(doit(obj.train))),
                    tuple(v for _, v in sorted(test, key=lambda kv: kv[0])),
                )
            case _ if dataclasses.is_dataclass(obj):
                return tuple(
                    doit(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                )
            case np.ndarray():
                return (
                    obj.shape,
                    str(obj.dtype),
                    bytes(obj.ravel()),
                )
            case list() | tuple():
                return tuple(doit(v) for v in obj)
            case None | int() | str():
                return obj
        raise TypeError(type(obj).__name__)


def parse_inputs(v, had_list=False, id=None):
    match v:
        case dict():
            v = {kk: parse_inputs(vv, had_list) for kk, vv in v.items()}
            if had_list:
                return IOPair(**v)
            else:
                assert id is not None
                return Challenge(id=id, **v)
        case list():
            if not had_list:
                return tuple(parse_inputs(vv, True) for vv in v)
            else:
                return Image(np.array(v, dtype="i1"))
        case _:
            raise TypeError(f"Unsupported type {type(v).__name__}")


def load_json(root: Path, relative: Path | str, mode="r"):
    match root.suffix:
        case ".zip":
            with zipfile.ZipFile(root, "r") as zfh:
                fh = zfh.open(relative)
                return json.load(fh)
        case "":
            with open(root / relative, mode) as fh:
                return json.load(fh)
        case _:
            raise KeyError(f"Unknown suffix {root.suffix!r}")


@serialisable
@attrs.frozen
class Dataset:
    id: str
    challenges: dict[str, Challenge] = attrs.field(factory=dict)
    subsets: dict[str, frozenset[str]] = attrs.field(factory=dict)

    @classmethod
    async def load_from_json(
        cls,
        *,
        root: Path,
        challenges: Path | str,
        solutions: Path | str | None = None,
        **kw,
    ) -> dict[str, Challenge]:
        self = cls(**kw)
        challenges = {
            k: parse_inputs(v, id=k)
            for k, v in (
                await anyio.to_thread.run_sync(load_json, root, challenges, "rt")
            ).items()
        }
        if solutions is not None:
            solutions = {
                k: parse_inputs(v)
                for k, v in (
                    await anyio.to_thread.run_sync(load_json, root, solutions, "rt")
                ).items()
            }
            dataset = {}
            for k, v in challenges.items():
                dataset[k] = Challenge(
                    id=k,
                    train=v.train,
                    test=tuple(
                        IOPair(
                            input=i.input,
                            output=o,
                        )
                        for i, o in zip(v.test, solutions[k])
                    ),
                )
        else:
            dataset = challenges
        ret = dict(
            challenges=dataset,
            subsets={self.id: frozenset(self.challenges)},
        )
        return attrs.evolve(self, **{k: MappingProxyType(v) for k, v in ret.items()})

    @classmethod
    async def from_binary(cls, src: Path) -> Self:
        encoded = await anyio.Path(src).read_bytes()
        match src.suffix:
            case ".xz":
                encoded = lzma.decompress(encoded)
            case ".cbor":
                pass
            case _:
                raise KeyError(f"Unsupported compression format {src.suffix!r}")
        ret = deserialise(cbor2.loads(encoded))
        challenges = {}
        for k, v in ret.challenges.items():
            challenges[k] = attrs.evolve(v, train=tuple(v.train), test=tuple(v.test))
        subsets = {k: frozenset(v) for k, v in ret.subsets.items()}
        return attrs.evolve(
            ret,
            challenges=MappingProxyType(challenges),
            subsets=MappingProxyType(subsets),
        )


class Explicitness(StrEnum):
    # this property is completely absent from the example
    absent = "absent"
    # this property is implicity implied by the examples,
    # but not explicitly referenced by the rule
    implicit = "implicit"
    # there are some rules among the shortest reasonable rules
    # which include that property explicitly, but others don't.
    # this may often apply to a background property,
    # which is easily enough inferred,
    # but it is about equally reasonable to specify property explictly
    # in the rule
    mixed = "mixed"
    # all known shortes reasonable rules require explicit mention
    # of the property
    explicit = "explicit"


@serialisable
@attrs.frozen(kw_only=True)
class RuleProps:
    """Describes properties of rules that are objective, but nontrivial to infer."""

    # these colors explicitly carry meaning; only `mixed` and `explicit`
    # appear in rule properties that are not tied to an example.
    explicit_colors: dict[Color, Explicitness] = attrs.field(
        factory=lambda: MappingProxyType({})
    )
    # Symmetries, and their effect on the rule. For symmetries that are not mentioned
    # in this attribute, the lowest explicitness of all atomic decompositions is
    # taken to be correct, or `explicit` if there aren't any.
    symmetries: dict[D4, Explicitness] = attrs.field(
        factory=lambda: MappingProxyType({})
    )


@serialisable
@attrs.frozen(kw_only=True)
class ReasonedSolution:
    descr: dict[str, str] = attrs.field(factory=lambda: MappingProxyType({}))
    rule_descr: str | None = None
    impl_plan_descr: str | None = None
    rule_impl: str | None = None
    # if none, we must assume anything that cannot be programmatically inferred as explicit
    props: RuleProps | None = None


@serialisable
@attrs.frozen(kw_only=True)
class WrongSolution:
    """Explains why a specific solution attempt is wrong."""

    attempt: ReasonedSolution
    error: str


@serialisable
@attrs.frozen
class Solution:
    id: str
    explanation: str
    rule: str

    @classmethod
    def make(cls, id, *, explanation="", rule=""):
        return Solution(id=id, explanation=explanation, rule=rule)

    @property
    def is_empty(self):
        return not self.explanation.strip() and not self.rule.strip()


@serialisable
@attrs.frozen
class SolutionDB:
    root: Path
    solutions: dict[str, Solution] = attrs.field(factory=dict)

    @classmethod
    async def load(cls, root: Path) -> Self:
        self = cls(root=root)
        await self.update_from_disk()
        return self

    async def update_from_disk(self):
        root = anyio.Path(self.root)
        async for f in root.glob("*.py"):
            id = f.with_suffix("").name
            rule = await f.read_text()
            self.solutions[id] = attrs.evolve(
                self.solutions.get(id, Solution.make(id)),
                rule=rule,
            )
        async for f in root.glob("*.txt"):
            id = f.with_suffix("").name
            explanation = await f.read_text()
            self.solutions[id] = attrs.evolve(
                self.solutions.get(id, Solution.make(id)),
                explanation=explanation,
            )

    async def store(self, solution: Solution):
        db_root = anyio.Path(self.root)
        sol = solution
        id = sol.id
        self.solutions[id] = sol
        if sol.rule.strip():
            await (db_root / f"{id}.py").write_text(sol.rule)
        if sol.explanation.strip():
            await (db_root / f"{id}.txt").write_text(sol.explanation)


async def load_datasets(challenges_root):
    logger.info(f"Loading data from {challenges_root}")
    datasets = {}
    for k in ["training", "evaluation", "test"]:
        ds = await Dataset.load(
            id=k,
            root=challenges_root,
            challenges=f"arc-agi_{k}_challenges.json",
            solutions=f"arc-agi_{k}_solutions.json" if k != "test" else None,
        )
        datasets[k] = ds
    datasets["combined"] = Dataset(
        id="combined",
        challenges=dict(
            itertools.chain(*[ds.challenges.items() for ds in datasets.values()])
        ),
    )
    for v in datasets.values():
        logger.debug(f"Dataset {v.title} has {len(v.challenges)} challenges")
    return datasets
