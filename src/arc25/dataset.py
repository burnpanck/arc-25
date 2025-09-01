import contextlib
import dataclasses
import itertools
import json
import logging
import lzma
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal, Self

import anyio
import attrs
import cbor2
import numpy as np

from .dsl.types import Canvas, Image, IOPair
from .serialisation import deserialise, serialisable, serialise

logger = logging.getLogger(__name__)


@serialisable
@attrs.frozen
class IAETriple:
    input: Canvas
    actual: Canvas | None = None
    expected: Canvas | None = None

    @classmethod
    def compare_output(cls, example: IOPair, actual: Canvas) -> Self:
        return cls(input=example.input, actual=actual, expected=example.output)


@serialisable
@attrs.frozen
class Challenge:
    id: str
    train: tuple[IOPair, ...]
    test: tuple[IOPair | Canvas, ...]

    def get_empty_eval_triples(
        self, subset: Literal["train", "test", "all"] = "all"
    ) -> Iterable[IAETriple]:
        for k in dict(all=["train", "test"]).get(subset, [subset]):
            for io in getattr(self, k):
                yield IAETriple.compare_output(io, None)


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
                return Canvas(Image(np.array(v, dtype="i1")))
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
