import contextlib
import json
import zipfile
from pathlib import Path
from typing import Self

import anyio
import attrs
import numpy as np

from .dsl.types import Canvas, Image


@attrs.frozen
class IAETriple:
    input: Canvas
    actual: Canvas
    expected: Canvas | None = None


@attrs.frozen
class IOPair:
    input: Canvas
    output: Canvas | None = None

    def compare_output(self, actual: Canvas) -> IAETriple:
        return IAETriple(input=self.input, actual=actual, expected=self.output)


@attrs.frozen
class Challenge:
    id: str
    train: tuple[IOPair, ...]
    test: tuple[IOPair | Canvas, ...]


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


@attrs.frozen
class Dataset:
    id: str
    title: str = attrs.field(
        default=attrs.Factory(lambda self: self.id.title(), takes_self=True)
    )
    challenges: dict[str, Challenge] = attrs.field(factory=dict)

    @classmethod
    async def load(
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
        self.challenges.update(dataset)
        return self


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
        if sol.rule.strip():
            await (db_root / f"{id}.py").write_text(sol.rule)
        if sol.explanation.strip():
            await (db_root / f"{id}.txt").write_text(sol.explanation)
        self.solutions[id] = sol
