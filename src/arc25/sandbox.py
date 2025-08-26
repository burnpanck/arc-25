import itertools
import traceback
from typing import Iterable, Literal, TypeAlias

import attrs
import numpy as np

from .dataset import Challenge, IAETriple, Solution
from .dsl import api as dsl
from .dsl import types
from .dsl.types import Canvas


@attrs.frozen(kw_only=True)
class ExampleEval:
    output: Canvas | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    # *_match is None if we don't know the correct output
    full_match: bool | None = None
    pixel_match: float | None = None


@attrs.frozen
class ChallengeEval:
    challenge: Challenge
    solution: Solution
    train_eval: tuple[ExampleEval, ...]
    test_eval: tuple[ExampleEval, ...]
    full_match: bool
    example_match: float
    pixel_match: float

    def get_eval_triples(
        self, subset: Literal["train", "test", "all"] = "all"
    ) -> Iterable[IAETriple]:
        for k in dict(all=["train", "test"]).get(subset, [subset]):
            for io, e in zip(getattr(self.challenge, k), getattr(self, f"{k}_eval")):
                yield io.compare_output(e.output)


async def evaluate_solution(challenge: Challenge, solution: Solution) -> ChallengeEval:
    chal = challenge
    sol = solution
    result = dict(train=[], test=[])
    for eset in ["train", "test"]:
        for io in getattr(chal, eset):
            glob = {k: getattr(dsl, k) for k in dsl.__all__}
            loc = dict()
            try:
                exec(sol.rule, globals=glob, locals=loc)
                solver = loc.get("solution")
                if solver is None:
                    raise ValueError("Rule must contain a function called `solution`")
                actual = solver(io.input)
                if not isinstance(actual, types.Canvas):
                    raise TypeError(
                        f"`solution` must return a `Canvas`, got `{type(actual).__name__}`"
                    )
            except Exception:
                error = "\n".join(traceback.format_exc().split("\n")[-12:])
                eval = ExampleEval(error=error, full_match=False, pixel_match=0)
            else:
                warnings = []
                if isinstance(actual.image, types.MaskedImage):
                    if not np.all(actual.image._mask):
                        warnings.append("Output had unpainted pixels")
                    mask = actual.image._mask
                else:
                    mask = True
                eval = ExampleEval(
                    output=actual,
                    warnings=tuple(warnings),
                )
                if io.output is not None:
                    if actual.shape != io.output.shape:
                        kw = dict(full_match=False, pixel_match=0)
                    else:
                        match = (actual.image._data == io.output.image._data) & mask
                        kw = dict(full_match=np.all(match), pixel_match=np.mean(match))
                    eval = attrs.evolve(eval, **kw)
            result[eset].append(eval)
    allv = list(itertools.chain(*result.values()))
    return ChallengeEval(
        challenge=challenge,
        solution=solution,
        **{f"{k}_eval": tuple(v) for k, v in result.items()},
        full_match=all(v.full_match for v in allv if v.full_match is not None),
        example_match=float(
            np.mean([v.full_match for v in allv if v.full_match is not None])
        ),
        pixel_match=float(
            np.mean([v.pixel_match for v in allv if v.pixel_match is not None])
        ),
    )
