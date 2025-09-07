import abc
import itertools
import random
import typing
from typing import Self

import attrs
import numpy as np

from ..dataset import Challenge, ReasonedSolution, RuleProps
from ..dsl.types import Color, Image, IOPair
from .base import ChallengeSynth, SynthChallenge, SynthSpec

tiny_spec = SynthSpec(max_cells=8 * 4, max_side_length=8, max_total_cells=3 * 8 * 4)


@attrs.frozen(kw_only=True)
class MostCommonColor(ChallengeSynth):
    @classmethod
    def rule_variations(cls, spec: SynthSpec) -> int:
        return 1 if spec.supports(io_pairs=3, side_length=2) else 0

    @classmethod
    def _make_variation(
        cls,
        spec: SynthSpec,
        idx: int,
        *,
        rgen: random.Random | None = None,
        n_impl_samples: int = 0,
    ) -> Self:
        return MostCommonColor(
            spec=spec,
            rules=tuple(
                ReasonedSolution(
                    rule_descr="""
- Identify the most common color among all cells
- Create a single-cell output of that color
""".strip(),
                    rule_impl=impl,
                    props=RuleProps(),
                )
                for impl in [
                    """
def solution(input: Image) -> AnyImage:
    c, = most_common_colors(input)
    output = make_canvas((1,1),fill=c)
    return output
""".strip(),
                    """
def solution(input: Image) -> AnyImage:
    c, = most_common_colors(input)
    return make_canvas((1,1),fill=c)
""".strip(),
                ]
            ),
        )

    @property
    def inherent_entropy(self) -> float:
        return 0.3

    def _sample_challenges(
        self, rgen: random.Random, *, k: int
    ) -> typing.Iterable[SynthChallenge]:
        """Sample `k` challenges without replacement."""
        s = self.spec
        slim = min(5, s.max_side_length)
        hlim = min(slim, self.spec.max_cells // 2)
        n_samples = k
        for _ in range(n_samples):
            pairs = []
            for _ in range(4):
                w = rgen.randint(2, hlim)
                h = rgen.randint(2, min(slim, self.spec.max_cells // w))
                if rgen.randint(0, 1):
                    w, h = h, w
                n = h * w
                nc = rgen.randint(2, 10)
                clist = list(rgen.sample(list(range(10)), k=nc))
                cmx = clist.pop(rgen.randint(0, len(clist) - 1))
                nmx = rgen.randint((n + nc - 1) // nc + 1, n)
                cells = [cmx] * nmx + list(
                    rgen.sample(
                        list(
                            itertools.chain(
                                *[itertools.repeat(c, nmx - 1) for c in clist]
                            )
                        ),
                        k=n - nmx,
                    )
                )
                rgen.shuffle(cells)
                pairs.append(
                    IOPair(
                        input=Image(np.array(cells, "i1").reshape(h, w)),
                        output=Image.from_array([[cmx]]),
                    )
                )
            yield self._make_chal(
                train=tuple(pairs[1:]),
                test=tuple(pairs[:1]),
            )
