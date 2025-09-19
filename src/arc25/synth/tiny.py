import abc
import itertools
import random
import typing

import attrs
import numpy as np

from ..dataset import Challenge, Explicitness, ReasonedSolution, RuleProps
from ..dsl import primitives as P
from ..dsl.types import Color, Image, IOPair, _color2index, _index2color
from ..lib.compat import Self
from .base import ChallengeSynth, SynthChallenge, SynthSpec

tiny_spec = SynthSpec(max_cells=8 * 4, max_side_length=8, max_total_cells=5 * 8 * 4)


@attrs.frozen
class ShapeGen:
    spec: SynthSpec
    min_size: int
    max_size: int

    def __call__(self, rgen: random.Random) -> tuple[int, int]:
        slim = min(self.max_size, self.spec.max_side_length)
        max_cells = rgen.randint(self.min_size**2, self.spec.max_cells)
        w = rgen.randint(self.min_size, min(slim, max_cells // self.min_size))
        h = rgen.randint(self.min_size, min(slim, max_cells // w))
        if rgen.randint(0, 1):
            w, h = h, w
        return h, w


@attrs.frozen(kw_only=True)
class NoiseBGGen:
    noise: float = 0.5
    base: Color | None = None
    include: frozenset[Color] | None = None
    exclude: frozenset[Color] | None = None

    _colors: tuple[int, ...] = attrs.field(
        default=attrs.Factory(
            lambda self: tuple(
                sorted(
                    _color2index[c]
                    for c in (
                        frozenset(Color)
                        if self.include is None
                        else frozenset(self.include)
                    )
                    - (frozenset() if self.exclude is None else frozenset(self.exclude))
                )
            ),
            takes_self=True,
        )
    )

    def __call__(self, rgen: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        cs = np.array(self._colors, dtype="i1")
        noise = cs[rgen.integers(0, cs.size, size=shape)]
        m = rgen.random(shape) < self.noise
        base = rgen.choice(cs) if self.base is None else _color2index[self.base]
        return np.where(m, noise, base)


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
        return cls(
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
        shg = ShapeGen(s, 2, 5)
        n_samples = k
        for _ in range(n_samples):
            pairs = []
            for _ in range(4):
                h, w = shg(rgen)
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


@attrs.frozen(kw_only=True)
class DrawBBox(ChallengeSynth):
    filled: bool
    marker: Color
    color: Color

    @classmethod
    def rule_variations(cls, spec: SynthSpec) -> int:
        return 2 * 9 * 8 if spec.supports(io_pairs=3, side_length=4, cells=16) else 0

    @classmethod
    def _make_variation(
        cls,
        spec: SynthSpec,
        idx: int,
        *,
        rgen: random.Random | None = None,
        n_impl_samples: int = 0,
    ) -> Self:
        filled = bool(idx & 1)
        marker = (idx // 2) % 9 + 1
        color = (idx // 18) % 8 + 1
        if color >= marker:
            color += 1
        marker, color = [_index2color[c] for c in [marker, color]]
        body = [
            f"markers = mask_color(input, {marker.name.upper()})",
            "bbox = find_bbox(markers)",
            "rect_mask = rect_to_mask(input.shape, bbox)",
        ]
        if not filled:
            body.append("rect_mask = rect_mask & ~erode(rect_mask)")
        body.append(f"output = fill(input, {color.name.upper()}, clip=rect_mask)")
        body = "\n    ".join(body)
        impl = f"""
def solution(input: Image) -> AnyImage:
    {body}
    return output
""".strip()
        return cls(
            filled=filled,
            marker=marker,
            color=color,
            spec=spec,
            rules=(
                ReasonedSolution(
                    rule_descr=f"""
- Find the smallest rectangle that encloses all {marker.name.lower()} cells
- {"Fill" if filled else "Stroke"} that rectangle in {color.name.lower()}
""".strip(),
                    rule_impl=impl,
                    props=RuleProps(
                        explicit_colors={
                            c: Explicitness.explicit for c in [marker, color]
                        }
                        | {Color.BLACK: Explicitness.mixed},
                    ),
                ),
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
        shg = ShapeGen(s, 4, 10)
        bgg = NoiseBGGen(noise=0.3, exclude=[self.marker, self.color])
        n_samples = k
        nprng = np.random.default_rng(seed=rgen.randint(0, (1 << 32) - 1))
        mci = _color2index[self.marker]
        for _ in range(n_samples):
            pairs = []
            for _ in range(4):
                shape = shg(rgen)
                bg = bgg(nprng, shape)
                n = rgen.randint(2, 5)
                idx = nprng.integers(0, bg.size - 1, size=n)
                bg.flat[idx] = mci
                input = Image(bg)
                bbox = P.find_bbox(P.mask_color(input, self.marker))
                mask = P.rect_to_mask(shape, bbox)
                if not self.filled:
                    mask = mask & ~P.erode(mask)
                output = P.fill(input, self.color, clip=mask)
                pairs.append(
                    IOPair(
                        input=input,
                        output=output,
                    )
                )
            yield self._make_chal(
                train=tuple(pairs[1:]),
                test=tuple(pairs[:1]),
            )
