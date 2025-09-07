import abc
import dataclasses
import enum
import random
import re
import typing
from types import MappingProxyType

import attrs
import numpy as np

from . import dataset
from .dsl.types import Color, Image, MaskedImage, _color2index

all_colors_rex = re.compile(
    r"\b(" + "|".join(re.escape(c.name.lower()) for c in Color) + r")\b", re.I
)


def permute_colors(arg: typing.Any, mapping: dict[Color, Color]) -> typing.Any:
    # avoid circulat import
    from arc25 import prompts

    # these are lazy; we want to be able to abuse this function for prompt tuning
    imap = None
    smap = None

    def apply(obj: typing.Any, strs=None, path=()) -> typing.Any:
        nonlocal imap, smap
        match obj:
            case Color():
                return mapping[obj]
            case dataset.Challenge():
                assert strs is None
                return attrs.evolve(
                    obj, **apply(attrs.asdict(obj, recurse=False), False, path)
                )
            case dataset.Solution():
                assert strs is None
                return attrs.evolve(
                    obj, **apply(attrs.asdict(obj, recurse=False), True, path)
                )
            case prompts.ReasonedSolution():
                assert strs is None
                return attrs.evolve(
                    obj, **apply(attrs.asdict(obj, recurse=False), True, path)
                )
            case Image() | MaskedImage():
                if imap is None:
                    imap = np.array([_color2index[mapping.get(c, c)] for c in Color])
                    assert (np.unique(imap) == np.r_[:10]).all()
                return dataclasses.replace(obj, _data=imap[obj._data])
            case _ if dataclasses.is_dataclass(obj):
                return dataclasses.replace(
                    obj,
                    **{
                        f.name: apply(getattr(obj, f.name), strs, path + (f.name,))
                        for f in dataclasses.fields(obj)
                    },
                )
            case _ if attrs.has(obj):
                return attrs.evolve(
                    obj, **apply(attrs.asdict(obj, recurse=False), strs, path)
                )
            case str():
                if not strs:
                    return obj
                if smap is None:
                    smap = {
                        style(c.name): style(mapping.get(c, c).name)
                        for c in Color
                        for style in [str.title, str.lower, str.upper]
                    }
                return all_colors_rex.sub(
                    lambda m: smap.get(m.group(1)),
                    obj,
                )
            case None | int() | enum.Enum():
                return obj
            case list() | tuple():
                return type(obj)(
                    apply(v, strs, path + (f"[{i}]",)) for i, v in enumerate(obj)
                )
            case dict():
                return {k: apply(v, strs, path + (k,)) for k, v in obj.items()}
            case _:
                raise TypeError(
                    f"Unknown object of type {type(obj).__name__} at {'.'.join(path)}"
                )

    return apply(arg)


class AugmentationBase(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def random(cls, rgen: random.Random) -> typing.Self:
        pass

    @abc.abstractmethod
    def __call__(
        self, obj: dataset.Challenge | dataset.Solution
    ) -> dataset.Challenge | dataset.Solution:
        pass


@attrs.frozen
class ColorPermutationBase(AugmentationBase):
    mapping: dict[Color, Color]

    def __call__(
        self, obj: dataset.Challenge | dataset.Solution
    ) -> dataset.Challenge | dataset.Solution:
        return permute_colors(obj, self.mapping)


@attrs.frozen
class RandomColorPermutation(ColorPermutationBase):
    @classmethod
    def random(cls, rgen: random.Random) -> typing.Self:
        colors = tuple(Color)
        new_colors = list(colors)
        rgen.shuffle(new_colors)
        mapping = dict(zip(colors, new_colors))
        return cls(mapping=mapping)


@attrs.frozen
class RandomColorPermutationExcludingBlack(ColorPermutationBase):
    @classmethod
    def random(cls, rgen: random.Random) -> typing.Self:
        colors = tuple(Color)[1:]
        assert Color.BLACK not in colors
        new_colors = list(colors)
        rgen.shuffle(new_colors)
        mapping = dict(zip(colors, new_colors))
        return cls(mapping=mapping)


@attrs.frozen
class RandomColorSwap(ColorPermutationBase):
    @classmethod
    def random(cls, rgen: random.Random) -> typing.Self:
        colors = tuple(rgen.choices(Color), k=2, replace=False)
        new_colors = list(colors[::-1])
        mapping = dict(zip(colors, new_colors))
        return cls(mapping=mapping)


@attrs.frozen
class MultiAugmentationBase(AugmentationBase):
    # The value represents the probability to apply a random augmentation of the given type
    # independent, and in addition to, any other augmentations applied.
    _choices: typing.ClassVar[dict[type[AugmentationBase], float]]
    sequence: tuple[AugmentationBase, ...]

    @classmethod
    def random(cls, rgen: random.Random) -> typing.Self:
        for _retry in range(100):
            seq = [k.random(rgen) for k, v in cls._choices if v <= rgen.random()]
            if seq:
                return cls(sequence=tuple(seq))
        raise ValueError()


class MultiAugmentation(MultiAugmentationBase):
    _choices = MappingProxyType(
        {
            RandomColorPermutation: 0.2,
            RandomColorPermutationExcludingBlack: 0.8,
        }
    )
