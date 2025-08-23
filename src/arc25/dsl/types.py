from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np

from ..symmetry import SymOp

Axis4 = StrEnum("Axis4", [(k.upper(), k) for k in "row col".split()])
Axis8 = StrEnum(
    "Axis8", [(k.upper(), k) for k in "row diag_anti col diag_main".split()]
)
Dir4 = StrEnum("Dir4", [(k.upper(), k) for k in "right up left down".split()])
Dir8 = StrEnum(
    "Dir8",
    [
        (k.upper(), k)
        for k in "right up_right up up_left left down_left down down_right".split()
    ],
)


class Color(StrEnum):
    BLACK = "#000000"
    BLUE = "#0074D9"
    RED = "#FF4136"
    GREEN = "#2ECC40"
    YELLOW = "#FFDC00"
    GRAY = "#AAAAAA"
    MAGENTA = "#F012BE"
    ORANGE = "#FF851B"
    CYAN = "#7FDBFF"
    BROWN = "#870C25"


@dataclass(frozen=True, slots=True)
class Pattern:
    sequence: tuple[Color | None, ...]  # None = gap (skip painting this cell)


@dataclass(frozen=True, slots=True)
class Coord:
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class Image:
    _data: np.ndarray

    @property
    def shape(self):
        return self._data.shape


@dataclass(frozen=True, slots=True)
class Mask:
    _mask: np.ndarray

    @property
    def shape(self):
        return self._mask.shape


@dataclass(frozen=True, slots=True)
class MaskedImage:
    _data: np.ndarray
    _mask: np.ndarray

    @property
    def shape(self):
        assert self._data.shape == self._mask.shape
        return self._data.shape


AnyImage: TypeAlias = Image | MaskedImage


@dataclass(frozen=True, slots=True)
class Canvas:
    image: Image | MaskedImage
    # describes how physical/original coordinates have been mapped to the current orientation
    orientation: SymOp = SymOp.e

    @property
    def shape(self):
        return self.image.shape


Paintable: TypeAlias = Canvas | AnyImage

ShapeSpec: TypeAlias = Paintable | tuple[int, int]


@dataclass(frozen=True, slots=True)
class ColorArray:
    data: np.ndarray
