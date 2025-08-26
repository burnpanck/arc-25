import typing
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Self, TypeAlias

import numpy as np

from ..symmetry import SymOp

_start = set(locals())

# --------

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


class Transform(Enum):
    IDENTITY = SymOp.e
    FLIP_LR = SymOp.x
    FLIP_UD = SymOp.y
    ROTATE_180 = SymOp.i
    FLIP_DIAG_MAIN = SymOp.t
    ROTATE_LEFT = SymOp.l
    ROTATE_RIGHT = SymOp.r
    FLIP_DIAG_ANTI = SymOp.d


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


_color2index = MappingProxyType({v: k for k, v in enumerate(Color)})
_index2color = tuple(Color)


@dataclass(frozen=True, slots=True)
class Pattern:
    sequence: tuple[Color | None, ...]  # None = gap (skip painting this cell)


@dataclass(frozen=True, slots=True)
class Coord:
    row: int
    col: int

    def as_tuple(self):
        return (self.row, self.col)

    @classmethod
    def to_array(cls, obj: Self | tuple[int, int]) -> np.ndarray:
        if isinstance(obj, Coord):
            obj = (obj.row, obj.col)
        ret = np.empty(2, int)
        ret[:] = obj
        return ret


def _shape_from_spec(shape: "ShapeSpec") -> tuple[int, int]:
    match shape:
        case tuple():
            assert len(shape) == 2
            return shape
        case Image() | MaskedImage() | Mask() | Canvas():
            return shape.shape
        case _:
            raise TypeError(
                f"Invalid type for `shape` argument: {type(shape).__name__}"
            )


@dataclass(frozen=True, slots=True)
class Rect:
    start: Coord
    stop: Coord

    @classmethod
    def full_shape(self, shape: "ShapeSpec") -> Self:
        shape = _shape_from_spec(shape)
        return Rect(start=Coord(0, 0), stop=Coord(*shape))

    def as_slices(self) -> tuple[slice, slice]:
        return tuple(
            slice(*v) for v in zip(self.start.as_tuple(), self.stop.as_tuple())
        )


@dataclass(frozen=True, slots=True)
class Image:
    _data: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape


@dataclass(frozen=True, slots=True)
class Mask:
    _mask: np.ndarray

    @property
    def shape(self):
        return self._mask.shape

    @property
    def as_numpy(self):
        return self._mask.copy()

    @property
    def count(self):
        return self._mask.sum()

    def __and__(self, other: Self | np.ndarray | bool) -> Self:
        if isinstance(other, Mask):
            other = other._mask
        return Mask(self._mask & other)

    def __rand__(self, other: Self | np.ndarray | bool) -> Self:
        return self.__and__(other)

    def __or__(self, other: Self | np.ndarray | bool) -> Self:
        if isinstance(other, Mask):
            other = other._mask
        return Mask(self._mask | other)

    def __ror__(self, other: Self | np.ndarray | bool) -> Self:
        return self.__or__(other)

    def __xor__(self, other: Self | np.ndarray | bool) -> Self:
        if isinstance(other, Mask):
            other = other._mask
        return Mask(self._mask ^ other)

    def __rxor__(self, other: Self | np.ndarray | bool) -> Self:
        return self.__xor__(other)

    def __invert__(self) -> Self:
        return Mask(~self._mask)


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

    @classmethod
    def make(
        cls,
        shape: tuple[int, int],
        *,
        orientation: SymOp = SymOp.e,
        fill: Color | None = None,
    ) -> Self:
        idata = np.tile(Color(fill).index if fill is not None else 0, shape)
        if fill is None:
            image = MaskedImage(_data=idata, _mask=np.zeros(shape, bool))
        else:
            image = Image(_data=idata)
        return cls(image=image, orientation=orientation)

    @property
    def shape(self):
        return self.image.shape


Paintable: TypeAlias = Canvas | AnyImage

ShapeSpec: TypeAlias = Paintable | tuple[int, int]


@dataclass(frozen=True, slots=True)
class ColorArray:
    data: np.ndarray


# -------------

_end = set(locals())

__all__ = sorted([k for k in _end - _start if not k.startswith("_")])
del _start, _end
