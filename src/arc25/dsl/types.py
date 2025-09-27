import abc
import math
import typing
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, overload

import numpy as np

from ..lib.compat import Self, StrEnum
from ..symmetry import D4

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
    IDENTITY = D4.e
    FLIP_LR = D4.x
    FLIP_UD = D4.y
    ROTATE_180 = D4.i
    FLIP_DIAG_MAIN = D4.t
    ROTATE_LEFT = D4.l
    ROTATE_RIGHT = D4.r
    FLIP_DIAG_ANTI = D4.d


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
class _VectorBase:
    row: int
    col: int

    def as_tuple(self):
        return (self.row, self.col)

    def as_array(self):
        return np.array([self.row, self.col])

    @classmethod
    def to_array(cls, obj: Self | tuple[int, int]) -> np.ndarray:
        if isinstance(obj, _VectorBase):
            obj = (obj.row, obj.col)
        ret = np.empty(2, int)
        ret[:] = obj
        return ret

    def __getitem__(self, idx: int) -> int:
        match idx:
            case -2 | 0:
                return self.row
            case -1 | 1:
                return self.col
        raise KeyError(idx)


@dataclass(frozen=True, slots=True)
class Vector(_VectorBase):
    _dir2vec = MappingProxyType(
        {
            d: dict(
                right=(0, 1),
                up_right=(-1, 1),
                up=(-1, 0),
                up_left=(-1, -1),
                left=(0, -1),
                down_left=(1, -1),
                down=(1, 0),
                down_right=(1, 1),
            )[d]
            for d in Dir8
        }
    )
    _vec2dir = MappingProxyType({v: k for k, v in _dir2vec.items()})

    @classmethod
    def coerce(cls, arg: Self | tuple[int, int]) -> Self:
        return _to_vec(arg)

    @classmethod
    def elementary_vector(cls, dir: Dir8) -> Self:
        return Vector(*cls._dir2vec[Dir8(dir)])

    def length(
        self,
        metric: typing.Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
    ):
        match metric:
            case "euclidean":
                return self.euclidean()
            case "manhattan":
                return self.manhattan()
            case "chebyshev":
                return self.chebyshev()
            case _:
                raise ValueError(
                    '`metric` must be one of {"euclidean","manhattan","chebyshev"}'
                )

    def euclidean(self):
        return math.sqrt(self.row**2 + self.col**2)

    def manhattan(self):
        return abs(self.row) + abs(self.col)

    def chebyshev(self):
        return max(abs(self.row), abs(self.col))

    def __add__(self, other: Self) -> Self:
        return Vector(self.row + other.row, self.col + other.col)

    def __sub__(self, other: Self) -> Self:
        return Vector(self.row - other.row, self.col - other.col)

    def __mul__(self, other: int | float) -> Self:
        return Vector(self.row * other, self.col * other)

    def __rmul__(self, other: int | float) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> Self:
        return Vector(self.row / other, self.col / other)

    def __rtruediv__(self, other: int | float) -> Self:
        return self.__truediv__(other)

    def __floordiv__(self, other: int | float) -> Self:
        return Vector(self.row // other, self.col // other)

    def __rfloordiv__(self, other: int | float) -> Self:
        return self.__floordiv__(other)


for d in Dir8:
    setattr(Vector, d.name.upper(), Vector.elementary_vector(d))


def _to_vec(vec: _VectorBase | tuple[int, int]):
    if isinstance(vec, Vector):
        return vec
    if isinstance(vec, _VectorBase):
        vec = vec.as_tuple()
    return Vector(*vec)


@dataclass(frozen=True, slots=True)
class Coord(_VectorBase):

    @classmethod
    def coerce(cls, arg: Self | tuple[int, int]) -> Self:
        return _to_coord(arg)

    def __add__(self, other: Vector) -> Self:
        return _to_coord(_to_vec(self) + other)

    @overload
    def __sub__(self, other: Vector) -> Self: ...

    @overload
    def __sub__(self, other: Self) -> Vector: ...

    def __sub__(self, other: Vector | Self) -> Vector | Self:
        match other:
            case Vector():
                wrap = _to_coord
            case Coord():
                wrap = lambda x: x
            case _:
                return NotImplemented
        return wrap(_to_vec(self) - _to_vec(other))


def _to_coord(vec: _VectorBase | tuple[int, int]):
    if isinstance(vec, Coord):
        return vec
    if isinstance(vec, _VectorBase):
        vec = vec.as_tuple()
    return Coord(*vec)


def _shape_from_spec(shape: "ShapeSpec") -> tuple[int, int]:
    match shape:
        case tuple():
            assert len(shape) == 2
            return shape
        case Image() | MaskedImage() | Mask():
            return shape.shape
        case _:
            raise TypeError(
                f"Invalid type for `shape` argument: {type(shape).__name__}"
            )


class HasShape(abc.ABC):
    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        pass


@dataclass(frozen=True, slots=True)
class Rect(HasShape):
    start: Coord
    stop: Coord

    @classmethod
    def full_shape(self, shape: "ShapeSpec") -> Self:
        shape = _shape_from_spec(shape)
        return Rect(start=Coord(0, 0), stop=Coord(*shape))

    @classmethod
    def make(
        cls,
        *,
        width: int | None = None,
        height: int | None = None,
        square_size: int | None = None,
        shape: tuple[int, int] | None = None,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
        topleft: Coord | None = None,
        bottomright: Coord | None = None,
    ) -> Self:
        if shape is not None:
            assert width is None and height is None
            width, height = shape
        if square_size is not None:
            assert width is None and height is None
            width = height = square_size
        if topleft is not None:
            assert top is None and left is None
            top, left = topleft.as_tuple()
        if bottomright is not None:
            assert bottom is None and right is None
            bottom, right = bottomright.as_tuple()
        assert (
            sum(v is not None for v in [left, right, width]) == 2
        ), 'Need exactly two among `["left","right","width"]`'
        assert (
            sum(v is not None for v in [top, bottom, height]) == 2
        ), 'Need exactly two among `["top","bottom","height"]`'
        if left is None:
            left = right + 1 - width
        if width is None:
            width = right + 1 - left
        if top is None:
            top = bottom + 1 - height
        if height is None:
            height = bottom + 1 - top
        start = Coord(top, left)
        return Rect(start=start, stop=start + Vector(height, width))

    def __str__(self):
        args = ",".join(
            f"{k}={getattr(self, k)}" for k in "left right top bottom".split()
        )
        return f"{type(self).__qualname__}({args})"

    @property
    def topleft(self) -> Coord:
        return self.start

    @property
    def bottomright(self) -> Coord:
        return self.stop - Vector(1, 1)

    @property
    def center(self) -> Coord:
        tl = self.topleft
        br = self.bottomright
        return tl + 0.5 * (br - tl)

    @property
    def top(self):
        return self.start.row

    @property
    def left(self):
        return self.start.col

    @property
    def bottom(self):
        return self.stop.row - 1

    @property
    def right(self):
        return self.stop.col - 1

    @property
    def height(self) -> tuple[int, int]:
        return max(0, self.stop.row - self.start.row)

    @property
    def width(self) -> tuple[int, int]:
        return max(0, self.stop.col - self.start.col)

    @property
    def shape(self) -> tuple[int, int]:
        return (_to_vec(self.stop) - _to_vec(self.start)).as_tuple()

    @property
    def finite(self) -> bool:
        return self.area > 0

    @property
    def area(self):
        return self.height * self.width

    def as_slices(self) -> tuple[slice, slice]:
        return tuple(
            slice(*v) for v in zip(self.start.as_tuple(), self.stop.as_tuple())
        )

    def contains(self, coord: Coord) -> bool:
        return all(
            lo <= v < hi
            for lo, v, hi in zip(
                *(c.as_tuple() for c in [self.start, Coord.coerce(coord), self.stop])
            )
        )


class AnyImage(HasShape):
    @abc.abstractmethod
    def __getitem__(self, coord: Coord) -> Color | None:
        pass


@dataclass(frozen=True, slots=True)
class Image(AnyImage):
    _data: np.ndarray

    @classmethod
    def from_array(cls, inp: np.ndarray) -> Self:
        ret = np.asarray(inp, dtype="i1")
        assert ret.ndim == 2
        assert 0 <= ret.min() and ret.max() < 10
        return Image(ret)

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    def __getitem__(self, coord: Coord) -> Color:
        coord = Coord.coerce(coord)
        return _index2color[self._data[coord.row, coord.col]]


@dataclass(frozen=True, slots=True)
class Mask:
    _mask: np.ndarray

    @classmethod
    def coerce(
        cls, arg: Self | np.ndarray, *, shape: "ShapeSpec | None" = None
    ) -> Self:
        if shape is not None:
            shape = _shape_from_spec(shape)
        match arg:
            case Mask():
                ret = arg
            case np.ndarray():
                assert arg.ndim == 2 and arg.dtype == bool
                ret = cls(arg)
            case Rect():
                if shape is None:
                    raise TypeError(
                        "`Rect` is only acceptable as mask where the mask shape is known beforehand"
                    )
                ret = np.zeros(shape, bool)
                ret[arg.as_slices()] = True
                ret = cls(ret)
            case _:
                raise TypeError(
                    f"Expected an argument that can be converted to a Mask; got {type(arg).__name__}"
                )
        if shape is not None and ret.shape != shape:
            raise ValueError(f"Expected a mask of shape {shape} but got {ret.shape}")
        return ret

    @property
    def shape(self):
        return self._mask.shape

    def as_numpy(self):
        return self._mask.copy()

    def count(self):
        return self._mask.sum()

    def __eq__(self, other: Self | np.ndarray | bool) -> Self:
        if isinstance(other, Mask):
            other = other._mask
        return Mask(self._mask == other)

    def __req__(self, other: Self | np.ndarray | bool) -> Self:
        return self.__req__(other)

    def __neq__(self, other: Self | np.ndarray | bool) -> Self:
        if isinstance(other, Mask):
            other = other._mask
        return Mask(self._mask != other)

    def __rneq__(self, other: Self | np.ndarray | bool) -> Self:
        return self.__rneq__(other)

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

    def __getitem__(self, coord: Coord) -> bool:
        coord = Coord.coerce(coord)
        return bool(self._mask[coord.row, coord.col])

    def all(self) -> bool:
        return self._mask.all()

    def any(self) -> bool:
        return self._mask.any()


@dataclass(frozen=True, slots=True)
class MaskedImage(AnyImage):
    _data: np.ndarray
    _mask: np.ndarray

    @property
    def shape(self):
        assert self._data.shape == self._mask.shape
        return self._data.shape

    def __getitem__(self, coord: Coord) -> Color | None:
        coord = Coord.coerce(coord)
        if not self._mask[coord.row, coord.col]:
            return None
        return _index2color[self._data[coord.row, coord.col]]


ShapeSpec: TypeAlias = HasShape | tuple[int, int]


@dataclass(frozen=True, slots=True)
class IOPair:
    input: Image
    output: Image | None = None


@dataclass(frozen=True, slots=True)
class ColorArray:
    data: np.ndarray

    def as_set(self) -> set[Color]:
        ret = set()
        for c, v in zip(Color, self.data):
            if v:
                ret.add(c)
        return ret

    def as_dict(self) -> dict[Color, int]:
        ret = dict()
        for c, v in zip(Color, self.data):
            if v:
                ret[c] = v
        return ret


# -------------

_end = set(locals())

__all__ = sorted([k for k in _end - _start if not k.startswith("_")])
del _start, _end
