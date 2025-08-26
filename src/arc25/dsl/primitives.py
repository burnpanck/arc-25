import dataclasses
from types import MappingProxyType
from typing import Iterable, Literal, TypeAlias

import numpy as np
from scipy import ndimage

from .. import symmetry
from .types import *
from .types import (
    Axis8,
    Canvas,
    Color,
)
from .types import Coord as _Coord
from .types import (
    Dir4,
    Dir8,
    Image,
    Mask,
    MaskedImage,
    Paintable,
    Pattern,
    Rect,
    ShapeSpec,
    Transform,
)

Coord: TypeAlias = _Coord | tuple[int, int]


_evolve = dataclasses.replace

_start = set(locals())

# ---------------------------------------------------------------------------
# Stroke patterns
# ---------------------------------------------------------------------------


def pattern_solid(color: Color) -> Pattern:
    return Pattern((color,))


def pattern_dotted(
    color: Color, *, gap: int | None = None, length: int | None = None
) -> Pattern:
    """[color, None, None, ...] length = gap+1"""
    if (gap is None) == (length is None):
        raise ValueError(
            "`pattern_dotted` requires exactly one keywords of `gap=` or `length=`"
        )
    if gap is None:
        gap = length - 1
    assert gap >= 0
    return Pattern((color,) + (None,) * gap)


def pattern_cycle(*colors: Color) -> Pattern:
    assert len(colors) >= 1
    return Pattern(tuple(colors))


def advance_pattern(pattern: Pattern, phase: int) -> Pattern:
    i = phase % len(pattern.sequence)
    return Pattern(pattern.sequence[i:] + pattern.sequence[:i])


# ---------------------------------------------------------------------------
# Path builders (produce coordinates; no painting here)
# ---------------------------------------------------------------------------


def path_segment(start: Coord, end: Coord) -> list[Coord]:
    """
    Cells from start to end inclusive along an axis-aligned or 45° diagonal.
    Raise ValueError for other angles.
    """
    ...


def path_ray(
    start: Coord,
    dir: Dir4 | Dir8,
    stop: Mask,
    *,
    endpoint: Literal["exclude", "include"] = "exclude",
) -> list[Coord]:
    """
    Walk from start in a direction until a cell is hit where `stop` is True
    include that cell if `endpoint="include"`.
    If no such cell is hit, stop at the edge.
    """
    ...


def path_span(
    shape: ShapeSpec,
    *,
    axis: Axis8,
    index: int | None = None,
    through: Coord | None = None,
) -> list[Coord]:
    """https://numpy.org/doc/stable/reference/generated/numpy.linalg.LinAlgError.html
    Whole-canvas line passing `through` or at `index`:
    - ROW: k = r in [0..H-1], i = c in [0..W-1]
    - COL: k = c in [0..W-1], i = r in [0..H-1]
    - DIAG_MAIN (↘): k = c - r in [-(H-1) .. (W-1)], i = (c+r-|k|)//2
    - DIAG_ANTI (↗︎): k = c + r in [0 .. H+W-2], i = (c-r+|k|)//2
    Index is the `k` described above.
    The `i` indexes the resulting list, which is listed in order of increasing col index,
    or increasing column index if set to `ROW`
    """
    if (index is None) == (through is None):
        raise ValueError(
            "`path_span` requires exactly one of the `index=` or `through=` keywords"
        )


# ---------------------------------------------------------------------------
# Painter
# ---------------------------------------------------------------------------


def stroke(
    canvas: Paintable,
    path: Iterable[Coord],
    style: Color | Pattern,
    *,
    clip: Mask | None = None,
) -> Paintable:
    """
    Paint along `path` with a solid Color or a repeating Pattern.
    - If `clip` is provided, only cells with clip[r,c]==True are painted.
    - Path determines order; Pattern advances per visited cell (including gaps).
    - Returns a NEW Canvas (immutability by design).
    """
    ...


def paste(canvas: Paintable, image: Paintable, *, at: Coord | None = None) -> Paintable:
    """Paste `image` into canvas, with the upper left corner positioned at `at`.

    If `image` has a mask, only pixels that are `True` will be painted.
    `at` can be outside of the canvas, and any image pixels outside of the canvas will be ignored.
    """
    if isinstance(canvas, Canvas):
        return _evolve(canvas, image=paste(canvas.image, image, at=at))
    if isinstance(image, Canvas):
        image = image.image
    assert isinstance(canvas, (Image, MaskedImage))
    assert isinstance(image, (Image, MaskedImage))
    at = _Coord.to_array(at)
    dslc = tuple(
        slice(*v)
        for v in zip(
            np.maximum(0, at),
            np.minimum(canvas.shape, at + image.shape),
        )
    )
    sslc = tuple(
        slice(*v)
        for v in zip(
            np.maximum(0, -at),
            np.minimum(image.shape, canvas.shape - at),
        )
    )
    nimg = canvas._data.copy()
    dimg = nimg[dslc]
    simg = image._data[sslc]
    if isinstance(image, MaskedImage):
        smask = image._mask[sslc]
        dimg[...] = np.where(smask, simg, dimg)
    else:
        smask = True
        dimg[...] = simg
    if not isinstance(canvas, MaskedImage):
        return Image(_data=nimg)
    nmask = canvas._mask.copy()
    nmask[dslc] |= smask
    return MaskedImage(_data=nimg, _mask=nmask)


def fill(
    canvas: Paintable,
    style: Color | Pattern,
    *,
    dir: Dir8 | None,
    clip: Mask | None = None,
) -> Paintable: ...


def make_canvas(
    nrow: int, ncol: int, orientation: symmetry.SymOp = symmetry.SymOp.e
) -> Canvas:
    return Canvas.make((nrow, ncol), orientation=orientation)


def extract_image(canvas: Paintable | Mask, *, rect: Rect) -> Paintable | Mask:
    slc = rect.as_slices()
    match canvas:
        case Canvas():
            return _evolve(canvas, image=extract_image(canvas.image, rect=rect))
        case Image():
            return _evolve(canvas, _data=canvas._data[slc])
        case MaskedImage():
            return _evolve(canvas, _data=canvas._data[slc], _mask=canvas._mask[slc])
        case Mask():
            return _evolve(canvas, _mask=canvas._mask[slc])
        case _:
            raise TypeError(
                f"Invalid type for `canvas` argument: {type(canvas).__name__}"
            )


def apply_mask(canvas: Paintable, mask: Mask) -> Paintable:
    if canvas.shape != mask.shape:
        raise ValueError(
            f"Mismatched shapes; `canvas` has {canvas.shape}, `mask` has {mask.shape}"
        )
    match canvas:
        case Canvas():
            return _evolve(canvas, image=apply_mask(canvas.image, mask))
        case Image():
            return MaskedImage(canvas, _data=canvas._data, _mask=mask)
        case MaskedImage():
            return _evolve(canvas, _mask=canvas._mask & mask)
        case _:
            raise TypeError(
                f"Invalid type for `canvas` argument: {type(canvas).__name__}"
            )


def transform(canvas: Paintable | Mask, op: Transform) -> Paintable | Mask:
    match op:
        case Transform():
            sop = op.value
        case symmetry.SymOp():
            sop = op
        case _:
            raise TypeError(f"Invalid type for `op` argument: {type(op).__name__}")
    match canvas:
        case Canvas():
            return _evolve(
                canvas,
                image=transform(canvas.image, op),
                orientation=sop.combine(canvas.orientation),
            )
        case Image():
            return _evolve(canvas, _data=symmetry.transform_image(sop, canvas._data))
        case MaskedImage():
            return _evolve(
                canvas,
                _data=symmetry.transform_image(sop, canvas._data),
                _mask=symmetry.transform_image(sop, canvas._mask),
            )
        case Mask():
            return _evolve(
                canvas,
                _mask=symmetry.transform_image(sop, canvas._mask),
            )
        case _:
            raise TypeError(
                f"Invalid type for `canvas` argument: {type(canvas).__name__}"
            )


# ----------------------------------------------------------------------------
# Counting & stats
# ----------------------------------------------------------------------------


def count_colors(canvas: Paintable) -> ColorArray: ...


def most_common_color(
    canvas: Paintable, *, exclude: Color | set[Color] | None = None
) -> Color: ...


_structures = MappingProxyType(
    {
        4: ndimage.generate_binary_structure(2, 1),
        8: ndimage.generate_binary_structure(2, 2),
    }
)


def find_objects(objects: Mask, *, connectivity: Literal[4, 8] = 4) -> Iterable[Mask]:
    labeled_array, num_features = ndimage.label(
        objects._mask, _structures[connectivity]
    )
    for label in range(1, num_features + 1):
        yield Mask(labeled_array == label)


def find_bbox(mask: Mask) -> Rect:
    if not np.any(mask):
        raise ValueError("`find_bbox` requires that at least one cell is selected")
    lim = []
    for axis in range(2):
        proj = mask._mask.any(axis=axis)
        lim.append(np.flatnonzero(proj)[[0, -1]] + [0, 1])
    start, stop = [Coord(*v) for v in zip(*lim)]
    return Rect(start=start, stop=stop)


# ----------------------------------------------------------------------------
# Mask constructors & helpers
# ----------------------------------------------------------------------------


def path_to_mask(shape: ShapeSpec, path: Iterable[Coord]) -> Mask: ...


def new_mask_like(shape: ShapeSpec, *, fill: bool) -> Mask: ...


def mask_all(shape: ShapeSpec) -> Mask: ...


def mask_none(shape: ShapeSpec) -> Mask: ...


def mask_color(canvas: Paintable, color: Color | set[Color]) -> Mask:
    """Build a Mask from Color | set[Color]."""
    ...


def mask_unpainted(canvas: Paintable) -> Mask:
    match canvas:
        case Canvas():
            return mask_unpainted(canvas.image)
        case MaskedImage():
            return Mask(~canvas._mask)
        case Image():
            return mask_none(canvas)
        case _:
            raise TypeError(
                f"Invalid type for `canvas` argument: {type(canvas).__name__}"
            )


def mask_row(shape: ShapeSpec, i: int) -> Mask: ...


def mask_col(shape: ShapeSpec, j: int) -> Mask: ...


def cell_count(mask: Mask) -> int:
    pass


def masks_touch(a: Mask, b: Mask, *, connectivity: Literal[4, 8] = 4) -> bool:
    pass


def dilate(mask: Mask, k: int = 1, *, connectivity: Literal[4, 8] = 4) -> Mask: ...


def erode(mask: Mask, k: int = 1, *, connectivity: Literal[4, 8] = 4) -> Mask: ...


# -------------

_end = set(locals())

__all__ = sorted([k for k in _end - _start if not k.startswith("_")])
del _start, _end
