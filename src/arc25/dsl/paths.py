from typing import Iterable, Literal, TypeAlias

from .types import (
    AnyImage,
    Axis8,
    Color,
    ColorArray,
)
from .types import Coord as _Coord
from .types import (
    Dir4,
    Dir8,
    Mask,
    Paintable,
    Pattern,
    ShapeSpec,
    SymOp,
)

Coord: TypeAlias = _Coord | tuple[int, int]

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
    ...


def fill(
    canvas: Paintable,
    style: Color | Pattern,
    *,
    dir: Dir8 | None,
    clip: Mask | None = None,
) -> Paintable: ...


def extract_image(canvas: Paintable, *, rect: ..., mask: Mask) -> AnyImage: ...


def transform(canvas: Paintable, symop: SymOp) -> Paintable: ...


# ----------------------------------------------------------------------------
# Counting & stats
# ----------------------------------------------------------------------------


def count_colors(canvas: Paintable) -> ColorArray[int]: ...


def most_common_color(
    canvas: Paintable, *, exclude: Color | set[Color] | None = None
) -> Color: ...


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


def mask_row(shape: ShapeSpec, i: int) -> Mask: ...


def mask_col(shape: ShapeSpec, j: int) -> Mask: ...


def cell_count(mask: Mask) -> int:
    pass


def masks_touch(a: Mask, b: Mask, *, connectivity: Literal[4, 8] = 4) -> bool:
    pass


def dilate(mask: Mask, k: int = 1, *, connectivity: Literal[4, 8] = 4) -> Mask: ...


def erode(mask: Mask, k: int = 1, *, connectivity: Literal[4, 8] = 4) -> Mask: ...
