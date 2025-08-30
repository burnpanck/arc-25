import dataclasses
import math
from types import MappingProxyType
from typing import Any, Iterable, Literal, TypeAlias

import numpy as np
from scipy import ndimage

from .. import symmetry
from .types import *
from .types import (
    Axis8,
    Canvas,
    Color,
    Coord,
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
    Vector,
    _color2index,
    _index2color,
    _shape_from_spec,
)

_evolve = dataclasses.replace


def _make_type_error(argument: Any, argument_name: str, type_signature: str):
    return TypeError(
        f"{type(argument).__name__!r} is an invalid type for argument {argument_name}."
        f" Expected {type_signature!r}"
    )


def _set_of_colors(color: Color | set[Color], *, argname: str = "color") -> set[Color]:
    match color:
        case str() | Color():
            return {Color(color)}
        case set() | list() | tuple():
            return {Color(c) for c in color}
        case _:
            raise _make_type_error(color, argname, "Color | set[Color]")


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
    raise NotImplementedError


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
    rng = Rect.full_shape(stop)
    step = Vector.elementary_vector(dir)
    ret = []
    cur = start
    while True:
        if not rng.contains(cur):
            break
        if stop[cur]:
            if endpoint == "include":
                ret.append(cur)
            break
        ret.append(cur)
        cur += step
    return ret


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
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Painter
# ---------------------------------------------------------------------------


class _BlackHole:
    def __setitem__(self, idx, value):
        pass


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
    match canvas:
        case Canvas():
            return _evolve(canvas, image=stroke(canvas.image, path, style, clip=clip))
        case MaskedImage():
            mask = canvas._mask.copy()
        case Image():
            mask = _BlackHole()
        case _:
            raise _make_type_error(canvas, "canvas", "Paintable")
    match style:
        case Color():
            pattern = Pattern((style,))
        case Pattern():
            pattern = style
        case _:
            raise _make_type_error(style, "style", "Color | Pattern")
    seq = tuple(_color2index[c] if c is not None else -1 for c in pattern.sequence)
    n = len(seq)
    assert n > 0
    ret = canvas._data.copy()
    for i, p in enumerate(path):
        if clip is not None and not clip[c]:
            continue
        c = seq[i % n]
        if c < 0:
            continue
        p = p.as_tuple()
        ret[p] = c
        mask[p] = True
    return _evolve(
        canvas,
        _data=ret,
        **(dict(_mask=mask) if not isinstance(mask, _BlackHole) else {}),
    )


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
    if at is not None:
        at = Coord.to_array(at)
    else:
        at = np.array([0, 0])
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
    dir: Dir8 | None = None,
    clip: Mask | None = None,
) -> Paintable:
    if isinstance(canvas, Canvas):
        return _evolve(canvas, image=fill(canvas.image, style, dir=dir, clip=clip))
    match style:
        case str() | Color():
            style = [Color(style)]
        case list():
            pass
        case _:
            raise _make_type_error(style, "style", "Color | Pattern")
    if clip is None:
        mask = True
    else:
        mask = clip._mask
    if len(style) > 1:
        if dir is None:
            raise ValueError("When filling with a pattern, `dir` cannot be `None`")
        raise NotImplementedError("Pattern-fill is not implemented")
    else:
        assert len(style) == 1
        (c,) = style
        fill_value = np.tile(_color2index[c], canvas.shape)
    new_img = np.where(mask, fill_value, canvas._data)
    match canvas:
        case Image():
            return _evolve(canvas, _data=new_img)
        case MaskedImage():
            return _evolve(canvas, _data=new_img, _mask=canvas._mask | mask)
        case _:
            raise _make_type_error(canvas, "canvas", "Paintable")


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
            raise _make_type_error(canvas, "canvas", "Paintable | Mask")


def apply_mask(canvas: Paintable, mask: Mask) -> Paintable:
    if canvas.shape != mask.shape:
        raise ValueError(
            f"Mismatched shapes; `canvas` has {canvas.shape}, `mask` has {mask.shape}"
        )
    match canvas:
        case Canvas():
            return _evolve(canvas, image=apply_mask(canvas.image, mask))
        case Image():
            return MaskedImage(_data=canvas._data, _mask=mask)
        case MaskedImage():
            return _evolve(canvas, _mask=canvas._mask & mask)
        case _:
            raise _make_type_error(canvas, "canvas", "Paintable")


def transform(canvas: Paintable | Mask, op: Transform) -> Paintable | Mask:
    match op:
        case Transform():
            sop = op.value
        case symmetry.SymOp():
            sop = op
        case _:
            raise _make_type_error(op, "op", "Transform")
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
            raise _make_type_error(canvas, "canvas", "Paintable | Mask")


# ----------------------------------------------------------------------------
# Counting & stats
# ----------------------------------------------------------------------------


def count_colors(canvas: Paintable) -> ColorArray:
    match canvas:
        case Canvas():
            return count_colors(canvas.image)
        case Image():
            mask = np.s_[:, :]
        case MaskedImage():
            mask = canvas._mask._mask
        case _:
            raise _make_type_error(canvas, "canvas", "Paintable")
    cells = canvas._data[mask].ravel()
    return ColorArray(np.bincount(cells, minlength=10))


def most_common_colors(
    input: ColorArray | Paintable, *, exclude: Color | set[Color] | None = None
) -> set[Color]:
    match input:
        case ColorArray():
            color_count = input
        case Canvas() | Image() | MaskedImage():
            color_count = count_colors(input)
        case _:
            raise _make_type_error(input, "input", "ColorArray | Paintable")
    match color_count:
        case ColorArray():
            color_count = color_count.data
        case np.ndarray():
            assert color_count.shape == (10,)
        case _:
            raise _make_type_error(color_count, "color_count", "ColorArray")
    for c in _set_of_colors(exclude, "exclude") if exclude is not None else []:
        color_count[_color2index[c]] = -1
    max_count = color_count.max()
    if max_count < 0:
        return set()
    return {_index2color[i] for i in np.flatnonzero(color_count == max_count)}


def identify_background(
    canvas: Paintable, *, mode: Literal["frequency", "edge"] = "frequency"
) -> Color:
    match mode:
        case "frequency":
            mask = mask_all(canvas)
        case "edge":
            mask = mask_all(canvas)
            mask = mask & ~erode(mask)
        case _:
            raise KeyError(
                f'Invalid `mode` {mode!r}, must be one of `["frequency","edge"]`'
            )
    counts = count_colors(apply_mask(canvas, mask))
    cs = most_common_colors(counts)
    if len(cs) != 1:
        raise RuntimeError(
            f"Ambigous background color with method `{mode!r}` ({counts=})"
        )
    (c,) = cs
    return c


_structures = MappingProxyType(
    {
        4: ndimage.generate_binary_structure(2, 1),
        8: ndimage.generate_binary_structure(2, 2),
    }
)


def find_objects(objects: Mask, *, connectivity: Literal[4, 8] = 4) -> tuple[Mask, ...]:
    """Returns one full-sized mask for each object found, in unspecified order.

    Objects are defined by their connectivity.
    """
    objects = Mask.coerce(objects)
    labeled_array, num_features = ndimage.label(
        objects._mask, _structures[connectivity]
    )
    ret = []
    for label in range(1, num_features + 1):
        ret.append(Mask(labeled_array == label))
    return tuple(ret)


def find_cells(cells: Mask) -> tuple[Coord]:
    cells = Mask.coerce(cells)
    return tuple(Coord(int(c), int(r)) for c, r in zip(*np.nonzero(cells._mask)))


def find_holes(object: Mask, *, connectivity: Literal[4, 8] = 4) -> tuple[Mask, ...]:
    """Returns one full-sized mask for each hole in each of the objects.

    Holes are defined as being completely enclosed by the object (i.e. not touching the edge).
    """
    object = Mask.coerce(object)
    complement = ~object
    edge = mask_all(object)
    edge = edge & ~erode(edge, connectivity=connectivity)
    ret = []
    for obj in find_objects(complement):
        if (obj & edge).count:
            continue
        ret.append(obj)
    return tuple(ret)


def find_bbox(mask: Mask) -> Rect:
    mask = Mask.coerce(mask)
    if not mask.any():
        raise ValueError("`find_bbox` requires that at least one cell is selected")
    lim = []
    for axis in range(2):
        proj = mask._mask.any(axis=axis)
        lim.append(np.flatnonzero(proj)[[0, -1]] + [0, 1])
    start, stop = [Coord(int(r), int(c)) for r, c in zip(*lim[::-1])]
    return Rect(start=start, stop=stop)


# ----------------------------------------------------------------------------
# Mask constructors & helpers
# ----------------------------------------------------------------------------


def path_to_mask(shape: ShapeSpec, path: Iterable[Coord]) -> Mask:
    ret = np.zeros(_shape_from_spec(shape), bool)
    rc = np.array([c.as_tuple() for c in path]).T
    r, c = rc
    ret[r, c] = True
    return Mask(ret)


def rect_to_mask(shape: ShapeSpec, rect: Rect) -> Mask:
    ret = np.zeros(_shape_from_spec(shape), bool)
    ret[rect.as_slices()] = True
    return Mask(ret)


def new_mask_like(shape: ShapeSpec, *, fill: bool) -> Mask:
    return Mask(np.tile(fill, _shape_from_spec(shape)).astype(bool))


def mask_from_string(descr: str) -> Mask:
    descr = descr.lstrip("[").rstrip("]")
    lines = descr.split("|")
    m = len(lines)
    n = len(lines[0])
    assert all(len(ln) == n and all(c in "xo" for c in ln) for ln in lines)
    ret = np.empty((m, n), bool)
    for i, ln in enumerate(lines):
        for j, c in enumerate(ln):
            ret[i, j] = c == "x"
    return Mask(ret)


def mask_all(shape: ShapeSpec) -> Mask:
    return Mask(np.ones(_shape_from_spec(shape), bool))


def mask_none(shape: ShapeSpec) -> Mask:
    return Mask(np.zeros(_shape_from_spec(shape), bool))


def mask_color(canvas: Paintable, color: Color | set[Color]) -> Mask:
    """Build a Mask from Color | set[Color]."""
    color = _set_of_colors(color)
    match canvas:
        case Canvas():
            return mask_color(canvas.image, color)
        case Image():
            mask = mask_all(canvas)
        case MaskedImage():
            mask = canvas._mask._mask
    ret = mask_none(canvas)
    for c in color:
        m = canvas._data == _color2index[c]
        ret |= m
    return ret & mask


def mask_unpainted(canvas: Paintable) -> Mask:
    match canvas:
        case Canvas():
            return mask_unpainted(canvas.image)
        case MaskedImage():
            return Mask(~canvas._mask)
        case Image():
            return mask_none(canvas)
        case _:
            raise _make_type_error(canvas, "canvas", "Paintable")


def mask_row(shape: ShapeSpec, i: int) -> Mask:
    raise NotImplementedError


def mask_col(shape: ShapeSpec, j: int) -> Mask:
    raise NotImplementedError


def correlate_masks(
    input: Mask, pattern: Mask, *, threshold: int | None = None
) -> Mask:
    """
    Essentially performs `scipy.ndimage.correlate(input,pattern,mode="constant",cval=0)>=threshold`
    """
    if threshold is None:
        threshold = pattern._mask.sum()
    return Mask(
        ndimage.correlate(
            input._mask.astype(int), pattern._mask.astype(int), mode="constant"
        )
        >= threshold
    )


def cell_count(mask: Mask) -> int:
    return mask.count


def masks_touch(a: Mask, b: Mask, *, connectivity: Literal[4, 8] = 4) -> bool:
    pass


def dilate(mask: Mask, k: int = 1, *, connectivity: Literal[4, 8] = 4) -> Mask:
    return Mask(ndimage.binary_dilation(mask._mask, _structures[connectivity], k))


def erode(mask: Mask, k: int = 1, *, connectivity: Literal[4, 8] = 4) -> Mask:
    return Mask(ndimage.binary_erosion(mask._mask, _structures[connectivity], k))


def connected_component(
    mask: Mask, seed: Coord | Mask, *, connectivity: Literal[4, 8] = 4
) -> Mask:
    raise NotImplementedError


def reduce_rect(
    rect: Rect,
    *,
    amount: int = 0,
    height: int | None = None,
    width: int | None = None,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> Rect | None:
    if height is None:
        height = amount
    if width is None:
        width = amount
    if left is None:
        left = width
    if right is None:
        right = width
    if top is None:
        top = height
    if bottom is None:
        bottom = height
    ret = Rect(
        start=rect.start + Vector(top, left),
        stop=rect.stop - Vector(bottom, right),
    )
    if not ret.area:
        return None
    return ret


def center_of_mass(obj: Mask | Rect) -> Coord:
    match obj:
        case Rect():
            if not obj.finite:
                raise ValueError("Empty rect has no center of mass")
            return Coord(0.5 * (obj.top + obj.bottom), 0.5 * (obj.left + obj.right))
        case Mask():
            h, w = obj.shape
            r, c = np.mgrid[:h, :w]
            m = obj._mask
            if not m.any():
                raise ValueError("Empty mask has no center of mass")
            return Coord(float(r[m].mean()), float(c[m].mean()))
        case _:
            raise _make_type_error(obj, "obj", "Mask | Rect")


def vec2dir8(vec: Vector) -> Dir8:
    lim = vec.length() * math.sin(math.pi / 8)
    elementary = tuple(
        0 if abs(v) < lim else -1 if v < 0 else 1 for v in [vec.row, vec.col]
    )
    return Vector._vec2dir[elementary]


def round2grid(coord: Coord) -> Coord:
    return Coord(*(int(round(v)) for v in coord.as_tuple()))


# -------------

_end = set(locals())

__all__ = sorted([k for k in _end - _start if not k.startswith("_")])
del _start, _end
