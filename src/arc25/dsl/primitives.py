import dataclasses
import math
from types import MappingProxyType
from typing import Any, Iterable, Literal

import numpy as np
from scipy import ndimage

from .. import symmetry
from .types import (
    AnyImage,
    Axis8,
    Color,
    ColorArray,
    Coord,
    Dir4,
    Dir8,
    Image,
    Mask,
    MaskedImage,
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
        case None:
            return set()
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
    *,
    stop_mask: Mask | None = None,
    limit: int | None = None,
    shape: ShapeSpec | None = None,
    endpoint: Literal["exclude", "include"] = "exclude",
) -> list[Coord]:
    """
    Walk from start in a direction until a cell is hit where `stop_mask` is True
    include that cell if `endpoint="include"`.
    If no such cell is hit, stop at the edge.
    """
    step = Vector.elementary_vector(dir)
    if stop_mask is None:
        if shape is None:
            raise TypeError("When `stop` is missing, `shape` must be given")
        rng = Rect.full_shape(shape)
    else:
        stop_mask = Mask.coerce(stop_mask)
        rng = Rect.full_shape(stop_mask)
        if shape is not None and Rect.full_shape(shape) != rng:
            raise ValueError("`shape` and `stop_mask` disagree on shape")

    ret = []
    cur = start
    while True:
        if not rng.contains(cur):
            break
        if stop_mask is not None and stop_mask[cur]:
            if endpoint == "include":
                ret.append(cur)
            break
        ret.append(cur)
        cur += step
        if limit is not None and len(ret) >= limit:
            break
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
    canvas: AnyImage,
    path: Iterable[Coord],
    style: Color | Pattern,
    *,
    clip: Mask | None = None,
) -> AnyImage:
    """
    Paint along `path` with a solid Color or a repeating Pattern.
    - If `clip` is provided, only cells with clip[r,c]==True are painted.
    - Path determines order; Pattern advances per visited cell (including gaps).
    - Returns a NEW Image (immutability by design).
    """
    match canvas:
        case MaskedImage():
            mask = canvas._mask.copy()
        case Image():
            mask = _BlackHole()
        case _:
            raise _make_type_error(canvas, "canvas", "AnyImage")
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
        if clip is not None and not clip[p]:
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


def paste(
    canvas: AnyImage,
    image: AnyImage,
    *,
    topleft: Coord | None = None,
    bottomright: Coord | None = None,
    center: Coord | None = None,
) -> AnyImage:
    """Paste `image` into canvas.

    If `image` has a mask, only pixels that are `True` will be painted.

    Unless `image` has the same shape as `canvas`, the position
    needs to be specified using one of `topleft`, `bottomright`, `center`,
    specifying the position of the corresponding corner of `image` within `canvas`.
    """
    assert isinstance(canvas, (Image, MaskedImage))
    assert isinstance(image, (Image, MaskedImage))
    if bottomright is not None:
        assert topleft is None
        topleft = Coord.to_array(bottomright) - image.shape + 1
    if center is not None:
        assert topleft is None
        topleft = Coord.to_array(center) - np.array(image.shape) // 2
    if topleft is None:
        if image.shape != canvas.shape:
            raise ValueError(
                "When `image` has a different shape than `canvas`, the position needs to be specified"
            )
        topleft = np.array([0, 0])
    else:
        topleft = Coord.to_array(topleft)
    dslc = tuple(
        slice(*v)
        for v in zip(
            np.maximum(0, topleft),
            np.minimum(canvas.shape, topleft + image.shape),
        )
    )
    sslc = tuple(
        slice(*v)
        for v in zip(
            np.maximum(0, -topleft),
            np.minimum(image.shape, canvas.shape - topleft),
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
    canvas: AnyImage,
    style: Color | Pattern,
    *,
    dir: Dir8 | None = None,
    clip: Mask | None = None,
    pattern_origin: Coord | None = None,
) -> AnyImage:
    if clip is not None:
        clip = Mask.coerce(clip, shape=canvas)
    if pattern_origin is not None:
        pattern_origin = Coord.coerce(pattern_origin)
    match style:
        case str() | Color():
            style = [Color(style)]
        case Pattern():
            style = style.sequence
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
        if pattern_origin is None:
            raise ValueError(
                "When filling with a pattern, `pattern_origin` cannot be `None`"
            )
        h, w = canvas.shape
        phase = sum(
            (coord - ref) * unit
            for coord, ref, unit in zip(
                np.ogrid[:h, :w],
                pattern_origin.as_tuple(),
                Vector.elementary_vector(dir).as_tuple(),
            )
        )
        seq = np.array([_color2index[c] if c is not None else -1 for c in style])
        fill_value = seq[phase % seq.size]
        if None in style:
            mask = mask & (fill_value != -1)
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
            raise _make_type_error(canvas, "canvas", "AnyImage")


_fill_primitive = fill


def make_canvas(
    shape: ShapeSpec,
    *,
    fill: Color | None = None,
) -> AnyImage:
    shape = _shape_from_spec(shape)
    if fill is not None:
        return Image(_data=np.tile(_color2index[fill], shape).astype("i1"))
    return MaskedImage(
        _data=np.zeros(shape, "i1"),
        _mask=np.zeros(shape, bool),
    )


def extract_image(canvas: AnyImage | Mask, *, rect: Rect) -> AnyImage | Mask:
    slc = rect.as_slices()
    match canvas:
        case Image():
            return _evolve(canvas, _data=canvas._data[slc])
        case MaskedImage():
            return _evolve(canvas, _data=canvas._data[slc], _mask=canvas._mask[slc])
        case Mask():
            return _evolve(canvas, _mask=canvas._mask[slc])
        case _:
            raise _make_type_error(canvas, "canvas", "AnyImage | Mask")


def apply_mask(canvas: AnyImage, mask: Mask) -> AnyImage:
    mask = Mask.coerce(mask, shape=canvas.shape)
    match canvas:
        case Image():
            return MaskedImage(_data=canvas._data, _mask=mask._mask)
        case MaskedImage():
            return _evolve(canvas, _mask=canvas._mask & mask)
        case _:
            raise _make_type_error(canvas, "canvas", "AnyImage")


def transform(canvas: AnyImage | Mask, op: Transform) -> AnyImage | Mask:
    match op:
        case Transform():
            sop = op.value
        case symmetry.D4():
            sop = op
        case _:
            raise _make_type_error(op, "op", "Transform")
    match canvas:
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
            raise _make_type_error(canvas, "canvas", "AnyImage | Mask")


def pattern_error(canvas: AnyImage, pattern_shape: tuple[int, int]) -> int:
    """Count the number of cells which do not form a regular pattern.

    The pattern repetition frequency is assumed to as given in `pattern_shape`.
    If `canvas` is masked, only cells with paint are considered.
    """
    match canvas:
        case Image() | MaskedImage():
            pass
        case _:
            raise _make_type_error(canvas, "canvas", "AnyImage")
    rrep, crep = pattern_shape
    msk = canvas._mask if isinstance(canvas, MaskedImage) else None
    err = 0
    for i in range(rrep):
        for j in range(crep):
            slc = np.s_[i::rrep, j::crep]
            m = msk[slc] if msk is not None else np.s_[...]
            c = canvas._data[slc][m].ravel()
            cnt = np.bincount(c, minlength=10)
            err += int(c.size - cnt.max())
    return err


# ----------------------------------------------------------------------------
# Counting & stats
# ----------------------------------------------------------------------------


def count_colors(canvas: AnyImage) -> ColorArray:
    match canvas:
        case Image():
            mask = np.s_[:, :]
        case MaskedImage():
            mask = canvas._mask
        case _:
            raise _make_type_error(canvas, "canvas", "AnyImage")
    cells = canvas._data[mask].ravel()
    return ColorArray(np.bincount(cells, minlength=10))


def most_common_colors(
    input: ColorArray | AnyImage, *, exclude: Color | set[Color] | None = None
) -> set[Color]:
    match input:
        case ColorArray():
            color_count = input
        case Image() | MaskedImage():
            color_count = count_colors(input)
        case _:
            raise _make_type_error(input, "input", "ColorArray | AnyImage")
    match color_count:
        case ColorArray():
            color_count = color_count.data
        case np.ndarray():
            assert color_count.shape == (10,)
        case _:
            raise _make_type_error(color_count, "color_count", "ColorArray")
    for c in _set_of_colors(exclude, argname="exclude") if exclude is not None else []:
        color_count[_color2index[c]] = -1
    max_count = color_count.max()
    if max_count < 0:
        return set()
    return {_index2color[i] for i in np.flatnonzero(color_count == max_count)}


def identify_background(
    canvas: AnyImage, *, mode: Literal["frequency", "edge"] = "frequency"
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


def find_objects(
    objects: AnyImage | Mask,
    *,
    connectivity: Literal[4, 8] = 4,
    gap: int = 0,
    exclude: Color | set[Color] | None = None,
) -> Iterable[Mask]:
    """Returns one full-sized mask for each object found, in unspecified order.

    If `objects` is an Image (potentially with mask),
    objects are defined as connected components of a single color, excluding `exclude`.
    Otherwise, `objects` must be Mask-like, and objects are defined as connected
    components of `True` values.
    """
    match objects:
        case MaskedImage():
            mask = objects._mask
            colors = np.unique(objects._data[mask])
        case Image():
            mask = True
            colors = np.unique(objects._data)
        case _:
            objects = Mask.coerce(objects)
            if exclude is not None:
                raise ValueError("`exclude` must be None when operating on a Mask")
            if gap > 0:
                # gapped object search:
                fused_objects = fill_gaps(objects, connectivity=connectivity)
                for obj in find_objects(fused_objects, connectivity=connectivity):
                    yield obj & objects
            labeled_array, num_features = ndimage.label(
                objects._mask, _structures[connectivity]
            )
            for label in range(1, num_features + 1):
                yield Mask(labeled_array == label)
            return
    exclude = _set_of_colors(exclude)
    for c in colors:
        if _index2color[c] in exclude:
            continue
        yield from find_objects((objects._data == c) & mask, connectivity=connectivity)


def find_cells(cells: Mask) -> tuple[Coord]:
    cells = Mask.coerce(cells)
    return tuple(Coord(int(c), int(r)) for c, r in zip(*np.nonzero(cells._mask)))


def find_holes(object: Mask, *, connectivity: Literal[4, 8] = 4) -> Iterable[Mask]:
    """Returns one full-sized mask for each hole in each of the objects.

    Holes are defined as being completely enclosed by the object (i.e. not touching the canvas edge).
    """
    object = Mask.coerce(object)
    complement = ~object
    edge = mask_all(object)
    edge = edge & ~erode(edge, connectivity=connectivity)
    for obj in find_objects(complement):
        if (obj & edge).count():
            continue
        yield obj


def fill_holes(object: Mask, *, connectivity: Literal[4, 8] = 4) -> Mask:
    object = Mask.coerce(object)
    ret = object
    for hole in find_holes(object, connectivity=connectivity):
        ret = ret | hole
    return ret


def fill_gaps(
    objects: Mask,
    gap_size: int = 1,
    *,
    connectivity: Literal[4, 8] | None = None,
    object_connectivity: Literal[4, 8] | None = None,
    gap_connectivity: Literal[4, 8] | None = None,
) -> Mask:
    objects = Mask.coerce(objects)
    if connectivity is not None:
        if object_connectivity is not None:
            raise ValueError(
                "Either provide `connectivity` or `object_connectivity` but not both"
            )
        if gap_connectivity is not None:
            raise ValueError(
                "Either provide `connectivity` or `object_connectivity` but not both"
            )
        object_connectivity = connectivity
        gap_connectivity = connectivity
    if object_connectivity is None or gap_connectivity is None:
        raise ValueError(
            "Either provide `connectivity` or both a `object_connectivity` and a `gap_connectivity`"
        )
    labeled_array, num_features = ndimage.label(
        objects._mask, _structures[object_connectivity]
    )
    s = _structures[gap_connectivity]
    if gap_size > 1:
        s = ndimage.iterate_structure(s, gap_size)
    minlbl = ndimage.minimum_filter(
        np.where(objects._mask, labeled_array, num_features + 1), footprint=s
    )
    maxlbl = ndimage.maximum_filter(labeled_array, footprint=s)
    bridges = maxlbl > minlbl
    return objects | Mask(bridges)


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


def mask_color(canvas: AnyImage, color: Color | set[Color]) -> Mask:
    """Build a Mask from Color | set[Color]."""
    color = _set_of_colors(color)
    match canvas:
        case Image():
            mask = mask_all(canvas)
        case MaskedImage():
            mask = canvas._mask
    ret = mask_none(canvas)
    for c in color:
        m = canvas._data == _color2index[c]
        ret |= m
    return ret & mask


def mask_unpainted(canvas: AnyImage) -> Mask:
    match canvas:
        case MaskedImage():
            return Mask(~canvas._mask)
        case Image():
            return mask_none(canvas)
        case _:
            raise _make_type_error(canvas, "canvas", "AnyImage")


def mask_row(shape: ShapeSpec, row: int) -> Mask:
    ret = np.zeros(_shape_from_spec(shape), bool)
    ret[row, :] = True
    return Mask(ret)


def mask_col(shape: ShapeSpec, col: int) -> Mask:
    ret = np.zeros(_shape_from_spec(shape), bool)
    ret[:, col] = True
    return Mask(ret)


def correlate_masks(
    input: Mask, pattern: Mask, *, threshold: int | None = None
) -> Mask:
    """
    Essentially performs `scipy.ndimage.correlate(input,pattern,mode="constant",cval=0)>=threshold`
    """
    input = Mask.coerce(input)
    pattern = Mask.coerce(pattern)
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


def vec2dir4(vec: Vector) -> Dir4:
    vec = np.array(Vector.as_tuple(vec))
    d = np.argmax(abs(vec))
    d = np.sign(vec) * (np.r_[:2] == d)
    elementary = tuple(int(v) for v in d)
    return Vector._vec2dir[elementary]


def vec2dir8(vec: Vector) -> Dir8:
    vec = Vector.coerce(vec)
    lim = vec.length("euclidean") * math.sin(math.pi / 8)
    elementary = tuple(
        0 if abs(v) < lim else -1 if v < 0 else 1 for v in vec.as_tuple()
    )
    return Vector._vec2dir[elementary]


def vec2dir(vec: Vector, *, directions: Literal[4, 8] = 8) -> Dir4 | Dir8:
    match directions:
        case 4:
            return vec2dir4(vec)
        case 8:
            return vec2dir8(vec)
        case _:
            raise ValueError("`directions` must be 4 or 8")


def round2grid(coord: Coord) -> Coord:
    return Coord(*(int(round(v)) for v in coord.as_tuple()))


# -------------

_end = set(locals())

__all__ = sorted([k for k in _end - _start if not k.startswith("_")])
del _start, _end
