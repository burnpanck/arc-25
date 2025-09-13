import enum

import numpy as np


class SymOp(enum.Enum):
    """Represents the symmetry operations of the rectangular grid;

    Internally encoded as the combinations of three sequential mirror ops.
    The axis-aligned mirror axes commute, but the mirror axis along the main diagonal does not;
    therefore, order matters. The operations chosen (in order) are:
     1. Flip the x coordinates
     2. Flip the y coordinates
     3. Transpose/exchange x and y
    """

    e = 0b000
    x = 0b001  # flip the x-coordinates
    y = 0b010  # flip the y-coordinates
    i = 0b011
    t = 0b100  # exchange coordinates = flip over the main diagonal
    l = 0b101  # noqa
    r = 0b110  # clock-wise rotation when x->right, y->down (vision coordinate system)
    d = 0b111  # flip over the anti-diagonal

    @property
    def inverse(self):
        S = type(self)
        return {S.l: S.r, S.r: S.l}.get(self, self)

    def combine(self, rhs):
        """Apply `rhs` first, then `self`"""
        v = rhs.value
        s = self.value
        if v & 0b100 and bool(v & 0b001) != bool(v & 0b010):
            # we need to swap flip_x and flip_y
            v ^= 0b011
        v ^= s
        return type(self)(v)


def transform_image(s: SymOp, image: np.ndarray) -> np.ndarray:
    """Applies the provided symmetry operation to the first two dimensions."""
    transpose = bool(s.value & 0b100)
    flip_y = bool(s.value & 0b010)
    flip_x = bool(s.value & 0b001)
    img = image
    if flip_y:
        img = img[::-1, :]
    if flip_x:
        img = img[:, ::-1]
    if transpose:
        img = img.transpose(1, 0, *range(2, img.ndim))
    return img


def transform_vector(s: SymOp, vec: np.ndarray) -> np.ndarray:
    """

    `vec` has shape `(..., 2)`, with the last dimension representing `[row, col] = [y, x]`.
    """
    transpose = bool(s.value & 0b100)
    flip_y = bool(s.value & 0b010)
    flip_x = bool(s.value & 0b001)
    vec = np.where([flip_y, flip_x], -vec, vec)
    if transpose:
        vec = vec[..., ::-1]
    return vec


def transform_coordinate(s: SymOp, co: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """

    `co` and `shape` have shape `(..., 2)`, with the last dimension representing `[row, col] = [y, x]`.
    """
    transpose = bool(s.value & 0b100)
    flip_y = bool(s.value & 0b010)
    flip_x = bool(s.value & 0b001)
    s = shape - 1
    co = np.where([flip_y, flip_x], shape - 1 - co, co)
    if transpose:
        co = co[..., ::-1]
    return co
