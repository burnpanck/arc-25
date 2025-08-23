from typing import TypeAlias

from .types import (
    Axis4,
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
)

Coord: TypeAlias = _Coord | tuple[int, int]

for e in [Axis8, Axis4, Dir8, Dir4, Color]:
    for v in e:
        globals()[v.name] = v
