from typing import TypeAlias

from . import primitives, types
from .primitives import *
from .types import *

__all__ = types.__all__ + primitives.__all__

for e in [Axis8, Axis4, Dir8, Dir4, Color, Transform]:
    for v in e:
        globals()[v.name] = v
        __all__.append(v.name)
