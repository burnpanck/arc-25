from typing import Any

from flax import nnx

try:
    data = nnx.data  # real thing (flax ≥ 0.11)
    static = nnx.static  # real thing (flax ≥ 0.11)
except AttributeError:
    # Minimal polyfill: returns the value unchanged
    def data(x: Any) -> Any:
        return x

    # Minimal polyfill: returns the value unchanged
    def static(x: Any) -> Any:
        return x


try:
    Dict = nnx.Dict  # real thing (flax ≥ 0.11)
except AttributeError:
    # Minimal polyfill: returns the value unchanged
    def Dict(x: dict) -> dict:
        return x
