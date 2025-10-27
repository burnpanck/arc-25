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
    def Dict(*args, **kw) -> dict:
        return dict(*args, **kw)


try:
    set_metadata = nnx.set_metadata  # real thing (flax > 0.12)
except AttributeError:

    def set_metadata(
        node: Any,
        /,
        *,
        only: nnx.filterlib.Filter = nnx.Variable,
        **metadata: dict[str, Any],
    ) -> None:
        def _set_metadata(path, variable) -> None:
            del path  # unused
            if isinstance(variable, nnx.Variable):
                variable.set_metadata(**metadata)

        # inplace update of variable_state metadata
        nnx.map_state(_set_metadata, nnx.state(node, only))
