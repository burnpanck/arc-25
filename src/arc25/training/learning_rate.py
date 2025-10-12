import jax
import jax.numpy as jnp
import optax


def scale_by_kwarg() -> optax.GradientTransformationExtraArgs:
    """Scale updates using a custom schedule for the `step_size`.

    Args:
      step_size_fn: A function that takes an update count as input and proposes
        the step_size to multiply the updates by.

    Returns:
      A :class:`optax.GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, **extra_args):
        del params
        lr = extra_args["learning_rate"]
        scale = -lr
        updates = jax.tree.map(lambda g: jnp.array(scale, dtype=g.dtype) * g, updates)
        return updates, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
