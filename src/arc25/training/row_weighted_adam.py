import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from optax._src import utils
from optax._src.base import GradientTransformationExtraArgs


class RSAdamState(NamedTuple):
    mu: jt.PyTree[jt.Array, "V ..."]
    nu: jt.PyTree[jt.Array, "V ..."]
    # the accumulated, weighted and decayed "steps"
    w1: jt.PyTree[jt.Array, " V"]
    w2: jt.PyTree[jt.Array, " V"]


def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""

    def apply(g, t):
        if g is None:
            return None
        d = decay[..., *(None,) * (g.ndim - decay.ndim)]
        return (1 - d).astype(g.dtype) * (g**order) + d.astype(t.dtype) * t

    return jax.tree.map(
        apply,
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


@functools.partial(jax.jit, inline=True)
def _bias_correction(moment, weight):
    """Performs bias correction. It becomes a no-op as count goes to infinity."""
    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    bc = 1 - weight

    def apply(t):
        c = bc.astype(t.dtype)[..., *(None,) * (t.ndim - bc.ndim)]
        return t / jnp.where(c > 0, c, 1)

    # Perform division in the original precision.
    return jax.tree.map(apply, moment)


def scale_by_adam_with_step_weights(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: chex.ArrayDType | None = None,
) -> GradientTransformationExtraArgs:
    r"""Rescale updates according to the Adam algorithm.

    However, this implementation allows individual weights to update at
    a different pace, by scaling both the weight update, as well as the
    momentum and variance updates with an individual "step weight".

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
      nesterov: Whether to use Nesterov momentum. The variant of Adam with
        Nesterov momentum is described in [Dozat 2016]

    Returns:
      A :class:`optax.GradientTransformation` object.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        V_cand = jax.tree.map(lambda a: a.shape[0] if a.ndim else None, params)

        def unique(*args):
            (val,) = set(args)
            return val

        V = jax.tree.reduce_associative(unique, V_cand)
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = optax.tree.zeros_like(params)  # Second moment
        w = jnp.ones((V,), jnp.float32)
        return RSAdamState(mu=mu, nu=nu, w1=w, w2=w)

    def update_fn(updates, state, params=None, *, row_weights, **extra_args):
        del params
        w = row_weights.astype(jnp.float32)
        f1 = 1 - (1 - b1) * w
        f2 = 1 - (1 - b2) * w
        mu = _update_moment(updates, state.mu, f1, 1)
        nu = _update_moment(updates, state.nu, f2, 2)
        w1 = state.w1 * f1
        w2 = state.w2 * f2
        mu_hat = _bias_correction(mu, w1)
        nu_hat = _bias_correction(nu, w2)
        updates = jax.tree.map(
            lambda m, v: (
                None
                if m is None
                else w.astype(m.dtype)[..., *(None,) * (m.ndim - w.ndim)]
                * m
                / (jnp.sqrt(v + eps_root) + eps)
            ),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return updates, RSAdamState(mu=mu, nu=nu, w1=w1, w2=w2)

    return GradientTransformationExtraArgs(init_fn, update_fn)
