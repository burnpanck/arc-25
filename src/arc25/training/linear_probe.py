import time
from functools import partial
from typing import Callable, Literal, Self

import attrs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import scipy.optimize
from jax import lax
from jax.scipy.special import logsumexp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ..lib.attrs import AttrsModel

jax.config.update("jax_default_matmul_precision", "tensorfloat32")  # FP32/TF32
# jax.config.update("jax_enable_x64", True)  # if you really want fp64

axis = "data"


class ClassifierFitTarget(AttrsModel):
    features: jt.Float[jt.Array, " ... n d"]
    labels: jt.Float[jt.Int, " ... n"]


class LinearClassifier(AttrsModel):
    W: jt.Float[jt.Array, " d C"]
    b: jt.Float[jt.Array, " C"]

    def evaluate_examples(self, tgt: ClassifierFitTarget):
        logits = tgt.features @ self.W + self.b
        correct_logit = jnp.take_along_axis(logits, tgt.labels[..., :, None], axis=-1)
        correct_logit = correct_logit.reshape(correct_logit.shape[:-1])
        # not yet aggregated across examples
        loss = -correct_logit + logsumexp(logits, axis=-1)
        is_correct = jnp.argmax(logits, axis=-1) == tgt.labels
        return loss, is_correct

    def evaluate(self, tgt: ClassifierFitTarget):
        loss, is_correct = self.evaluate_examples(tgt)
        loss = loss.mean(axis=-1)
        accuracy = is_correct.mean(axis=-1)
        return loss, accuracy


class ShardedClsFitTarget(AttrsModel):
    """
    Note: Low-level methods relating to loss and it's derivative expect a `LinearClassifier` operating
    on standardised features. High-level methods (i.e. `fit`) on the other hand operates on unscaled classifiers.
    """

    shards: ClassifierFitTarget
    # normalisation
    mu: jt.Float[jt.Array, " d"]
    sigma: jt.Float[jt.Array, " d"]

    features: int = attrs.field(metadata=dict(static=True))
    classes: int = attrs.field(metadata=dict(static=True))

    mesh: Mesh = attrs.field(metadata=dict(static=True))

    def _loss_and_grad(
        self, params: LinearClassifier, tgt: ClassifierFitTarget, lam: float
    ):
        ce_fac = 1.0 / self.shards.labels.size
        l2_fac = 0.5 * lam * (tgt.labels.size / self.shards.labels.size)

        # partial losses, to be summed over batches
        def fun(p: LinearClassifier):
            ce_loss, _ = p.evaluate_examples(tgt)
            ce_loss = ce_fac * jnp.sum(ce_loss)
            l2_loss = l2_fac * jnp.sum(p.W**2)
            return ce_loss + l2_loss

        loss, grads = jax.value_and_grad(fun)(params)
        return loss, grads

    @jax.jit
    def loss_and_grad(self, params: LinearClassifier, lam: float):
        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(
                P(
                    axis,
                ),
            ),
            out_specs=(
                P(),
                P(
                    None,
                ),
            ),
        )
        def impl(tgt: ClassifierFitTarget):
            print(f"Tracing loss_and_grad: {tgt.features.shape=} {tgt.labels.shape=}")
            loss, grads = self._loss_and_grad(params, tgt, lam)
            # global sums across devices
            loss = lax.psum(loss, axis)
            grads = jax.tree.map(lambda x: lax.psum(x, axis), grads)
            return loss, grads

        return impl(self.shards)

    @jax.jit
    def loss(self, params: LinearClassifier, lam: float):
        return self.loss_and_grad(params, lam)[0]

    @jax.jit
    def loss_hessp(self, params: LinearClassifier, vec: LinearClassifier, lam: float):
        if True:
            # rely on [efficient-transpose](https://docs.jax.dev/en/latest/jep/17111-shmap-transpose.html)
            # to correctly move the `pmap`s around
            def grad_fun(p: LinearClassifier):
                print(f"Tracing grad_fun: {p.W.shape=} {p.b.shape=}")
                loss, grad = self.loss_and_grad(p, lam)
                return grad

            grad, hessp = jax.jvp(grad_fun, (params,), (vec,))
            return hessp
        else:
            # manual pmap outside of grad
            @jax.shard_map(
                mesh=self.mesh, in_specs=(P(axis),), out_specs=(P(None), P(None))
            )
            def impl(tgt: ClassifierFitTarget):
                def grad_fun(p):
                    loss, grad = self._loss_and_grad(p, tgt, lam)
                    return grad

                hessp = jax.jvp(grad_fun, (params,), (vec,))[1]
                # global sums across devices
                hessp = lax.psum(hessp, axis)
                return hessp

            return impl(self.shards)

    @classmethod
    def prepare(
        cls,
        tgt: ClassifierFitTarget,
        *,
        classes: int | None = None,
        devices: tuple[str, ...] | None = None,
    ) -> Self:
        if devices is None:
            devices = jax.local_devices()
        if classes is None:
            classes = 1 + int(tgt.labels.max())

        X = tgt.features
        mu = X.mean(0)
        sigma = X.std(0) + 1e-8

        norm_features = (X - mu) / sigma
        mesh = jax.sharding.Mesh(np.array(devices), (axis,))
        shards = jax.tree.map(
            lambda x: jax.device_put(
                x,
                jax.sharding.NamedSharding(
                    mesh,
                    P(
                        axis,
                    ),
                ),
            ),
            attrs.evolve(tgt, features=norm_features),
        )
        return cls(
            shards=shards,
            features=X.shape[-1],
            classes=classes,
            mu=mu,
            sigma=sigma,
            mesh=mesh,
        )

    def init_params(self):
        return LinearClassifier(
            W=np.zeros((self.features, self.classes), "f4"),
            b=np.zeros((self.classes,), "f4"),
        )

    def fit(
        self,
        start: LinearClassifier | None = None,
        lam: float = 1.0,
        *,
        method: Literal["trust-krylov", "trust-ncg"] = "trust-krylov",
        callback: Callable | None = None,
        **kw,
    ) -> LinearClassifier:
        if start is None:
            start = self.init_params()
        else:
            # standardise: X @ W + b=
            #   = (X-mu)/sigma @ Wp + bp
            #   = X/sigma @ Wp + bp - mu/sigma @ Wp
            #   = X @ (Wp / sigma[:,None]) + bp - mu @ (Wp / sigma[:,None])
            # -> Wp = W * sigma[:,None]
            # -> bp = b + mu @ W
            start = attrs.evolve(
                start,
                W=start.W * self.sigma[:, None],
                b=start.b + self.mu @ start.W,
            )

        n_steps = 0
        n_eval = 0
        n_hvp = 0
        last_loss = None

        if callback is not None:
            callback_wrap = lambda: callback(
                n_eval=n_eval,
                n_hvp=n_hvp,
                step=n_steps,
                loss=f"{last_loss:.2f}" if last_loss is not None else None,
            )

            def scipy_callback(p):
                nonlocal n_steps
                n_steps += 1
                callback_wrap()

        else:
            callback_wrap = lambda: None
            scipy_callback = None

        # flatten
        ravel_pytree = jax.flatten_util.ravel_pytree
        x0, unflatten = ravel_pytree(start)

        jax_dtype = jnp.float32
        opt_dtype = np.float32

        def fun_flat(x):
            nonlocal n_eval, last_loss
            # ts = time.monotonic()
            params = unflatten(jnp.asarray(x, jax_dtype))
            loss, grad = self.loss_and_grad(params, lam)
            loss = np.asarray(loss, dtype=opt_dtype)
            grad = np.asarray(ravel_pytree(grad)[0], dtype=opt_dtype)
            # dt = time.monotonic() -ts
            # print(f"fun_flat: {loss=:.2f} (took {1e3*dt:.0f} ms)")
            n_eval += 1
            last_loss = loss
            callback_wrap()
            return loss, grad

        def hessp_flat(x, p):
            nonlocal n_hvp
            params = unflatten(jnp.asarray(x, jax_dtype))
            vec = unflatten(jnp.asarray(p, jax_dtype))
            hv = self.loss_hessp(params, vec, lam)
            hv = np.asarray(ravel_pytree(hv)[0], dtype=opt_dtype)
            n_hvp += 1
            return hv

        res = scipy.optimize.minimize(
            fun=fun_flat,
            x0=np.asarray(x0, dtype=opt_dtype),
            jac=True,
            hessp=hessp_flat,
            method=method,
            callback=scipy_callback,
            **kw,
        )

        scipy_result = res
        res = unflatten(jnp.asarray(res.x, np.float32))

        # de-standardise
        W = res.W / self.sigma[:, None]
        res = attrs.evolve(
            res,
            W=W,
            b=res.b - self.mu @ W,
        )

        return res, scipy_result
