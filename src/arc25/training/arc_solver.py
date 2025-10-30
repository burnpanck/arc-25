"""Training infrastructure for ArcSolver (encoder-decoder for ARC tasks)."""

import contextlib
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from ..serialisation import serialisable
from ..vision2.arc_solver import ARCSolver
from ..vision2.fields import Field
from .base import TrainerBase, TrainStateBase
from .config import ImageTrainConfigBase, describe_config_json
from .dataset import (
    BatchData,
    BatchSpec,
    BucketedCollator,
    BucketedDataset,
    ImagesDataset,
    MiniBatchData,
    MinibatchSizeFunction,
    OnDemandBucketDataset,
)
from .learning_rate import scale_by_kwarg
from .row_weighted_adam import scale_by_adam_with_step_weights
from .saving import save_model


@serialisable
@dataclass(frozen=True)
class ArcSolverConfig(ImageTrainConfigBase):
    """Configuration for ArcSolver training."""

    # loss focus: weight example proportional to $\bar g^\alpha$
    # where $\bar g$ is the *average* cell crossentropy in the example.
    # Thus, easier examples are up-weighted in order not distract
    # us from the hopeless cases. $\alpha$ is dynamically adapted
    # as learning progresses such that on average $E((\alpha log g)^2) = loss_focus^2$.
    loss_focus: float = float(np.log(10))
    loss_focus_eps: float = 0.01
    loss_focus_limit: float = float(np.log(10))
    loss_focus_beta: float = 0.95

    num_solution_attempts: int = 1


class TrainState(TrainStateBase):
    """Training state for ArcSolver with encoder-decoder architecture.

    The encoder is frozen (stop_gradient applied), only decoder is trained.
    """

    train_filter: typing.ClassVar = nnx.filterlib.All(
        nnx.Param, nnx.filterlib.Not("encoder")
    )

    @classmethod
    def _make_optimiser(cls, config: ImageTrainConfigBase, trainable_state: nnx.State):
        embeddings = {"latent_program_embeddings"}
        assert not embeddings - set(
            trainable_state.keys()
        ), "Embeddings seem to be missing"
        main_opt_chain = optax.chain(*super()._make_optimiser(config, trainable_state))
        # we currently don't do any weight-decay. If we did, we'd need to
        # also scale it with the weight, perhaps easiest to just include in this custom Adam.
        embedding_chain = scale_by_adam_with_step_weights(
            b1=config.betas[0], b2=config.betas[1], eps=config.eps
        )
        partition_labels = nnx.State(
            {
                k: "embedding" if k in embeddings else "main"
                for k in trainable_state.keys()
            }
        )
        ret = optax.partition(
            dict(main=main_opt_chain, embedding=embedding_chain),
            partition_labels,
        )
        return (ret,)

    @classmethod
    def make(
        cls,
        model: nnx.Module,
        config: ArcSolverConfig,
        *,
        rngs: nnx.Rngs,
    ) -> typing.Self:
        self = super().make(model, config, rngs=rngs)
        self.reference_entropy = nnx.Variable(jnp.array(np.log(10), jnp.float32))
        self.reference_entropy_var = nnx.Variable(
            jnp.array(np.log(10) ** 2, jnp.float32)
        )
        self.reference_entropy_weight = nnx.Variable(jnp.array(0, jnp.float32))
        return self

    @classmethod
    def _loss_fn_impl(cls, logits, minibatch_dict, params, **kw):
        """Compute ArcSolver loss for a single minibatch.

        Args:
            model: The ARCSolver model to evaluate
            minibatch_dict: Dict with keys: inputs, input_sizes, outputs, output_masks,
                           latent_program_idx, cell_weight, image_weight
            **kw: Additional kwargs passed to model (e.g., mode, remat, unroll)

        Returns:
            Tuple of (loss, stats_dict)
        """
        outputs = minibatch_dict["outputs"]
        output_masks = minibatch_dict["output_masks"]
        example_weight = minibatch_dict["example_weight"]
        cell_weights = minibatch_dict["cell_weight"]

        reference_entropy = params["reference_entropy"]
        loss_weight_scale = params["loss_weight_scale"]
        loss_weight_scale_limit = params["loss_weight_scale_limit"]

        # Loss on ALL output cells (not masked like MAE)
        cell_crossentropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=outputs, axis=-1
        )

        # Mask to valid output regions and weight by pre-normalized cell weights
        pair_crossentropy = jnp.where(
            output_masks, cell_crossentropy * cell_weights, 0
        ).sum(axis=(-2, -1))

        # Focal loss: focus on those examples where we have a reasonable shot at
        # getting them right.
        rel_logprob = reference_entropy - jax.lax.stop_gradient(pair_crossentropy)
        loss_weight = jnp.exp(
            jnp.clip(
                loss_weight_scale * rel_logprob,
                -loss_weight_scale_limit,
                loss_weight_scale_limit,
            )
        )
        loss = loss_weight * pair_crossentropy
        relprob = jnp.exp(
            jnp.clip(rel_logprob, -loss_weight_scale_limit, loss_weight_scale_limit)
        )
        rel_logprob_squared = jnp.square(rel_logprob)

        # Per-cell accuracy
        predictions = jnp.argmax(logits, axis=-1)
        cell_correct = predictions == outputs
        cell_accuracy = (
            jnp.where(cell_correct & output_masks, cell_weights, 0)
            .astype(jnp.float32)
            .sum(axis=(-2, -1))
        )

        # Per-pair accuracy (all cells in output must be correct)
        pair_accuracy = (
            (
                # Padding doesn't count against accuracy
                cell_correct
                | ~output_masks
            )
            .all(axis=(-2, -1))
            .astype(jnp.float32)
        )

        res = dict(
            loss=loss,
            loss_weight=loss_weight,
            relprob=relprob,
            rel_logprob_squared=rel_logprob_squared,
            pair_crossentropy=pair_crossentropy,
            cell_accuracy=cell_accuracy,
            pair_accuracy=pair_accuracy,
        )

        if example_weight is not None:
            res = {k: v * example_weight for k, v in res.items()}

        return res

    @classmethod
    def loss_fn(cls, model, minibatch_dict, params, **kw):
        inputs = minibatch_dict["inputs"]
        input_sizes = minibatch_dict["input_sizes"]
        latent_program_idx = minibatch_dict["latent_program_idx"]

        # Decode to predict outputs
        logits = model(
            inputs,
            input_sizes,
            latent_program_idx=latent_program_idx,
            **kw,
        ).astype(jnp.float32)

        res = cls._loss_fn_impl(logits, minibatch_dict, params, **kw)

        weight = res["loss_weight"]
        K = model.latent_program_embeddings.shape[0]
        per_class_weight = (
            jnp.zeros(K, dtype=weight.dtype).at[latent_program_idx].add(weight)
        )

        res = {k: v.sum() for k, v in res.items()}
        loss = res["loss"]
        stats = res
        stats["class_weight"] = per_class_weight
        return loss, stats

    @nnx.pmap(
        axis_name="data",
        in_axes=(
            nnx.StateAxes({...: None}),
            0,
            0,
            None,
        ),
        out_axes=None,
    )
    def _apply_update(self, grads, stats, params):
        """Apply accumulated gradients (pmap'd, with pmean)."""
        print("Tracing _apply_update")
        sys.stdout.flush()

        total_weight = params["total_weight"]
        learning_rate = params["learning_rate"]
        loss_focus_beta = params["loss_focus_beta"]

        grads = jax.lax.psum(grads, axis_name="data")
        stats = jax.lax.psum(stats, axis_name="data")

        loss_weight = stats.pop("loss_weight")
        loss = stats.pop("loss")
        class_weight = stats.pop("class_weight")

        lw = jnp.array(1 / loss_weight, jnp.float32)

        grads = jax.tree.map(lambda a: a * jnp.array(lw, dtype=a.dtype), grads)
        sw = jnp.array(1 / total_weight, jnp.float32)
        stats = jax.tree.map(lambda a: a * sw, stats)
        relprob = stats.pop("relprob")
        rel_logprob_squared = stats.pop("rel_logprob_squared")
        stats["loss"] = loss * lw

        class_weight = class_weight * (class_weight.size * lw)
        # class weight should now be scaled to mean 1 again

        # normalise embedding gradients - will be undone by the row-adaptive optimiser afterwards.
        grads["latent_program_embeddings"] /= jnp.where(
            class_weight > 0, class_weight, 1
        )[:, None, None]

        # limit embedding update weights to a reasonable value
        class_weight = jnp.minimum(class_weight, 3)

        # Apply optimizer update
        self.optimizer.update(
            self.model, grads, learning_rate=learning_rate, row_weights=class_weight
        )

        # Apply EMA update to reference entropy
        beta = loss_focus_beta
        bb = 1 - beta
        self.reference_entropy_weight.value += bb * (1 - self.reference_entropy_weight)
        fac = bb / self.reference_entropy_weight
        # note log(relprob) ~ -entropy !
        # re' = -log( (1-f)*exp(-re) + f*relprob*exp(-re) )
        #     = -log( (1-f)          + f*relprob          ) + re
        #     = -log(  1             + f*(relprob-1)      ) + re
        self.reference_entropy.value -= jnp.log1p(fac * (relprob - 1))
        # rev = (1-f)*rev + f*rlps = rev + f*(rlps-rev)
        self.reference_entropy_var.value += fac * (
            rel_logprob_squared - self.reference_entropy_var
        )

        stats["reference_entropy"] = self.reference_entropy.value
        if True:
            stats["debug"] = dict(
                # reference_entropy=self.reference_entropy.value,
                reference_entropy_var=self.reference_entropy_var.value,
                # reference_entropy_weight=self.reference_entropy_weight.value,
                # fac=fac,
                relprob=relprob,
                # log_relprob=jnp.log(relprob),
                rel_logprob_squared=rel_logprob_squared,
                loss_weight_scale=params["loss_weight_scale"],
            )

        # compute weight metrics/stats; for now just propgram embedding norms
        embedding_norms = jnp.linalg.norm(
            self.model.latent_program_embeddings, axis=-1
        ).mean(-1)
        stats["program_embedding_norms"] = {
            k: getattr(embedding_norms, k)() for k in ["mean", "std", "min", "max"]
        }

        return stats

    @classmethod
    @nnx.jit(static_argnums=(0, 3))
    def embed_inputs(cls, model, minibatch_dict, kw):
        kw = dict(kw)
        shapes = ", ".join(
            f"{k}=({','.join(str(i) for i in v.shape)})"
            for k, v in minibatch_dict.items()
            if v is not None
        )
        kwstr = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"Tracing embed_inputs for shape dict({shapes}) (kw=dict({kwstr}))")
        sys.stdout.flush()

        inputs = minibatch_dict["inputs"]
        input_sizes = minibatch_dict["input_sizes"]
        # Decode to predict outputs
        return model.encoder(
            inputs,
            input_sizes,
            **kw,
        )

    @classmethod
    @nnx.jit(static_argnums=(0, 4))
    def evaluate(cls, model, minibatch_dict, params, kw):
        kw = dict(kw)
        shapes = ", ".join(
            (
                f"{k}=({','.join(str(i) for i in v.shape)})"
                if not isinstance(v, Field)
                else f"{k}={v.batch_shape}"
            )
            for k, v in minibatch_dict.items()
            if v is not None
        )
        kwstr = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"Tracing evaluate for shape dict({shapes}) (kw=dict({kwstr}))")
        sys.stdout.flush()

        nsa = kw.pop("num_solution_attempts")

        mbd = jax.tree.map(
            lambda a: jnp.tile(a[:, None, ...], (1, nsa) + (1,) * (a.ndim - 1)),
            minibatch_dict,
        )

        embeddings = mbd["embeddings"]
        output_sizes = mbd["output_sizes"]
        latent_program_idx = mbd["latent_program_idx"] * nsa
        latent_program_idx += np.arange(nsa)[
            None, :, *(None,) * (latent_program_idx.ndim - 2)
        ]

        # Decode to predict outputs
        logits = model.decode(
            embeddings,
            output_size=output_sizes,
            latent_program_idx=latent_program_idx,
            **kw,
        ).astype(jnp.float32)
        res = cls._loss_fn_impl(logits, mbd, params, **kw)

        stats = {
            k: v for k, v in res.items() if k not in {"relprob", "rel_logprob_squared"}
        }

        per_example_stats = jnp.stack(list(stats.values()), axis=-1)

        K = model.latent_program_embeddings.shape[0]
        per_class_stats = (
            jnp.zeros((K, len(stats)), dtype=per_example_stats.dtype)
            .at[latent_program_idx]
            .add(per_example_stats)
        ).reshape(-1, nsa, len(stats))

        return {k: v.sum() for k, v in stats.items()} | dict(
            per_class={k: per_class_stats[..., i] for i, k in enumerate(stats)},
        )


@attrs.mutable(kw_only=True)
class ArcSolverTrainer(TrainerBase):
    """Manages the ArcSolver training pipeline."""

    # ArcSolver-specific: input lookup for matching outputs to inputs
    inputs_src: OnDemandBucketDataset = attrs.field(on_setattr=attrs.setters.frozen)

    eval_dataset: ImagesDataset | None = attrs.field(
        default=None, on_setattr=attrs.setters.frozen
    )
    # minibatches of evaluation data, already with pre-computed input embeddings
    _eval_data_cache: tuple[dict, ...] | None = None

    @classmethod
    def make(
        cls,
        config: ArcSolverConfig,
        model: ARCSolver,
        collator: BucketedCollator,
        *,
        inputs_src: OnDemandBucketDataset,
        num_devices: int = 1,
        lr_schedule: typing.Callable | None = None,
        rngs: nnx.Rngs,
        **kw,
    ) -> typing.Self:
        """Create an ArcSolverTrainer instance."""
        # Create training state
        train_state = TrainState.make(model, config, rngs=rngs)

        # Calculate total steps using base class helper
        total_steps = cls._calculate_total_steps(config, collator)

        # Create learning rate schedule using base class helper
        if lr_schedule is None:
            lr_schedule = cls._make_lr_schedule(config, total_steps)

        return cls(
            config=config,
            train_state=train_state,
            collator=collator,
            inputs_src=inputs_src,
            lr_schedule=lr_schedule,
            num_devices=num_devices,
            total_steps=total_steps,
            **kw,
        )

    def prepare_params(self) -> dict:
        re = self.train_state.reference_entropy.value
        rev = self.train_state.reference_entropy_var.value
        loss_weight_scale = self.config.loss_focus / jnp.sqrt(
            jnp.maximum(0, rev) + self.config.loss_focus_eps
        )
        return dict(
            reference_entropy=re,
            loss_weight_scale=loss_weight_scale * self._warmup_factor(),
            loss_weight_scale_limit=self.config.loss_focus_limit,
            loss_focus_beta=self.config.loss_focus_beta,
        )

    def prepare_batch(self, batch: BatchData) -> tuple[tuple[dict, ...], dict]:
        """Prepare minibatches for ArcSolver training by looking up inputs.

        Args:
            minibatches: Tuple of MiniBatchData containing outputs

        Returns:
            Tuple of dicts with inputs, outputs, and metadata for training
        """
        prepared = []

        n_sols = self.config.num_solution_attempts
        rngs = self.train_state.rngs

        for output_mb in batch.minibatches:
            # Look up corresponding inputs from InputLookup
            input_mb = self.inputs_src.get_peer_batch(
                output_mb,
                target_shape=output_mb.images.shape[-2:],
                transpose="match",
            )

            # Create training dict
            mb_dict = self._prepare_output_minibatch(output_mb)
            base_pgm_idx = mb_dict.pop("latent_program_idx")
            pgm_idx = base_pgm_idx * n_sols + jax.random.randint(
                rngs.data(), minval=0, maxval=n_sols, shape=base_pgm_idx.shape
            )
            mb_dict.update(
                latent_program_idx=pgm_idx,
                inputs=input_mb.images,
                input_sizes=input_mb.sizes,
            )

            prepared.append(mb_dict)

        params = self.prepare_params()

        return tuple(prepared), params

    def _prepare_output_minibatch(self, output_mb: MiniBatchData) -> dict:

        # Extract latent_program_idx from labels (challenge_id is labels[:, 0])
        latent_program_idx = output_mb.labels[..., 0].astype(np.int32)

        output_masks = output_mb.masks

        # Compute weights
        # output_mb.weight is pre-computed from the image size
        weights = output_mb.weight

        # Compute cell weights; normalised per example.
        # Here, we apply uniform weight to all cells.
        cell_weight = output_masks / np.maximum(
            output_masks.sum(axis=(-2, -1), keepdims=True), 1
        )

        # Example prior weight
        example_weight = weights

        # Create training dict
        return dict(
            outputs=output_mb.images,
            output_masks=output_masks,
            latent_program_idx=latent_program_idx,
            cell_weight=cell_weight,
            example_weight=example_weight,
        )

    def _cache_embeddings(self):
        # setup buckets - borrow from the main bucketing

        challenge_order = self.inputs_src.challenges
        bucket_shapes = self.inputs_src.bucket_shapes

        eval_batch_size = self.config.eval_batch_size
        if eval_batch_size is None:
            eval_batch_size = 8 * min(
                b.minibatch_size for b in self.collator.buckets.values()
            )

        input_ds, output_ds = self.eval_dataset.split_input_output()

        (eval_ds,) = [
            BucketedDataset.make(
                ds,
                bucket_shapes,
                challenges=challenge_order,
            )
            for ds in [output_ds]
        ]

        input_src = OnDemandBucketDataset(
            input_ds,
            bucket_shapes=bucket_shapes,
            challenges=challenge_order,
            weight_fun=lambda area: None,
        )

        mesh = jax.make_mesh(
            (self.num_devices,),
            ("batch",),
            axis_types=(jax.sharding.AxisType.Auto,),
        )

        model_graph, model_state = nnx.split(self.train_state.model)
        model_state = jax.tree.map(
            lambda a: jax.device_put(
                a,
                jax.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            ),
            model_state,
        )
        resharded_model = nnx.merge(model_graph, model_state)

        total_weight = 0
        per_class_total_weight = 0
        minibatches = []

        num_devices = self.num_devices
        rgen = np.random.default_rng(self.config.seed)

        with jax.set_mesh(mesh):
            for bucket_shape, _batch_idx, output_mb in eval_ds.all_data_in_batches(
                lambda _image_area: eval_batch_size,  # heuristic, constant, large batch size
                num_devices=num_devices,
                rgen=rgen,
                with_progress=self.with_progress_bars,
            ):
                input_mb = input_src.get_peer_batch(
                    output_mb, target_shape=bucket_shape, transpose="match"
                )

                # shard input data onto devices
                input_data = jax.tree.map(
                    lambda a: jax.device_put(
                        a,
                        jax.NamedSharding(
                            mesh,
                            jax.sharding.PartitionSpec("batch", *[None] * (a.ndim - 1)),
                        ),
                    ),
                    dict(inputs=input_mb.images, input_sizes=input_mb.sizes),
                )

                embeddings = self.train_state.embed_inputs(
                    resharded_model,
                    input_data,
                    tuple(
                        dict(
                            mode=self.config.mode,
                            remat=self.config.remat,
                            unroll=self.config.unroll,
                            deterministic=True,
                        ).items()
                    ),
                )
                # reap embeddings back from the devices
                embeddings = jax.tree.map(lambda a: jax.device_get(a), embeddings)

                mb_dict = self._prepare_output_minibatch(output_mb)
                mb_dict.update(
                    embeddings=embeddings,
                    output_sizes=input_mb.sizes,
                )

                minibatches.append(mb_dict)
                total_weight += (
                    output_mb.weight
                    if output_mb.weight is not None
                    else output_mb.n_examples
                )
                per_class_total_weight = (
                    np.bincount(
                        output_mb.labels[..., 0].ravel(),
                        minlength=self.train_state.model.latent_program_embeddings.shape[
                            0
                        ],
                        weights=(
                            output_mb.weight.ravel()
                            if output_mb.weight is not None
                            else None
                        ),
                    )
                    + per_class_total_weight
                )

        self._eval_data_cache = SimpleNamespace(
            minibatches=minibatches,
            total_weight=total_weight,
            per_class_total_weight=per_class_total_weight,
        )

    def _evaluate(self):
        if self._eval_data_cache is None:
            self._cache_embeddings()

        eval_data = self._eval_data_cache

        params = self.prepare_params()

        mesh = jax.make_mesh(
            (self.num_devices,),
            ("batch",),
            axis_types=(jax.sharding.AxisType.Auto,),
        )

        def reshard(a, *args):
            return jax.device_put(
                a,
                jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args)),
            )

        model_graph, model_state = nnx.split(self.train_state.model)
        model_state = jax.tree.map(lambda a: reshard(a), model_state)
        resharded_model = nnx.merge(model_graph, model_state)

        params = jax.tree.map(lambda a: reshard(a), params)

        res = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(jax.set_mesh(mesh))
            if self.with_progress_bars:
                import tqdm.auto

                pbar = stack.enter_context(
                    tqdm.auto.tqdm(total=len(eval_data.minibatches), leave=False)
                )
            else:
                pbar = None

            for minibatch in eval_data.minibatches:
                stats = self.train_state.evaluate(
                    resharded_model,
                    minibatch,
                    params,
                    tuple(
                        dict(
                            mode=self.config.mode,
                            remat=self.config.remat,
                            unroll=self.config.unroll,
                            deterministic=True,
                            num_solution_attempts=self.config.num_solution_attempts,
                        ).items()
                    ),
                )
                if res is None:
                    res = stats
                else:
                    res = jax.tree.map(lambda a, b: a + b, res, stats)

                if pbar is not None:
                    pbar.update()

        per_class = res.pop("per_class")
        per_class_loss_weight = np.asarray(per_class.pop("loss_weight"))
        per_class_loss = np.where(
            per_class_loss_weight > 0,
            np.asarray(per_class.pop("loss")) / per_class_loss_weight,
            np.nan,
        )
        shape = per_class_loss.shape
        per_class = {
            k: np.asarray(v)
            / np.maximum(1, eval_data.per_class_total_weight).reshape(*shape)
            for k, v in per_class.items()
        }
        per_class["loss"] = per_class_loss
        loss = res.pop("loss") / res.pop("loss_weight")
        stats = {k: float(v) / max(1, eval_data.total_weight) for k, v in res.items()}
        stats["loss"] = loss
        per_class_accuracy = per_class["pair_accuracy"]
        class_accuracy_histogram, _ = np.histogram(
            per_class_accuracy,
            bins=10,
            range=(0, 1),
        )
        best_per_class_accuracy = per_class_accuracy.max(1)
        mean_per_class_accuracy = per_class_accuracy.mean(1)
        std_per_class_accuracy = per_class_accuracy.std(1)
        stats.update(
            class_accuracy_histogram=class_accuracy_histogram,
            per_class=per_class,
            best_per_class_accuracy=best_per_class_accuracy.mean(),
            mean_per_class_accuracy=mean_per_class_accuracy.mean(),
            std_per_class_accuracy=std_per_class_accuracy.std(),
        )
        return stats

    def periodic_evaluation(self, stats: dict) -> tuple[dict, float] | None:
        """Run evaluation periodically."""
        if self.eval_dataset is None:
            return None

        training_step = stats["training_step"]

        ret = dict()

        if self._eval_data_cache is None:
            print(
                f"\n[Step {training_step}] Preparing input embeddings for evaluation..."
            )
            sys.stdout.flush()
            embed_start = time.monotonic()
            self._cache_embeddings()
            embed_time = time.monotonic() - embed_start
            ret.update(eval_prep_time=embed_time)
            print(f"Embedding inputs for evaluation completed in {embed_time:.1f}s")
        else:
            embed_time = 0

        print(f"\n[Step {training_step}] Running evaluation...")
        sys.stdout.flush()
        eval_start = time.monotonic()
        eval_results = self._evaluate()
        eval_results.pop("per_class")  # not suitable for wandb
        eval_time = time.monotonic() - eval_start
        ret.update(eval=eval_results, eval_time=eval_time)

        # Print results
        print(
            f"Evaluation completed in {eval_time:.1f}s: "
            + " ".join(
                f"{kk}: "
                + (
                    f"[{','.join(f'{v:.0f}' for k,v in enumerate(vv))}]"
                    if isinstance(vv, np.ndarray)
                    else f"{vv:.3f}"
                )
                for kk, vv in eval_results.items()
            )
        )
        sys.stdout.flush()

        # Return results for wandb logging
        return ret, eval_time + embed_time

    @classmethod
    def main(
        cls,
        config: ArcSolverConfig,
        model: ARCSolver,
        dataset: ImagesDataset,
        *,
        bucket_shapes: typing.Iterable[tuple[int, int]] = ((30, 30),),
        eval_dataset: ImagesDataset | None = None,
        checkpoint_dir: Path | str | None = None,
        lr_schedule: typing.Callable | None = None,
        wandb_project: str | None = None,
        run_name: str | None = None,
        num_devices: int | None = None,
        run_metadata: dict | None = None,
        **kw,
    ):
        # Detect available devices
        if num_devices is None:
            num_devices = jax.local_device_count()

        challenges = dataset.challenges
        if eval_dataset is not None:
            challenges |= eval_dataset.challenges

        challenge_order = tuple(sorted(challenges))
        bucket_shapes = tuple(
            sorted(bucket_shapes, key=lambda sh: (sh[0] * sh[1], abs(sh[0] - sh[1])))
        )

        input_ds, output_ds = dataset.split_input_output()

        (training_ds,) = [
            BucketedDataset.make(
                ds,
                bucket_shapes,
                challenges=challenge_order,
            )
            for ds in [output_ds]
        ]

        # Create collator with proper seed and granularity
        minibatch_size_fn = MinibatchSizeFunction(
            reference_minibatch_size=config.minibatch_size,
            reference_image_size=config.reference_image_size,
            base_cost=config.base_cell_cost,
            granularity=num_devices,  # Ensure divisibility for pmap
        )

        batch_spec = BatchSpec(
            target_batch_weight=config.batch_size,
            reference_image_size=config.reference_image_size,
            area_weight_exponent=None,
        )

        collator = BucketedCollator.make(
            dataset=training_ds,
            batch_spec=batch_spec,
            minibatch_size=minibatch_size_fn,
            seed=config.seed,  # Tie dataset seed to training seed
        )

        input_src = OnDemandBucketDataset(
            input_ds,
            bucket_shapes=bucket_shapes,
            challenges=challenge_order,
            weight_fun=lambda area: None,
        )

        # Initialize trainer
        self = cls.make(
            config=config,
            model=model,
            collator=collator,
            inputs_src=input_src,
            num_devices=num_devices,
            rngs=nnx.Rngs(config.seed),
            lr_schedule=lr_schedule,
            eval_dataset=eval_dataset,
            **kw,
        )

        # Run common training loop
        res = self.run_main(
            checkpoint_dir=checkpoint_dir,
            wandb_project=wandb_project,
            run_name=run_name,
            run_metadata=run_metadata,
        )
        return self, res
