"""Training infrastructure for ArcSolver (encoder-decoder for ARC tasks)."""

import contextlib
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
from .saving import save_model


@serialisable
@dataclass(frozen=True)
class ArcSolverConfig(ImageTrainConfigBase):
    """Configuration for ArcSolver training."""

    loss_focus: float = 0.1


_adaptive_lim = np.log([1e-2, 1e2]).astype(np.float32)
_adaptation_lim = np.log([1e-10, 1e10]).astype(np.float32)


class TrainState(TrainStateBase):
    """Training state for ArcSolver with encoder-decoder architecture.

    The encoder is frozen (stop_gradient applied), only decoder is trained.
    """

    @classmethod
    def make(
        cls,
        model: nnx.Module,
        config: ArcSolverConfig,
        *,
        rngs: nnx.Rngs,
    ) -> typing.Self:
        self = super().make(model, config, rngs=rngs)
        self.reference_entropy = nnx.Variable(jnp.array(500, jnp.float32))
        self.reference_entropy_weight = nnx.Variable(jnp.array(0, jnp.float32))
        return self

    @staticmethod
    def loss_fn(model, minibatch_dict, **kw):
        """Compute ArcSolver loss for a single minibatch.

        Args:
            model: The ARCSolver model to evaluate
            minibatch_dict: Dict with keys: inputs, input_sizes, outputs, output_masks,
                           latent_program_idx, cell_weight, image_weight
            **kw: Additional kwargs passed to model (e.g., mode, remat, unroll)

        Returns:
            Tuple of (loss, stats_dict)
        """
        inputs = minibatch_dict["inputs"]
        input_sizes = minibatch_dict["input_sizes"]
        outputs = minibatch_dict["outputs"]
        output_masks = minibatch_dict["output_masks"]
        latent_program_idx = minibatch_dict["latent_program_idx"]
        image_weights = minibatch_dict["image_weight"]
        cell_weights = minibatch_dict["cell_weight"]
        reference_entropy = minibatch_dict["reference_entropy"]
        loss_focus = minibatch_dict["loss_focus"]

        assert reference_entropy.size == 1
        assert loss_focus.size == 1

        # Decode to predict outputs
        logits = model(
            inputs,
            input_sizes,
            latent_program_idx=latent_program_idx,
            **kw,
        ).astype(jnp.float32)

        # Loss on ALL output cells (not masked like MAE)
        cell_crossentropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=outputs, axis=-1
        )

        # Mask to valid output regions and weight by pre-normalized cell weights
        pair_crossentropy = jnp.where(output_masks, cell_crossentropy, 0).sum(
            axis=(-2, -1)
        )
        rel_logprob = reference_entropy - pair_crossentropy
        lolim, hilim = _adaptive_lim
        loss_weight = jnp.exp(jnp.clip(loss_focus * rel_logprob, lolim, hilim))
        lolim, hilim = _adaptation_lim
        relprob = jnp.exp(jnp.clip(rel_logprob, lolim, hilim))
        if image_weights is not None:
            loss_weight = loss_weight * image_weights
            loss = loss_weight * pair_crossentropy
            relprob = relprob * image_weights
            pair_crossentropy = pair_crossentropy * image_weights
        else:
            loss = loss_weight * pair_crossentropy
        total_loss = loss.sum()
        total_loss_weight = loss_weight.sum()
        total_relprob = relprob.sum()
        total_pair_crossentropy = pair_crossentropy.sum()

        # Per-cell accuracy
        predictions = jnp.argmax(logits, axis=-1)
        cell_correct = (predictions == outputs) * output_masks * cell_weights
        total_cell_accuracy = cell_correct.sum()

        # Per-pair accuracy (all cells in output must be correct)
        pair_correct = (
            (
                # Padding doesn't count against accuracy
                (predictions == outputs)
                | ~output_masks
            )
            .all(axis=(-2, -1))
            .astype(jnp.float32)
        )
        total_pair_accuracy = (
            pair_correct * image_weights if image_weights is not None else pair_correct
        ).sum()

        return total_loss, dict(
            loss=total_loss,
            loss_weight=total_loss_weight,
            relprob=total_relprob,
            pair_crossentropy=total_pair_crossentropy,
            cell_accuracy=total_cell_accuracy,
            pair_accuracy=total_pair_accuracy,
        )

    @nnx.pmap(
        axis_name="data",
        in_axes=(
            nnx.StateAxes({...: None}),
            0,
            0,
            None,
            None,
        ),
        out_axes=None,
    )
    def _apply_update(self, grads, stats, total_weight, learning_rate):
        """Apply accumulated gradients (pmap'd, with pmean)."""
        print("Tracing _apply_update")

        grads = jax.lax.psum(grads, axis_name="data")
        stats = jax.lax.psum(stats, axis_name="data")

        loss_weight = stats.pop("loss_weight")
        loss = stats.pop("loss")

        lw = 1 / loss_weight
        grads = jax.tree.map(lambda a: a * jnp.array(lw, dtype=a.dtype), grads)
        sw = jnp.array(1 / total_weight, jnp.float32)
        stats = jax.tree.map(lambda a: a * sw, stats)
        relprob = stats.pop("relprob")
        stats["loss"] = loss * lw

        # Apply optimizer update
        self.optimizer.update(self.model, grads, learning_rate=learning_rate)

        # Apply EMA update to reference entropy
        beta = 0.02
        log_relprob = jnp.log(relprob)
        lolim, hilim = _adaptation_lim
        is_clipped = (log_relprob <= lolim + 0.05) | (log_relprob >= hilim - 0.05)
        decay = beta * self.reference_entropy_weight
        norm = self.reference_entropy_weight.value + beta - decay
        self.reference_entropy_weight.value += jnp.where(is_clipped, 0, beta) - decay
        # note relprob ~ -log(entropy) !
        delta = -(fac := beta / norm) * log_relprob
        self.reference_entropy.value += delta
        stats["reference_entropy"] = self.reference_entropy.value
        if False:
            stats["debug"] = dict(
                reference_entropy=self.reference_entropy.value,
                is_clipped=is_clipped,
                relprob=relprob,
                delta=delta,
                fac=fac,
                log_relprob=log_relprob,
                reference_entropy_weight=self.reference_entropy_weight.value,
            )

        return stats

    @staticmethod
    @nnx.jit(static_argnums=2)
    def _embed_inputs(model, minibatch_dict, kw):
        kw = dict(kw)
        shapes = ", ".join(
            f"{k}=({','.join(str(i) for i in v.shape)})"
            for k, v in minibatch_dict.items()
            if v is not None
        )
        kwstr = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"Tracing _embed_inputs for shape dict({shapes}) (kw=dict({kwstr}))")

        inputs = minibatch_dict["inputs"]
        input_sizes = minibatch_dict["input_sizes"]
        # Decode to predict outputs
        return model.encoder(
            inputs,
            input_sizes,
            **kw,
        )

    @staticmethod
    @nnx.jit(static_argnums=(3,))
    def _evaluate(model, minibatch_dict, params, kw):
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
        print(f"Tracing _evaluate for shape dict({shapes}) (kw=dict({kwstr}))")

        embeddings = minibatch_dict["embeddings"]
        outputs = minibatch_dict["outputs"]
        output_sizes = minibatch_dict["output_sizes"]
        output_masks = minibatch_dict["output_masks"]
        latent_program_idx = minibatch_dict["latent_program_idx"]
        image_weights = minibatch_dict["image_weight"]
        cell_weights = minibatch_dict["cell_weight"]
        reference_entropy = params["reference_entropy"]
        loss_focus = params["loss_focus"]

        # Decode to predict outputs
        logits = model.decode(
            embeddings,
            output_size=output_sizes,
            latent_program_idx=latent_program_idx,
            **kw,
        ).astype(jnp.float32)

        # Loss on ALL output cells (not masked like MAE)
        cell_crossentropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=outputs, axis=-1
        )

        # Mask to valid output regions and weight by pre-normalized cell weights
        pair_crossentropy = jnp.where(output_masks, cell_crossentropy, 0).sum(
            axis=(-2, -1)
        )
        rel_logprob = reference_entropy - pair_crossentropy
        lolim, hilim = _adaptive_lim
        loss_weight = jnp.exp(jnp.clip(loss_focus * rel_logprob, lolim, hilim))
        if image_weights is not None:
            loss_weight = loss_weight * image_weights
            loss = loss_weight * pair_crossentropy
            pair_crossentropy = pair_crossentropy * image_weights
        else:
            loss = loss_weight * pair_crossentropy

        # Per-cell accuracy
        predictions = jnp.argmax(logits, axis=-1)
        cell_correct = (predictions == outputs) * output_masks * cell_weights
        cell_accuracy = cell_correct.sum(axis=(-2, -1))

        # Per-pair accuracy (all cells in output must be correct)
        pair_correct = (
            (
                # Padding doesn't count against accuracy
                (predictions == outputs)
                | ~output_masks
            )
            .all(axis=(-2, -1))
            .astype(jnp.float32)
        )
        pair_accuracy = (
            jnp.where(pair_correct, image_weights, 0)
            if image_weights is not None
            else pair_correct
        )

        per_image_stats = jnp.stack(
            [
                loss,
                loss_weight,
                pair_crossentropy,
                cell_accuracy,
                pair_accuracy,
            ],
            axis=-1,
        )

        K = model.latent_program_embeddings.shape[0]
        per_class_stats = (
            jnp.zeros((K, 5), dtype=per_image_stats.dtype)
            .at[latent_program_idx]
            .add(per_image_stats)
        )

        return dict(
            loss=loss.sum(),
            loss_weight=loss_weight.sum(),
            pair_crossentropy=pair_crossentropy.sum(),
            cell_accuracy=cell_accuracy.sum(),
            pair_accuracy=pair_accuracy.sum(),
            per_class=dict(
                loss=per_class_stats[:, 0],
                loss_weight=per_class_stats[:, 1],
                pair_crossentropy=per_class_stats[:, 2],
                cell_accuracy=per_class_stats[:, 3],
                pair_accuracy=per_class_stats[:, 4],
            ),
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

    def prepare_batch(self, batch: BatchData) -> tuple[dict, ...]:
        """Prepare minibatches for ArcSolver training by looking up inputs.

        Args:
            minibatches: Tuple of MiniBatchData containing outputs

        Returns:
            Tuple of dicts with inputs, outputs, and metadata for training
        """
        prepared = []

        for output_mb in batch.minibatches:
            # Look up corresponding inputs from InputLookup
            input_mb = self.inputs_src.get_peer_batch(
                output_mb,
                target_shape=output_mb.images.shape[-2:],
                transpose="match",
            )

            # Create training dict
            mb_dict = self._prepare_output_minibatch(output_mb, params_separate=False)
            mb_dict.update(
                inputs=input_mb.images,
                input_sizes=input_mb.sizes,
            )

            prepared.append(mb_dict)

        return tuple(prepared)

    def _prepare_output_minibatch(
        self, output_mb: MiniBatchData, *, params_separate: bool
    ) -> dict:

        # Extract latent_program_idx from labels (challenge_id is labels[:, 0])
        latent_program_idx = output_mb.labels[..., 0].astype(np.int32)

        # Compute weights
        # output_mb.weight is per-example, pre-normalized across all minibatches
        weights = output_mb.weight
        output_masks = output_mb.masks

        # Compute per-cell weights (uniform across valid cells in each example)
        # Normalized so each example contributes proportionally to weights
        cell_weight_per_example = output_masks / jnp.maximum(
            output_masks.sum(axis=(-2, -1), keepdims=True), 1
        )
        cell_weight = (
            weights[..., None, None] * cell_weight_per_example
            if weights is not None
            else cell_weight_per_example
        )

        # Per-image weight (for per-pair accuracy)
        image_weight = weights

        # Create training dict
        ret = dict(
            outputs=output_mb.images,
            output_masks=output_masks,
            latent_program_idx=latent_program_idx,
            cell_weight=cell_weight,
            image_weight=image_weight,
        )

        params = dict(
            reference_entropy=self.train_state.reference_entropy.value,
            loss_focus=self.config.loss_focus * self._warmup_factor(),
        )

        if params_separate:
            return ret, params
        # params are scalars, but `TrainerBase` distributes
        # all attributes in minibatch_dict, we need to replicate here
        replication_shape = (self.num_devices,)
        ret.update(
            {
                k: jnp.tile(v, replication_shape).astype(jnp.float32)
                for k, v in params.items()
            }
        )

        return ret

    def _cache_embeddings(self):
        # setup buckets - borrow from the main bucketing

        challenge_order = self.inputs_src.challenges
        bucket_shapes = self.inputs_src.bucket_shapes

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
                lambda _image_area: 256,  # heuristic, constant, large batch size
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

                embeddings = self.train_state._embed_inputs(
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

                mb_dict, params = self._prepare_output_minibatch(
                    output_mb, params_separate=True
                )
                mb_dict.update(
                    embeddings=embeddings,
                    output_sizes=input_mb.sizes,
                )

                minibatches.append((mb_dict, params))
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

            for minibatch, params in eval_data.minibatches:
                minibatch = jax.tree.map(lambda a: reshard(a, "batch"), minibatch)
                params = jax.tree.map(lambda a: reshard(a), params)
                stats = self.train_state._evaluate(
                    resharded_model,
                    minibatch,
                    params,
                    tuple(
                        dict(
                            mode=self.config.mode,
                            remat=self.config.remat,
                            unroll=self.config.unroll,
                            deterministic=True,
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
        per_class = {
            k: np.asarray(v) / np.maximum(1, eval_data.per_class_total_weight)
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
        stats.update(
            class_accuracy_histogram=class_accuracy_histogram,
            per_class=per_class,
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
            embed_start = time.monotonic()
            self._cache_embeddings()
            embed_time = time.monotonic() - embed_start
            ret.update(eval_prep_time=embed_time)
            print(f"Embedding inputs for evaluation completed in {embed_time:.1f}s")
        else:
            embed_time = 0

        print(f"\n[Step {training_step}] Running evaluation...")
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
        )
        return self, res
