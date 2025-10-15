import datetime
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
from ..vision2.mae import MaskedAutoencoder
from .config import ImageTrainConfigBase
from .dataset import (
    BatchData,
    BatchSpec,
    BucketedCollator,
    BucketedDataset,
    MiniBatchData,
    MinibatchSizeFunction,
)
from .knn_eval import KNNEvaluator
from .learning_rate import scale_by_kwarg
from .saving import save_model


@serialisable
@dataclass(frozen=True)
class MAETaskConfig(ImageTrainConfigBase):
    """Configuration for the training script."""

    test_ratio: float = 0.4
    # fraction of test cells that are *not* masked
    nonmask_fraction: float = 0.2
    # fraction of non-masked cells that are randomised
    randomise_fraction: float = 0.5

    # inline validation in ref_batch/batch_size optimizer steps
    knn_validation_every_ref_batch: float = 32


class TrainState(nnx.Module):
    """Holds model and optimizer state, performs gradient accumulation."""

    model: nnx.Module
    optimizer: nnx.Optimizer

    @classmethod
    def make(
        cls,
        model: nnx.Module,
        config: MAETaskConfig,
        *,
        rngs: nnx.Rngs,
    ) -> typing.Self:
        """Initialize training state with model and optimizer."""
        # Create the AdamW optimizer with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            optax.scale_by_adam(b1=config.betas[0], b2=config.betas[1], eps=config.eps),
            optax.add_decayed_weights(config.weight_decay),
            scale_by_kwarg(),
        )

        self = cls()
        self.model = model
        self.optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return self

    def model_stats(self):
        stats = {}
        for k, v in dict(
            total=self, model=self.model, optimizer=self.optimizer
        ).items():
            _, params = nnx.split(v, nnx.Param)
            leaves = jax.tree_util.tree_leaves(params)
            total_params = sum((leaf.size for leaf in leaves), start=0)
            total_bytes = sum((leaf.nbytes for leaf in leaves), start=0)
            stats[k] = SimpleNamespace(params=total_params, bytes=total_bytes)

        return SimpleNamespace(**stats)

    @staticmethod
    def loss_fn(model, minibatch_dict, **kw):
        """Compute MAE loss for a single minibatch.

        Args:
            model: The model to evaluate
            minibatch_dict: Dict with keys: images, masks, sizes, labels, transpose, weight,
                           input_mask, prediction_mask
            **kw: Additional kwargs passed to model (e.g., mode)
        """
        images = minibatch_dict["images"]
        sizes = minibatch_dict["sizes"]
        weights = minibatch_dict["weight"]
        input_mask = minibatch_dict["input_mask"]
        prediction_mask = minibatch_dict["prediction_mask"]

        # Forward pass
        logits = model(images, sizes, mask=input_mask, **kw)

        # Compute loss only on masked cells
        per_cell_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=images, axis=-1
        )

        # Weight by example weight and normalize
        w = weights
        loss = (jnp.where(prediction_mask, per_cell_loss, 0) * w).sum()

        # Compute accuracy
        accuracy = ((jnp.argmax(logits, axis=-1) == images) * w).sum()

        return loss, dict(loss=loss, accuracy=accuracy)

    def batch_stats(
        self, batch_size: int | None = None, shape: tuple[int, int] = (30, 30)
    ):
        graph, state = nnx.split(self)

        def inference(state, batch):
            state = nnx.merge(graph, state)
            model = state.model
            model.eval()
            return model(
                batch["image"],
                batch["size"],
                mask=batch["input_mask"],
                mode=self.config.mode,
            )

        def train(state, batch):
            state = nnx.merge(graph, state)
            model = state.model
            model.train()

            def loss_fn(model):
                return self.loss_fn(model, batch, mode=self.config.mode)

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (_, stats), grads = grad_fn(model)

            # Update optimizer state and compute new parameters
            state.optimizer.update(state.model, grads, learning_rate=1.0)
            _, state = nnx.split(state)
            return state, stats

        if batch_size is None:
            batch_size = self.config.global_batch_size
        batch = dict(
            images=np.zeros((batch_size,) + shape, "i1"),
            sizes=np.tile(shape, (batch_size, 1)),
            input_mask=np.zeros((batch_size,) + shape, bool),
            prediction_mask=np.zeros((batch_size,) + shape, bool),
            weight=np.zeros((batch_size,) + shape, "f4"),
        )
        stats = {}
        for k, fun in dict(
            inference=inference,
            train=train,
        ).items():
            # Analyze the forward pass function
            jfun = jax.jit(fun)
            tfun = jfun.trace(state, batch)
            cfun = tfun.lower()  # .compile()
            cost = cfun.cost_analysis()
            stats[k] = SimpleNamespace(
                flops=cost.get("flops"),
                bytes_accessed=cost.get("bytes accessed"),
                bytes_out=cost.get("bytes accessedout"),
            )

        return SimpleNamespace(**stats)

    @staticmethod
    @jax.jit
    def _tree_accumulate(acc, g):
        # acc, g: jax.Array with identical sharding (PmapSharding)
        return jax.tree.map(lambda a, b: a + b, acc, g)

    def train_step(
        self,
        minibatch_dicts: tuple[dict, ...],
        total_weight: float,
        learning_rate: float,
        num_devices: int,
        **kw,
    ) -> dict:
        """Perform one training step with gradient accumulation over minibatches.

        Args:
            minibatch_dicts: Tuple of minibatch dicts (with masks already prepared)
            learning_rate: Learning rate for this step
            num_devices: Number of devices for pmap
            **kw: Additional kwargs (e.g., mode)

        Returns:
            Dictionary of training statistics
        """
        kw_tuple = tuple(sorted(kw.items()))

        @nnx.split_rngs(splits=num_devices)
        def compute_grads(state, sharded_minibatch):
            return state._compute_grads(sharded_minibatch, kw_tuple)

        # Accumulate gradients over minibatches (outside pmap)
        accumulated_grads = None
        accumulated_stats = None

        for minibatch_dict in minibatch_dicts:
            # Shard minibatch across devices
            sharded_minibatch = jax.tree.map(
                lambda x: x.reshape(num_devices, -1, *x.shape[1:]),
                minibatch_dict,
            )

            # Compute gradients for this minibatch (pmap'd, no update)
            grads, stats = compute_grads(self, sharded_minibatch)

            # Accumulate gradients
            # (stays on device, no communication)
            if accumulated_grads is None:
                accumulated_grads = grads
                accumulated_stats = stats
            else:
                accumulated_grads = self._tree_accumulate(accumulated_grads, grads)
                accumulated_stats = self._tree_accumulate(accumulated_stats, stats)

        # Apply optimizer update (pmap'd, with psum)
        stats_dev = self._apply_update(
            accumulated_grads, accumulated_stats, total_weight, learning_rate
        )

        # Initiate async to-host transfer; if you schedule more work before doing `jax.device_get`,
        # then this won't synchronise with the devices, thus ensuring high throughput.
        stats = jax.tree.map(
            lambda x: jax.copy_to_host_async(x), stats_dev
        )  # non-blocking
        stats["minibatches"] = len(minibatch_dicts)
        return stats

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
        # aggregate gradients and stats across devices
        print("Tracing _apply_update")

        grads = jax.lax.psum(grads, axis_name="data")
        stats = jax.lax.psum(stats, axis_name="data")

        s = 1 / total_weight
        grads = jax.tree.map(lambda a: a * jnp.array(s, dtype=a.dtype), grads)
        stats = jax.tree.map(lambda a: a * jnp.array(s, dtype=a.dtype), stats)

        # Apply optimizer update
        self.optimizer.update(self.model, grads, learning_rate=learning_rate)

        return stats

    @nnx.pmap(
        axis_name="data",
        in_axes=(
            nnx.StateAxes({nnx.RngState: 0, ...: None}),
            0,
            None,
        ),
        static_broadcasted_argnums=2,
    )
    def _compute_grads(self, minibatch_dict, kw):
        """Compute gradients for one minibatch (pmap'd, no update)."""
        print(f"Tracing _compute_grads for shape {minibatch_dict['images'].shape}")

        def loss_fn(model):
            return self.loss_fn(model, minibatch_dict, **dict(kw))

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (_, stats), grads = grad_fn(self.model)

        grads = nnx.to_pure_dict(grads)

        return grads, stats


@attrs.mutable
class MAETrainer:
    """Manages the MAE training pipeline."""

    config: MAETaskConfig = attrs.field(on_setattr=attrs.setters.frozen)
    train_state: TrainState
    collator: BucketedCollator
    lr_schedule: typing.Callable
    num_devices: int = attrs.field(on_setattr=attrs.setters.frozen)

    total_steps: int = attrs.field(on_setattr=attrs.setters.frozen)

    # k-NN evaluation
    knn_evaluator: KNNEvaluator | None = None

    # Training progress tracking
    step: int = 0
    examples_seen: int = 0

    # Track seen bucket shapes for JIT detection
    _seen_bucket_shapes: set[tuple[int, int]] = attrs.field(factory=set, init=False)

    @classmethod
    def make(
        cls,
        config: MAETaskConfig,
        model: MaskedAutoencoder,
        collator: BucketedCollator,
        num_devices: int = 1,
        *,
        eval_dataset: BucketedDataset | None = None,
        minibatch_size_fn: MinibatchSizeFunction | None = None,
        lr_schedule: typing.Callable | None = None,
        rngs: nnx.Rngs,
    ) -> typing.Self:
        """Create a MAETrainer instance."""
        # Create training state
        train_state = TrainState.make(model, config, rngs=rngs)

        # The schedule is queried in terms of steps (of size `config.batch_size`).
        # calculate the full schedule length
        total_steps = (
            min(
                config.max_num_ref_batches * config.ref_batch,
                config.max_num_epochs * collator.total_example_weight,
            )
            / config.batch_size
        )

        if lr_schedule is None:
            # --- Create learning rate schedule
            # The configured learning-date is specified at `config.ref_batch`,
            # but we optimise using batches approximately `config.batch_size`.
            # With Adam, we use sqrt-scaling
            lr_scale = np.sqrt(config.batch_size / config.ref_batch)
            lr = config.learning_rate * lr_scale

            lr_schedule = optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=total_steps,
                alpha=0.001,
            )

        # Create k-NN evaluator if eval_dataset provided
        knn_evaluator = None
        if eval_dataset is not None:
            if minibatch_size_fn is None:
                raise ValueError(
                    "minibatch_size_fn required when eval_dataset is provided"
                )
            knn_evaluator = KNNEvaluator(
                dataset=eval_dataset,
                batch_size=minibatch_size_fn,
                seed=config.seed,
            )

        return cls(
            config=config,
            train_state=train_state,
            collator=collator,
            lr_schedule=lr_schedule,
            num_devices=num_devices,
            total_steps=total_steps,
            knn_evaluator=knn_evaluator,
        )

    def prepare_mae_batch(
        self, minibatches: tuple[MiniBatchData, ...]
    ) -> tuple[dict, ...]:
        """Prepare minibatches for MAE training by adding random masks.

        Uses the collator's RNG for reproducibility.

        Args:
            minibatches: Tuple of MiniBatchData from collator

        Returns:
            Tuple of dicts with added input_mask and prediction_mask fields
        """
        prepared = []
        test_ratio = self.config.test_ratio
        nonmask_fraction = self.config.nonmask_fraction
        randomise_fraction = self.config.randomise_fraction
        rng = self.collator.rng

        for minibatch in minibatches:
            # Convert to dict
            mb_dict = attrs.asdict(minibatch, recurse=False)

            images = mb_dict["images"]
            weights = mb_dict["weight"]
            masks = mb_dict["masks"]

            # Generate random input mask using collator's numpy RNG
            # Each visible cell has (1 - mask_ratio) chance of being in input
            random_vals = rng.uniform(size=images.shape)
            prediction_mask = (random_vals < test_ratio) & masks
            unmasked_prediction = masks & (random_vals < nonmask_fraction * test_ratio)
            input_mask = unmasked_prediction | (masks & ~prediction_mask)
            changeup_mask = masks & (
                random_vals < randomise_fraction * nonmask_fraction * test_ratio
            )

            mb_dict["images"] = np.where(
                changeup_mask,
                (
                    images
                    + rng.integers(
                        low=1, high=10, size=images.shape, dtype=images.dtype
                    )
                )
                % 10,
                images,
            )

            # Add masks to dict
            mb_dict["input_mask"] = input_mask
            mb_dict["prediction_mask"] = prediction_mask

            # Compute cell weights from class histograms per example
            targets = (images[..., None] == np.r_[:10]) & prediction_mask[..., None]
            mask_hist = targets.sum(axis=(-3, -2), keepdims=True)
            # weight inversely proportional to frequeny
            cell_weight = jnp.where(targets, 1 / jnp.maximum(1, mask_hist), 0).sum(-1)
            # normalise weight
            cell_weight = cell_weight / jnp.maximum(
                cell_weight.sum(axis=(-2, -1), keepdims=True), 1e-3
            )

            mb_dict["weight"] = weights[..., None, None] * cell_weight

            prepared.append(mb_dict)

        return tuple(prepared)

    def train(self) -> typing.Iterator[tuple[dict, bool]]:
        """Main training loop. Yields (stats, is_jit_step) for each training step.

        Returns:
            Iterator of (stats_dict, is_jit_step) tuples where:
            - stats_dict: Training statistics (with async device-to-host copy initiated)
            - is_jit_step: True if this batch contains new bucket shapes (likely triggers JIT)
        """
        start_weight = None
        for batch_data in self.collator.generate():
            # Check if we've reached max examples
            if self.step >= self.total_steps:
                break

            if start_weight is None:
                start_weight = batch_data.accumulated_weight - batch_data.total_weight

            # Check if this batch contains new bucket shapes
            bucket_shapes = {mb.images.shape[1:] for mb in batch_data.minibatches}
            is_jit_step = bool(bucket_shapes - self._seen_bucket_shapes)
            self._seen_bucket_shapes.update(bucket_shapes)

            # Prepare batch with MAE masks (using collator's RNG)
            prepared_minibatches = self.prepare_mae_batch(batch_data.minibatches)

            # Compute learning rate based on total example weight seen so far
            target_lr = self.lr_schedule(
                (batch_data.accumulated_weight - start_weight) / self.config.batch_size
            )

            # Apply warmup (this one goes really in steps, not in accumlated weight)
            if self.step < self.config.warmup_steps:
                target_lr *= (self.step + 1) / self.config.warmup_steps

            # linearly scale learning rate with respect to batch weight fluctuations
            weighted_lr = (
                target_lr
                * batch_data.total_weight
                / self.collator.batch_spec.target_batch_weight
            )

            # Perform training step (returns stats with async copy initiated)
            stats = self.train_state.train_step(
                minibatch_dicts=prepared_minibatches,
                total_weight=batch_data.total_weight,
                learning_rate=weighted_lr,
                num_devices=self.num_devices,
                mode=self.config.mode,
                remat=self.config.remat,
                unroll=self.config.unroll,
            )

            # Update tracking
            self.step += 1
            self.examples_seen += batch_data.total_examples

            # Add tracking info to stats (these are scalars, not device arrays)
            stats.update(
                data_step=batch_data.step,
                training_step=self.step,
                epoch=batch_data.epoch,
                epoch_progress=batch_data.epoch_progress,
                examples_seen=self.examples_seen,
                accumulated_weight=batch_data.accumulated_weight - start_weight,
                learning_rate=weighted_lr,
                batch_weight=batch_data.total_weight,
            )

            yield stats, is_jit_step

    @classmethod
    def main(
        cls,
        config: MAETaskConfig,
        model: MaskedAutoencoder,
        dataset: BucketedDataset,
        *,
        eval_dataset: BucketedDataset | None = None,
        checkpoint_dir: Path | str | None = None,
        lr_schedule: typing.Callable | None = None,
        wandb_run=None,
        run_name: str | None = None,
        num_devices: int | None = None,
    ):
        """Main training entry point with progress tracking and logging.

        Args:
            config: Training configuration
            model: Model to train
            dataset: Bucketed training dataset
            eval_dataset: Optional bucketed evaluation dataset for k-NN validation
            checkpoint_dir: Optional directory for saving checkpoints
            lr_schedule: Optional custom learning rate schedule
            wandb_run: Optional wandb run object for logging
            num_devices: Number of devices (default: all available)

        Returns:
            Tuple of (trainer, stats_list) where stats_list contains all training statistics
        """
        import tqdm.auto

        if run_name is None:
            now = datetime.datetime.now().astimezone(datetime.timezone.utc)
            run_name = f"{now:%Y%m%d-%H%M}"

        # Setup checkpoint directory
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            chkp_pfx = f"{run_name}-chkp"

        # Detect available devices
        if num_devices is None:
            num_devices = jax.local_device_count()

        # Setup PRNG key
        rngs = nnx.Rngs(config.seed)

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
            # Hard-code area_weight_exponent=0.5 (heuristic: sqrt scaling)
            area_weight_exponent=0.5,
        )

        collator = BucketedCollator.make(
            dataset=dataset,
            batch_spec=batch_spec,
            minibatch_size=minibatch_size_fn,
            seed=config.seed,  # Tie dataset seed to training seed
        )

        # Initialize trainer
        trainer = cls.make(
            config=config,
            model=model,
            collator=collator,
            num_devices=num_devices,
            eval_dataset=eval_dataset,
            minibatch_size_fn=minibatch_size_fn,
            rngs=rngs,
            lr_schedule=lr_schedule,
        )

        import arc25

        base_metadata = dict(
            config=dict(
                trainer=config,
                model=model.config,
            ),
            code_version=arc25.__version__,
        )

        print(f"--- MAE Training ---")
        print(f"Devices: {num_devices} × {jax.devices()[0].device_kind}")
        print(f"Training batch size: {config.batch_size} (1 step)")
        print(
            f"Reference step size: {config.ref_batch} (~{config.ref_batch/config.batch_size:.2f} steps)"
        )
        print(f"Total steps: {int(trainer.total_steps)}")
        if trainer.knn_evaluator is not None:
            print(
                f"k-NN evaluation: every {config.knn_validation_every_ref_batch} reference steps"
            )
        if checkpoint_dir is not None:
            print(
                f"Checkpoints: every {config.checkpoint_every_steps} steps → {checkpoint_dir}/{chkp_pfx}-XXXXXX.msgpack.xz"
            )
        print("----------------------------\n")

        # Training loop with timing and logging
        print("Starting training...")
        start_time = time.monotonic()

        # Timing tracking (exclude JIT compilation time)
        excluded_weight = 0.0
        excluded_time = 0.0

        # For latency hiding: keep previous step's stats and JIT flag
        pending_stats = None
        pending_is_jit_step = False

        # tracking
        last_eval = None
        all_stats = []

        def process_stats(stats_dev, was_jit_step):
            """Process stats from device, compute metrics, log and display."""
            nonlocal excluded_weight, excluded_time, last_eval

            # Get stats from device (blocks if needed)
            stats = jax.device_get(stats_dev)
            elapsed_time = time.monotonic() - start_time

            training_step = stats["training_step"]
            training_progress = stats["accumulated_weight"] / config.ref_batch
            prev_progress = (
                all_stats[-1]["accumulated_weight"] / config.ref_batch
                if all_stats
                else 0
            )
            step_progress = training_progress - prev_progress

            # If this step triggered JIT, count its time as JIT time
            if was_jit_step:
                step_time = elapsed_time - (
                    all_stats[-1]["elapsed_time"] if all_stats else 0
                )

                excluded_weight += step_progress
                excluded_time += step_time

            # Compute throughput metrics (excluding JIT time)
            weight = stats["accumulated_weight"]
            weight_per_sec = (
                (weight - excluded_weight) / max(elapsed_time - excluded_time, 1e-6)
                if elapsed_time > excluded_time + 1
                else weight / max(elapsed_time, 1e-6)
            )

            # Add timing info
            stats["elapsed_time"] = elapsed_time
            stats["weight_per_sec"] = weight_per_sec

            # Store stats
            all_stats.append(stats)

            # k-NN evaluation (periodic)
            if (
                trainer.knn_evaluator is not None
                and training_progress // config.knn_validation_every_ref_batch
                > prev_progress // config.knn_validation_every_ref_batch
            ):
                print(f"\n[Step {training_step}] Running k-NN evaluation...")
                eval_start = time.monotonic()
                last_eval = knn_results = trainer.knn_evaluator.evaluate(
                    trainer.train_state.model.encoder,
                    mode=config.mode,
                    with_progress=True,
                )
                eval_time = time.monotonic() - eval_start
                excluded_time += eval_time

                # Print results
                print(
                    f"k-NN evaluation completed in {eval_time:.1f}s: "
                    + " ".join(
                        f"{kk}: [{','.join(f'{k}={v:.3f}' for k,v in sorted(vv.items()))}]"
                        for kk, vv in knn_results.items()
                    )
                )

                # Add to stats for wandb logging (nested dict for grouped plotting)
                stats["knn"] = knn_results
                stats["knn_eval_time"] = eval_time

            # Checkpointing (periodic)
            if (
                checkpoint_dir is not None
                and not training_step % config.checkpoint_every_steps
            ):
                checkpoint_path = (
                    checkpoint_dir / f"{chkp_pfx}-{training_step:06d}.msgpack.xz"
                )
                print(
                    f"[Step {training_step}] Saving checkpoint to {checkpoint_path.name}..."
                )
                checkpoint_start = time.monotonic()
                save_model(
                    trainer.train_state,
                    checkpoint_path,
                    metadata=dict(
                        **base_metadata, stats=dict(last_knn=last_eval, **stats)
                    ),
                )
                checkpoint_time = time.monotonic() - checkpoint_start
                excluded_time += checkpoint_time
                print(f"Checkpoint saved in {checkpoint_time:.1f}s")
                stats["checkpoint_time"] = checkpoint_time

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                rstep=f"{training_progress:.1f}",
                loss=f"{stats['loss']:.3f}",
                acc=f"{stats['accuracy']:.3f}",
                wps=f"{weight_per_sec:.1f}",
                lr=f"{stats['learning_rate']:.2e}",
                ep=f"{stats['epoch']+stats['epoch_progress']:.2f}",
            )

            # Log to wandb
            if wandb_run is not None:
                wandb_run.log(stats, step=training_step)

        with tqdm.auto.tqdm(total=int(trainer.total_steps), desc="Training") as pbar:
            try:
                for stats_dev, is_jit_step in trainer.train():
                    # If we have pending stats from previous step, process them now
                    # (device-to-host copy should be complete by now, or we block here)
                    if pending_stats is not None:
                        process_stats(pending_stats, pending_is_jit_step or is_jit_step)

                    # Store current stats and JIT flag as pending for next iteration
                    pending_stats = stats_dev
                    pending_is_jit_step = is_jit_step

                # Process final pending stats
                if pending_stats is not None:
                    process_stats(pending_stats, pending_is_jit_step)

            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                if pending_stats is not None:
                    process_stats(pending_stats, pending_is_jit_step)

        # Save final checkpoint
        if checkpoint_dir is not None and all_stats:
            stats = all_stats[-1]
            training_step = stats["training_step"]
            checkpoint_path = (
                checkpoint_dir / f"{chkp_pfx}-{training_step:06d}-final.msgpack.xz"
            )
            print(f"\nSaving final checkpoint to {checkpoint_path.name}...")
            save_model(
                trainer.train_state,
                checkpoint_path,
                metadata=dict(**base_metadata, stats=stats),
            )
            print(f"Final checkpoint saved")

        elapsed_time = time.monotonic() - start_time
        print("\n--- Training Finished ---")
        print(f"Total time: {elapsed_time:.1f}s")
        wps = (all_stats[-1]["accumulated_weight"] - excluded_weight) / (
            elapsed_time - excluded_time
        )
        print(f"Average throughput: {wps:.1f} weight/s")

        return trainer, all_stats
