"""Base classes for training infrastructure shared between MAE and ArcSolver."""

import contextlib
import datetime
import sys
import time
import typing
from abc import abstractmethod

import attrs
import etils.epath
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import arc25

from .config import ImageTrainConfigBase, describe_config_json
from .dataset import BatchData, BucketedCollator, MiniBatchData
from .learning_rate import scale_by_kwarg
from .saving import load_model, save_model


class TrainStateBase(nnx.Module):
    """Base training state with distributed training logic.

    Subclasses must implement the loss_fn() method.
    """

    model: nnx.Module
    optimizer: nnx.Optimizer
    rngs: nnx.Rngs

    train_filter: typing.ClassVar = nnx.Param

    @classmethod
    def _make_optimiser(cls, config: ImageTrainConfigBase, trainable_state: nnx.State):
        exclusions = []
        for e in config.exclude_from_wd:
            match e:
                case "embedding":
                    exclusions.append(
                        nnx.Any(
                            "embedding",
                            lambda path, v: any("embedding" in p.lower() for p in path),
                        )
                    )
                case "norm":
                    exclusions.append("norm")
                case "bias":
                    exclusions.append(nnx.PathContains("bias"))
                case _:
                    raise ValueError(
                        f"Unknown weight group to exclude from weight decay: {e!r}"
                    )
        if exclusions:
            wd_filter = nnx.Not(nnx.Any(*exclusions))
        else:
            wd_filter = Ellipsis
        weight_decay_mask = nnx.map_state(
            nnx.filterlib.to_predicate(wd_filter), trainable_state
        )
        return (
            optax.scale_by_adam(b1=config.betas[0], b2=config.betas[1], eps=config.eps),
            optax.add_decayed_weights(config.weight_decay, mask=weight_decay_mask),
        )

    @classmethod
    def make(
        cls,
        model: nnx.Module,
        config: ImageTrainConfigBase,
        *,
        rngs: nnx.Rngs,
    ) -> typing.Self:
        """Initialize training state with model and optimizer."""
        # Create the AdamW optimizer with gradient clipping
        trainable_state = nnx.state(model, cls.train_filter)

        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            *cls._make_optimiser(config, trainable_state),
            scale_by_kwarg(),
        )

        self = cls()
        self.model = model
        self.optimizer = nnx.Optimizer(model, tx, wrt=cls.train_filter)
        self.rngs = rngs
        return self

    @abstractmethod
    def loss_fn(self, model, minibatch_dict, params, **kw):
        """Compute loss for a single minibatch.

        Gradients will be computed against `model`, but not any other
        arguments (in particular, not against `self`).

        Args:
            model: The model to evaluate
            minibatch_dict: Dict with task-specific keys
            **kw: Additional kwargs passed to model

        Returns:
            Tuple of (loss, stats_dict)
        """
        pass

    @staticmethod
    @jax.jit
    def _tree_accumulate(acc, g):
        """Accumulate gradients/stats across minibatches."""
        return jax.tree.map(lambda a, b: a + b, acc, g)

    def train_step(
        self,
        minibatch_dicts: tuple[dict, ...],
        params: dict,
        num_devices: int,
        loss_kw: dict,
    ) -> dict:
        """Perform one training step with gradient accumulation over minibatches.

        Args:
            minibatch_dicts: Tuple of minibatch dicts
            total_weight: Total weight across all minibatches
            learning_rate: Learning rate for this step
            num_devices: Number of devices for pmap
            **kw: Additional kwargs (e.g., mode, remat, unroll)

        Returns:
            Dictionary of training statistics
        """
        kw_tuple = tuple(sorted(loss_kw.items()))

        @nnx.split_rngs(splits=num_devices)
        def compute_grads(state, sharded_minibatch):
            return state._compute_grads(sharded_minibatch, params, kw_tuple)

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

            # Accumulate gradients (stays on device, no communication)
            if accumulated_grads is None:
                accumulated_grads = grads
                accumulated_stats = stats
            else:
                accumulated_grads = self._tree_accumulate(accumulated_grads, grads)
                accumulated_stats = self._tree_accumulate(accumulated_stats, stats)

        # Apply optimizer update (pmap'd, with psum)
        stats_dev = self._apply_update(accumulated_grads, accumulated_stats, params)

        # Initiate async to-host transfer
        stats = jax.tree.map(lambda x: jax.copy_to_host_async(x), stats_dev)
        stats["minibatches"] = len(minibatch_dicts)
        return stats

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

        total_weight = params["total_weight"]
        learning_rate = params["learning_rate"]

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
            None,
        ),
        static_broadcasted_argnums=3,
    )
    def _compute_grads(self, minibatch_dict, params, loss_kw):
        """Compute gradients for one minibatch (pmap'd, no update)."""
        loss_kw = dict(loss_kw)
        shapes = ", ".join(
            f"{k}=({','.join(str(i) for i in v.shape)})"
            for k, v in minibatch_dict.items()
            if v is not None
        )
        kwstr = ", ".join(f"{k}={v!r}" for k, v in loss_kw.items())
        print(f"Tracing _compute_grads for shape dict({shapes}) (kw=dict({kwstr}))")
        sys.stdout.flush()

        def loss_fn(model):
            return self.loss_fn(model, minibatch_dict, params, **loss_kw)

        grad_fn = nnx.value_and_grad(
            loss_fn,
            argnums=nnx.DiffState(0, self.train_filter),
            has_aux=True,
        )
        (_, stats), grads = grad_fn(self.model)

        grads = nnx.to_pure_dict(grads)

        return grads, stats


@attrs.mutable
class TrainerBase:
    """Base trainer with common training loop logic.

    Subclasses must implement prepare_batch() method.
    """

    config: ImageTrainConfigBase = attrs.field(on_setattr=attrs.setters.frozen)
    train_state: TrainStateBase
    collator: BucketedCollator
    lr_schedule: typing.Callable
    num_devices: int = attrs.field(on_setattr=attrs.setters.frozen)

    total_steps: int = attrs.field(on_setattr=attrs.setters.frozen)

    # Training progress tracking
    step: int = 0
    examples_seen: int = 0

    # Track seen bucket shapes for JIT detection
    _seen_bucket_shapes: set[tuple[int, int]] = attrs.field(factory=set, init=False)

    with_progress_bars: bool = False

    @classmethod
    def _make_lr_schedule(
        cls, config: ImageTrainConfigBase, total_steps: float
    ) -> typing.Callable:
        """Create default learning rate schedule."""
        lr_scale = np.sqrt(config.batch_size / config.ref_batch)
        lr = config.learning_rate * lr_scale

        return optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=total_steps,
            alpha=0.001,
        )

    @classmethod
    def _calculate_total_steps(
        cls, config: ImageTrainConfigBase, collator: BucketedCollator
    ) -> float:
        """Calculate total training steps from config."""
        total_weight = None
        if config.max_num_ref_batches is not None:
            total_weight = config.max_num_ref_batches * config.ref_batch
        if config.max_num_epochs is not None:
            tw = config.max_num_epochs * collator.total_example_weight
            total_weight = tw if total_weight is None else min(tw, total_weight)
        assert total_weight is not None
        return total_weight / config.batch_size

    @abstractmethod
    def prepare_batch(self, batch: BatchData) -> tuple[tuple[dict, ...], dict]:
        """Prepare minibatches for training (task-specific).

        Args:
            batch: BatchData from collator

        Returns:
            minibatch_dicts: Tuple of dicts ready for train_step()
            params: dict ready for train_step()
        """
        pass

    def periodic_evaluation(self, stats: dict) -> tuple[dict, float] | None:
        """Periodic evaluation hook (task-specific).

        Called during training loop to perform periodic evaluations.

        Args:
            stats: Current training statistics

        Returns:
            Tuple of (eval_results_dict, eval_time_seconds) or None if no evaluation performed.
            Eval results are merged into stats for logging.
        """
        return None

    def _warmup_factor(self) -> float:
        if self.step >= self.config.warmup_steps:
            return 1.0
        return (self.step + 1) / self.config.warmup_steps

    def train(
        self, *, start_weight: float | None = None
    ) -> typing.Iterator[tuple[dict, bool]]:
        """Main training loop. Yields (stats, is_jit_step) for each training step.

        Returns:
            Iterator of (stats_dict, is_jit_step) tuples where:
            - stats_dict: Training statistics (with async device-to-host copy initiated)
            - is_jit_step: True if this batch contains new bucket shapes (likely triggers JIT)
        """
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

            # Prepare batch (task-specific)
            prepared_minibatches, params = self.prepare_batch(batch_data)

            # Compute learning rate based on total example weight seen so far
            target_lr = self.lr_schedule(
                (batch_data.accumulated_weight - start_weight) / self.config.batch_size
            )

            # Apply warmup
            target_lr *= self._warmup_factor()

            # Linearly scale learning rate with respect to batch weight fluctuations
            weighted_lr = (
                target_lr
                * batch_data.total_weight
                / self.collator.batch_spec.target_batch_weight
            )

            # Perform training step (returns stats with async copy initiated)
            stats = self.train_state.train_step(
                minibatch_dicts=prepared_minibatches,
                params=params
                | dict(
                    total_weight=batch_data.total_weight,
                    learning_rate=weighted_lr,
                ),
                num_devices=self.num_devices,
                loss_kw=dict(
                    mode=self.config.mode,
                    remat=self.config.remat,
                    unroll=self.config.unroll,
                ),
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
                accumulated_weight=batch_data.accumulated_weight,
                learning_rate=weighted_lr,
                batch_weight=batch_data.total_weight,
            )

            yield stats, is_jit_step

    def run_main(
        self,
        *,
        checkpoint_dir: etils.epath.Path | str | None = None,
        wandb_project: str | None = None,
        run_name: str | None = None,
        run_metadata: dict | None = None,
    ) -> list[dict]:
        """Common checkpoint/wandb/training loop logic.

        Args:
            checkpoint_dir: Optional checkpoint directory
            wandb_project: Optional wandb project name
            run_name: Run name for checkpoints/wandb
            config: Training configuration

        """

        import tqdm.auto

        config = self.config
        model = self.train_state.model

        if run_name is None:
            now = datetime.datetime.now().astimezone(datetime.timezone.utc)
            run_name = f"{now:%Y%m%d-%H%M}-{type(self).__name__}"

        devices = jax.devices()

        # Print training info
        print(f"--- {type(self).__name__} ---")
        print(f"Run: {run_name}")
        print(f"Devices: {self.num_devices} × {devices[0].device_kind}")
        print(f"Training batch data weight: {config.batch_size} (1 optimizer step)")
        print(
            f"Reference step data weight: {config.ref_batch} (~{config.ref_batch/config.batch_size:.2f} optimizer steps)"
        )
        print(f"Total steps: {int(self.total_steps)}")
        print(f"Evaluation: every {config.eval_every_ref_batch} reference steps")
        if checkpoint_dir is not None:
            print(f"Checkpoints: every {config.checkpoint_every_steps} optimizer steps")
        print("----------------------------\n")

        # Build metadata/config dicts
        wandb_config = describe_config_json(
            dict(
                **vars(config),
                model=model.config,
                code_version=arc25.__version__,
                **run_metadata or {},
            )
        )

        base_metadata = dict(
            config=dict(self=config, model=model.config),
            code_version=arc25.__version__,
            **(run_metadata or {}),
        )

        # Setup checkpoint directory and scan for existing checkpoints
        resume_from_checkpoint = None
        resume_step = 0
        resume_accumulated_weight = None
        resume_wandb_id = None

        if checkpoint_dir is not None:
            checkpoint_dir = etils.epath.Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            chkp_pfx = f"{run_name}-chkp"

            # Scan for existing checkpoints
            def parse_step(path: etils.epath.Path) -> int | None:
                stem = path.stem
                if stem.endswith(".msgpack"):
                    stem = stem[: -len(".msgpack")]
                try:
                    return int(stem.split("-")[-1])
                except ValueError:
                    return None

            checkpoint_files = [
                (p, step)
                for p in checkpoint_dir.glob(f"{chkp_pfx}-*.msgpack.xz")
                if "-final" not in p.name and (step := parse_step(p)) is not None
            ]

            if checkpoint_files:
                resume_from_checkpoint, resume_step = max(
                    checkpoint_files, key=lambda x: x[1]
                )
                print(
                    f"Found checkpoint to resume from: {resume_from_checkpoint.name} "
                    f"(step {resume_step})"
                )
        else:
            chkp_pfx = None

        # Resume from checkpoint if found
        if resume_from_checkpoint is not None:
            print(f"Loading checkpoint from {resume_from_checkpoint}...")
            checkpoint_data = load_model(resume_from_checkpoint)

            # Update training state
            nnx.update(self.train_state, checkpoint_data.state)

            # Restore progress tracking
            if (stats := checkpoint_data.metadata.get("stats")) is not None:
                self.step = stats.get("training_step", resume_step)
                data_step = stats.get("data_step", self.step)
                resume_accumulated_weight = stats.get("accumulated_weight")
                self.examples_seen = stats.get("examples_seen", 0)
                print(
                    f"Checkpoint loaded. Resuming from step {self.step} (data: {data_step}, "
                    f"examples seen: {self.examples_seen})"
                )
            else:
                self.step = resume_step
                data_step = resume_step
                print(f"Checkpoint loaded. Resuming from step {resume_step}")

            # Extract wandb run ID
            resume_wandb_id = checkpoint_data.metadata.get("wandb_run_id")
            if resume_wandb_id:
                print(f"Will resume wandb run: {resume_wandb_id}")

            # Fast-forward collator
            print(f"Fast-forwarding data pipeline by {data_step} steps...")
            self.collator.fast_forward(data_step)
            print("Data pipeline synchronized")

        # Initialize wandb if project name provided
        wandb_run = None
        if wandb_project is not None:
            import wandb

            wandb_run = wandb.init(
                project=wandb_project,
                id=resume_wandb_id,
                resume="allow",
                name=run_name,
                config=wandb_config,
            )
            print(f"Wandb run initialized: {wandb_run.id}")

            # Update base_metadata with wandb run ID
            base_metadata["wandb_run_id"] = wandb_run.id

        # Helper function to save checkpoints
        def save_checkpoint(checkpoint_path: etils.epath.Path, metadata: dict) -> None:
            """Save checkpoint (works for both local and GCS paths)."""
            save_model(self.train_state, checkpoint_path, metadata=metadata)

        print("Starting training...")
        start_time = time.monotonic()

        # Timing tracking (exclude JIT compilation time)
        excluded_weight = None
        excluded_time = 0.0

        # For latency hiding
        pending_stats = None
        pending_is_jit_step = False

        # Tracking
        last_training_step = 0
        all_stats = []

        def process_stats(stats_dev, was_jit_step):
            """Process stats from device, compute metrics, log and display."""
            nonlocal excluded_weight, excluded_time, last_training_step

            # Get stats from device (blocks if needed)
            stats = jax.device_get(stats_dev)
            elapsed_time = time.monotonic() - start_time

            if excluded_weight is None:
                excluded_weight = stats["accumulated_weight"] - stats["batch_weight"]

            training_step = stats["training_step"]
            training_progress = stats["accumulated_weight"] / self.config.ref_batch
            prev_progress = (
                all_stats[-1]["accumulated_weight"] / self.config.ref_batch
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

            # Check if it's time for Periodic evaluation (task-specific)
            if (
                training_progress // config.eval_every_ref_batch
                > prev_progress // config.eval_every_ref_batch
            ):
                eval_result = self.periodic_evaluation(stats)
            else:
                eval_result = None
            if eval_result is not None:
                eval_dict, eval_time = eval_result
                stats.update(eval_dict)
                excluded_time += eval_time

            # Checkpointing (periodic)
            if (
                checkpoint_dir is not None
                and not training_step % self.config.checkpoint_every_steps
            ):
                checkpoint_path = (
                    checkpoint_dir / f"{chkp_pfx}-{training_step:06d}.msgpack.xz"
                )
                print(
                    f"[Step {training_step}] Saving checkpoint to {checkpoint_path.name}..."
                )
                checkpoint_start = time.monotonic()
                save_checkpoint(
                    checkpoint_path,
                    metadata=dict(**base_metadata, stats=stats),
                )
                checkpoint_time = time.monotonic() - checkpoint_start
                excluded_time += checkpoint_time
                print(f"Checkpoint saved in {checkpoint_time:.1f}s")
                stats["checkpoint_time"] = checkpoint_time

            # Update progress bar
            n_step_done = max(0, training_step - last_training_step)
            last_training_step += n_step_done
            if pbar is not None:
                pbar.update(n_step_done)
                pbar.set_postfix(
                    rstep=f"{training_progress:.1f}",
                    loss=f"{stats['loss']:.3f}",
                    wps=f"{weight_per_sec:.1f}",
                    lr=f"{stats['learning_rate']:.2e}",
                    ep=f"{stats['epoch']+stats['epoch_progress']:.2f}",
                )

            # Log to wandb
            if wandb_run is not None:
                wandb_run.log(stats, step=training_step)

        with contextlib.ExitStack() as stack:
            if self.with_progress_bars or True:
                pbar = stack.enter_context(
                    tqdm.auto.tqdm(total=int(self.total_steps), desc="Training")
                )
            else:
                pbar = None

            try:
                for stats_dev, is_jit_step in self.train(
                    start_weight=resume_accumulated_weight
                ):
                    # Process pending stats from previous step
                    if pending_stats is not None:
                        process_stats(pending_stats, pending_is_jit_step or is_jit_step)

                    # Store current stats as pending
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
            save_checkpoint(
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

        if wandb_run is not None:
            wandb_run.finish(0)

        return all_stats
