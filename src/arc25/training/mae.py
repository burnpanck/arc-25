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
from .base import TrainerBase, TrainStateBase
from .config import ImageTrainConfigBase, describe_config_json
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


@serialisable(
    renamed_attrs=dict(
        knn_validation_every_ref_batch="eval_every_ref_batch",
    ),
)
@dataclass(frozen=True)
class MAETaskConfig(ImageTrainConfigBase):
    """Configuration for the training script."""

    test_ratio: float = 0.4
    # fraction of test cells that are *not* masked
    nonmask_fraction: float = 0.2
    # fraction of non-masked cells that are randomised
    randomise_fraction: float = 0.5


class TrainState(TrainStateBase):
    """MAE-specific training state."""

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

    def loss_fn(self, model, minibatch_dict, params, **kw):
        """Compute MAE loss for a single minibatch.

        Args:
            model: The model to evaluate
            minibatch_dict: Dict with keys: images, masks, sizes, labels, transpose, weight,
                           input_mask, prediction_mask
            **kw: Additional kwargs passed to model (e.g., mode)
        """
        images = minibatch_dict["images"]
        sizes = minibatch_dict["sizes"]
        # Note: The normalised weight of image `i`` in the fully accumulated batch
        # is given by `weights[i]*N[i]`, where `N[i]` is the number of
        # non-masked cells in image `i`.
        # Therefore, we can simply sum over all cells and end up with a proper
        # average after gradient accumulation.
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
        raise NotImplementedError("The implementation below is outdated")

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


@attrs.mutable
class MAETrainer(TrainerBase):
    """Manages the MAE training pipeline."""

    # k-NN evaluation (MAE-specific)
    knn_evaluator: KNNEvaluator | None = None

    @classmethod
    def make(
        cls,
        config: MAETaskConfig,
        model: MaskedAutoencoder,
        collator: BucketedCollator,
        num_devices: int = 1,
        *,
        eval_dataset: BucketedDataset | None = None,
        eval_batch_size_fn: MinibatchSizeFunction | None = None,
        lr_schedule: typing.Callable | None = None,
        rngs: nnx.Rngs,
    ) -> typing.Self:
        """Create a MAETrainer instance."""
        # Create training state
        train_state = TrainState.make(model, config, rngs=rngs)

        # Calculate total steps using base class helper
        total_steps = cls._calculate_total_steps(config, collator)

        # Create learning rate schedule using base class helper
        if lr_schedule is None:
            lr_schedule = cls._make_lr_schedule(config, total_steps)

        # Create k-NN evaluator if eval_dataset provided (MAE-specific)
        knn_evaluator = None
        if eval_dataset is not None:
            if eval_batch_size_fn is None:
                raise ValueError(
                    "minibatch_size_fn required when eval_dataset is provided"
                )
            knn_evaluator = KNNEvaluator(
                dataset=eval_dataset,
                batch_size=eval_batch_size_fn,
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

    def prepare_batch(self, batch: BatchData) -> tuple[tuple[dict, ...], dict]:
        """Prepare minibatches for MAE training by adding random masks.

        Args:
            batch: BatchData from collator

        Returns:
            minibatch_dicts: Tuple of dicts ready for train_step()
            params: dict ready for train_step()
        """
        prepared = []
        test_ratio = self.config.test_ratio
        nonmask_fraction = self.config.nonmask_fraction
        randomise_fraction = self.config.randomise_fraction
        rngs = self.train_state.rngs

        for minibatch in batch.minibatches:
            # Convert to dict
            mb_dict = attrs.asdict(minibatch, recurse=False)

            images = mb_dict["images"]
            weights = mb_dict["weight"]
            masks = mb_dict["masks"]

            # Generate random input mask using collator's numpy RNG
            # Each visible cell has (1 - mask_ratio) chance of being in input
            random_vals = jax.random.uniform(rngs.data(), shape=images.shape)
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
                    + jax.random.randint(
                        rngs.data(),
                        minval=1,
                        maxval=10,
                        shape=images.shape,
                        dtype=images.dtype,
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
            # weight inversely proportional to frequency
            cell_weight = jnp.where(targets, 1 / jnp.maximum(1, mask_hist), 0).sum(-1)
            # normalise weight
            cell_weight = cell_weight / jnp.maximum(
                cell_weight.sum(axis=(-2, -1), keepdims=True), 1e-3
            )

            mb_dict["weight"] = weights[..., None, None] * cell_weight

            prepared.append(mb_dict)

        params = dict()

        return tuple(prepared), params

    def periodic_evaluation(self, stats: dict) -> tuple[dict, float] | None:
        """Run k-NN evaluation periodically."""
        # Check if it's time for k-NN evaluation
        if self.knn_evaluator is None:
            return None

        training_step = stats["training_step"]
        print(f"\n[Step {training_step}] Running k-NN evaluation...")
        eval_start = time.monotonic()
        knn_results = self.knn_evaluator.evaluate(
            self.train_state.model.encoder,
            mode=self.config.mode,
            with_progress=self.with_progress_bars,
        )
        eval_time = time.monotonic() - eval_start

        # Print results
        print(
            f"k-NN evaluation completed in {eval_time:.1f}s: "
            + " ".join(
                f"{kk}: [{','.join(f'{k}={v:.3f}' for k,v in sorted(vv.items()))}]"
                for kk, vv in knn_results.items()
            )
        )

        # Return results for wandb logging
        return dict(knn=knn_results, knn_eval_time=eval_time), eval_time

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
        wandb_project: str | None = None,
        run_name: str | None = None,
        num_devices: int | None = None,
        **kw,
    ):
        """Main training entry point with progress tracking and logging.

        Args:
            config: Training configuration
            model: Model to train
            dataset: Bucketed training dataset
            eval_dataset: Optional bucketed evaluation dataset for k-NN validation
            checkpoint_dir: Optional directory for saving checkpoints
            lr_schedule: Optional custom learning rate schedule
            wandb_project: Optional wandb project name for logging (enables wandb if provided)
            run_name: Optional run name (defaults to timestamp)
            num_devices: Number of devices (default: all available)

        Returns:
            Tuple of (trainer, stats_list) where stats_list contains all training statistics
        """
        # Detect available devices
        if num_devices is None:
            num_devices = jax.local_device_count()

        # Create collator with proper seed and granularity
        minibatch_size_fn = MinibatchSizeFunction(
            reference_minibatch_size=config.minibatch_size,
            reference_image_size=config.reference_image_size,
            base_cost=config.base_cell_cost,
            granularity=num_devices,  # Ensure divisibility for pmap
        )
        eval_batch_size_fn = MinibatchSizeFunction(
            # heuristic, large batch size - eval has no gradients, so memory is usually not a problem.
            reference_minibatch_size=8 * config.minibatch_size,
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
        self = cls.make(
            config=config,
            model=model,
            collator=collator,
            num_devices=num_devices,
            eval_dataset=eval_dataset,
            eval_batch_size_fn=eval_batch_size_fn,
            rngs=nnx.Rngs(config.seed),
            lr_schedule=lr_schedule,
        )

        # Run common training loop
        res = self.run_main(
            checkpoint_dir=checkpoint_dir,
            wandb_project=wandb_project,
            run_name=run_name,
            **kw,
        )
        return self, res
