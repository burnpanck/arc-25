"""Training infrastructure for ArcSolver (encoder-decoder for ARC tasks)."""

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
from .base import TrainerBase, TrainStateBase
from .config import ImageTrainConfigBase, describe_config_json
from .dataset import (
    BatchData,
    BatchSpec,
    BucketedCollator,
    BucketedDataset,
    ImagesDataset,
    InputLookup,
    MiniBatchData,
    MinibatchSizeFunction,
)
from .learning_rate import scale_by_kwarg
from .saving import save_model


@serialisable
@dataclass(frozen=True)
class ArcSolverConfig(ImageTrainConfigBase):
    """Configuration for ArcSolver training."""

    # No masking parameters (outputs are labels, all cells used for loss)

    # Evaluation settings
    eval_every_ref_batch: float = 32  # Evaluate on test split periodically


class TrainState(TrainStateBase):
    """Training state for ArcSolver with encoder-decoder architecture.

    The encoder is frozen (stop_gradient applied), only decoder is trained.
    """

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

        # Decode to predict outputs
        logits = model(
            inputs,
            input_sizes,
            latent_program_idx=latent_program_idx,
            **kw,
        )

        # Loss on ALL output cells (not masked like MAE)
        per_cell_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=outputs, axis=-1
        )

        # Mask to valid output regions and weight by pre-normalized cell weights
        loss = (jnp.where(output_masks, per_cell_loss, 0) * cell_weights).sum()

        # Per-cell accuracy
        predictions = jnp.argmax(logits, axis=-1)
        per_cell_correct = (predictions == outputs) * output_masks * cell_weights
        per_cell_accuracy = per_cell_correct.sum()

        # Per-pair accuracy (all cells in output must be correct)
        per_pair_correct = (
            (
                # Padding doesn't count against accuracy
                (predictions == outputs)
                | ~output_masks
            )
            .all(axis=(-2, -1))
            .astype(jnp.float32)
        )
        per_pair_accuracy = (per_pair_correct * image_weights).sum()

        return loss, dict(
            loss=loss,
            per_cell_accuracy=per_cell_accuracy,
            per_pair_accuracy=per_pair_accuracy,
        )


@attrs.mutable
class ArcSolverTrainer(TrainerBase):
    """Manages the ArcSolver training pipeline."""

    # ArcSolver-specific: input lookup for matching outputs to inputs
    input_lookup: InputLookup

    @classmethod
    def make(
        cls,
        config: ArcSolverConfig,
        model: ARCSolver,
        collator: BucketedCollator,
        input_lookup: InputLookup,
        num_devices: int = 1,
        *,
        lr_schedule: typing.Callable | None = None,
        rngs: nnx.Rngs,
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
            input_lookup=input_lookup,
            lr_schedule=lr_schedule,
            num_devices=num_devices,
            total_steps=total_steps,
        )

    def prepare_batch(self, minibatches: tuple[MiniBatchData, ...]) -> tuple[dict, ...]:
        """Prepare minibatches for ArcSolver training by looking up inputs.

        Args:
            minibatches: Tuple of MiniBatchData containing outputs

        Returns:
            Tuple of dicts with inputs, outputs, and metadata for training
        """
        prepared = []
        challenges = self.collator.dataset.challenges

        for output_mb in minibatches:
            # Look up corresponding inputs from InputLookup
            input_mb = self.input_lookup.get_inputs_for_batch(
                output_mb, challenges=challenges
            )

            # Extract latent_program_idx from labels (challenge_id is labels[:, 0])
            latent_program_idx = output_mb.labels[:, 0].astype(np.int32)

            # Compute weights
            # output_mb.weight is per-example, pre-normalized across all minibatches
            weights = output_mb.weight
            output_masks = output_mb.masks

            # Compute per-cell weights (uniform across valid cells in each example)
            # Normalized so each example contributes proportionally to weights
            cell_weight_per_example = output_masks / jnp.maximum(
                output_masks.sum(axis=(-2, -1), keepdims=True), 1
            )
            cell_weight = weights[..., None, None] * cell_weight_per_example

            # Per-image weight (for per-pair accuracy)
            image_weight = weights

            # Create training dict
            mb_dict = dict(
                inputs=input_mb.images,
                input_sizes=input_mb.sizes,
                outputs=output_mb.images,
                output_masks=output_masks,
                latent_program_idx=latent_program_idx,
                cell_weight=cell_weight,
                image_weight=image_weight,
            )

            prepared.append(mb_dict)

        return tuple(prepared)
