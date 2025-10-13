from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ImageTrainConfigBase:
    """Configuration for the training script."""

    seed: int = 42
    # this is the batch used per optimiser step, counted in full-size (30x30) examples;
    # smaller examples may count less towards this
    batch_size: int = 1024
    # this is the minibatch-size used for full-size (30x30) examples;
    # smaller examples may use larger minibatches
    minibatch_size: int = 128
    # the memory cost is estimated as proportional to `base_cell_cost + image_area` in units of cells
    base_cell_cost: int = 10

    ref_batch: int = 256  # all learning rates refer to this batch size

    # Optimiser
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    # Schedule
    learning_rate: float = 3e-4
    max_num_ref_batches: float = 128
    max_num_epochs: float = 1
    # optimizer steps are counted in batch_size increments
    warmup_steps: float = 64
    checkpoint_every_steps: int = 128

    # implementation
    mode: Literal["split", "flat"] | None = "flat"
    unroll: int | None = None
    remat: bool = True
