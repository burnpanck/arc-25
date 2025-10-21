import dataclasses
import enum
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import attrs

from .vision2.symrep import RepSpec


@dataclass(frozen=True)
class ImageTrainConfigBase:
    """Configuration for the training script."""

    seed: int = 42
    # this is the batch used per optimiser step, counted in full-size (30x30) examples;
    # smaller examples may count less towards this
    batch_size: int = 1024
    # this is the minibatch-size used for reference_image_size - sized examples;
    # smaller examples may use larger minibatches
    minibatch_size: int = 128
    # the memory cost is estimated as proportional to `base_cell_cost + image_area` in units of cells
    reference_image_size: int = 15
    base_cell_cost: int = 10

    ref_batch: int = 256  # all learning rates refer to this batch size

    # Optimiser
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    # Schedule
    learning_rate: float = 1e-5
    max_num_ref_batches: float | None = None
    max_num_epochs: float | None = 10
    # measured in optimizer steps which are counted in batch_size increments
    warmup_steps: float = 64

    # measured in optimizer steps which are counted in batch_size increments
    checkpoint_every_steps: int = 128

    # implementation
    mode: Literal["split", "flat"] | None = "flat"
    unroll: int | None = None
    remat: bool = True


def describe_config_json(obj):
    """Convert `obj` to the JSON object model, in a way that may not be necessarily 100 % accurate,
    but easily human-readable. The intent is to store configuration info in training logs
    that are visaulised in interactive web tools (W&B, TensorBoard).
    """
    match obj:
        case dict() | MappingProxyType():
            return {
                describe_config_json(k): describe_config_json(v) for k, v in obj.items()
            }
        case tuple() | list():
            return type(obj)(describe_config_json(v) for v in obj)
        case RepSpec():
            return describe_config_json(
                {k: getattr(obj, k) for k in ["space", "n_flavours"]}
            )
        case _ if attrs.has(type(obj)):
            return describe_config_json(attrs.asdict(obj, recurse=False))
        case enum.Enum():
            return f"{type(obj).__qualname__}.{obj.name}"
        case type():
            return obj.__qualname__
        case _ if dataclasses.is_dataclass(obj):
            return {
                f.name: describe_config_json(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            }
        case _:
            return obj
