import contextlib
import dataclasses
import datetime
import gc
import itertools
import os
import sys
import typing
from pathlib import Path
from typing import Literal

import attrs
import click
import jax
import jax.numpy as jnp
import json5
import numpy as np
from flax import nnx

import arc25

from ..vision2 import mae
from . import dataset
from . import mae as mae_trainer
from .config import describe_config_json

proj_root = Path(os.environ.get("ARC25_APP_ROOT", Path(__file__).parents[3])).resolve()
data_root = proj_root / "data"


def attrs_to_click_options(attrs_cls):
    """
    Decorator that automatically generates Click options from attrs class fields.
    DRY approach - single source of truth for configuration parameters.
    """

    def decorator(func):
        # Introspect attrs fields in reverse order (Click applies decorators bottom-up)
        for field in reversed(attrs.fields(attrs_cls)):
            option_name = f"--{field.name.replace('_', '-')}"

            # Determine Click type from field type annotation
            field_type = field.type
            click_type = None
            click_kwargs = dict(
                help=field.metadata.get("help", f"{field.name} parameter")
            )

            # Handle Literal types (enum-like)
            origin = typing.get_origin(field_type)
            if origin is Literal:
                choices = typing.get_args(field_type)
                click_type = click.Choice([str(c) for c in choices])
                click_kwargs["type"] = click_type
            # Handle list types
            elif origin in {list, set, frozenset}:
                args = typing.get_args(field_type)
                inner_type = args[0] if args else str
                click_kwargs["multiple"] = True
                click_kwargs["type"] = inner_type
            # Handle basic types
            elif field_type in {int, float, str, bool}:
                click_kwargs["type"] = field_type
            else:
                # Default to string for complex types
                click_kwargs["type"] = str

            # Set default value
            if isinstance(field.default, attrs.Factory):
                if field.default.takes_self:
                    click_kwargs["default"] = None
                else:
                    click_kwargs["default"] = field.default.factory()
            elif field.default != attrs.NOTHING:
                click_kwargs["default"] = field.default
            else:
                click_kwargs["required"] = True

            # Apply Click option decorator
            func = click.option(option_name, **click_kwargs)(func)

        # Add JSON config option for bulk configuration
        func = click.option(
            "--config-json",
            type=str,
            help="JSON string with configuration (individual options override this)",
        )(func)

        return func

    return decorator


@attrs.frozen
class ModelSelection:
    type: Literal["mae"] = "mae"
    config: Literal["tiny", "small"] = "small"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"


@attrs.frozen
class BatchSizeTuning:
    model: ModelSelection = ModelSelection
    image_sizes: frozenset[int] = frozenset([30])
    start: int = 16
    resolution: float = 0.05


def tune_batch_size_impl(task: BatchSizeTuning):
    """Implementation of batch size tuning."""
    assert task.type == "mae"

    data_file = data_root / "repack/re-arc.cbor.xz"
    print(f"Loading data from {data_file} ({data_file.stat().st_size} bytes)")
    src_dataset = dataset.ImagesDataset.load_compressed_cbor(data_file)
    base_config = mae_trainer.MAETaskConfig(
        seed=42,
        base_cell_cost=0,
        learning_rate=1e-5,
        max_num_epochs=100,
        max_num_ref_batches=40,
        warmup_steps=5,
        mode="flat",
        remat=True,
        unroll=None,
        test_ratio=0.25,
        nonmask_fraction=0.2,
        randomise_fraction=0.2,
    )

    num_devices = len(jax.devices())
    print(f"*** Tuning batch size for image sizes {set(task.image_sizes)}")
    print(f"Devices: {num_devices} × {jax.devices()[0].device_kind}")
    print()
    sys.stdout.flush()

    model = mae.MaskedAutoencoder(
        **mae.configs[task.config],
        dtype=getattr(jnp, task.dtype),
        rngs=nnx.Rngs(42),
    )

    cur = task.start

    for image_size in sorted(task.image_sizes):
        print(f"\n*** Tuning batch size for image size {image_size}x{image_size}")
        training_ds = dataset.BucketedDataset.make(
            src_dataset,
            [(image_size, image_size)],
        )

        cur = 2 ** (cur - 1).bit_length()
        lo = 0
        hi = None
        best = None
        while hi is None or hi - lo > max(1, task.resolution * lo):
            config = dataclasses.replace(
                base_config,
                reference_image_size=image_size,
                batch_size=cur,
                minibatch_size=cur * num_devices,
                ref_batch=cur,
            )

            try:
                train_state, stats = mae_trainer.MAETrainer.main(
                    model=model,
                    config=config,
                    dataset=training_ds,
                )
            except Exception as ex:
                print(f"  Batch size {cur}x{image_size}x{image_size} FAILED: {ex!r}")
                hi = cur
            else:
                weight_per_sec = stats[-1]["weight_per_sec"]
                print(
                    f"  Batch size {cur}x{image_size}x{image_size} succeeded: {weight_per_sec:.2f} wt/s"
                )
                lo = cur
                if best is None or weight_per_sec > best[1]:
                    best = cur, weight_per_sec

            gc.collect()
            jax.clear_caches()
            # Device barrier to flush any pending work before next probe
            jax.device_put(0).block_until_ready()

            cur = lo * 2 if hi is None else (hi + lo) // 2

        print(f">>> Maximum batch size for {image_size}x{image_size}: {lo}")
        if best is not None:
            print(
                f">>> Best batch size for {image_size}x{image_size}: {best[0]} -> {best[1]:.2f} wt/s"
            )
        sys.stdout.flush()


@attrs.frozen
class Pretraining:
    run_name: str
    size_bins: frozenset[int] = frozenset([12, 21, 30])
    model: ModelSelection = ModelSelection
    training: mae_trainer.MAETaskConfig = mae_trainer.MAETaskConfig


def full_pretraining_impl(task: Pretraining):
    import wandb

    assert task.type == "mae"

    now = datetime.datetime.now().astimezone()

    data_file = data_root / "repack/re-arc.cbor.xz"
    print(f"Loading data from {data_file} ({data_file.stat().st_size} bytes)")
    src_dataset = dataset.ImagesDataset.load_compressed_cbor(data_file)
    config = task.training

    run_name = f"{now:%Y%m%d-%H%M}-{task.run_name}"

    size_cuts = list(task.size_bins)

    full_eval_split, train_split = src_dataset.split_by_challenge(
        np.random.default_rng(seed=42),
        n_min=100,
    )
    eval_split, _ = full_eval_split.split_by_challenge(
        np.random.default_rng(seed=77),
        n_min=24,
    )
    valid_split, valid_train_split = full_eval_split.split_by_challenge(
        np.random.default_rng(seed=83),
        n_min=9,
    )

    training_ds, eval_ds, valid_ds, valid_train_ds = [
        dataset.BucketedDataset.make(
            s,
            (
                set(itertools.product(size_cuts, size_cuts))
                if s is not valid_split
                else [(30, 30)]
            ),
        )
        for s in [train_split, eval_split, valid_split, valid_train_split]
    ]

    model = mae.MaskedAutoencoder(
        **mae.configs[task.config],
        dtype=getattr(jnp, task.dtype),
        rngs=nnx.Rngs(config.seed),
    )

    with contextlib.ExitStack() as stack:
        run = stack.enter_context(
            wandb.init(
                project="arc-vision-v2-mae",
                name=run_name,
                reinit="create_new",
                config=vars(config)
                | dict(
                    encoder_config=describe_config_json(model.config),
                    code_version=arc25.__version__,
                ),
            )
        )
        train_state, stats = mae_trainer.MAETrainer.main(
            model=model,
            config=config,
            dataset=training_ds,
            eval_dataset=eval_ds,
            wandb_run=run,
            run_name=run_name,
            #            checkpoint_dir=model_root,
        )


@click.group()
def cli():
    """ARC25 Training CLI - train vision models on ARC challenges."""
    pass


@cli.command()
@attrs_to_click_options(BatchSizeTuning)
def tune_batch_size(config_json, **kwargs):
    """
    Tune batch size for training by binary search.

    Finds the maximum batch size that fits in memory for each image size.
    """
    # Start with config from JSON if provided
    config_dict = {}
    if config_json:
        config_dict = json5.loads(config_json)

    # Override with any explicitly provided CLI arguments
    for key, value in kwargs.items():
        if value is not None:  # Only override if explicitly set
            config_dict[key] = value

    # Instantiate the attrs class
    task = BatchSizeTuning(**config_dict)

    # Run the implementation
    tune_batch_size_impl(task)


if __name__ == "__main__":
    cli()
