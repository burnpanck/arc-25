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
import numpy as np
from flax import nnx

import arc25

from ..lib.click_tools import (
    attrs_to_click_options,
    reconstruct_hierarchical_config,
    unflatten_config,
)
from ..vision2 import mae
from . import dataset, gcp
from . import mae as mae_trainer
from .config import describe_config_json

proj_root = Path(os.environ.get("ARC25_APP_ROOT", Path(__file__).parents[3])).resolve()
data_root = proj_root / "data"


@attrs.frozen
class ModelSelection:
    type: Literal["mae"] = "mae"
    config: Literal["tiny", "small"] = "small"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"


@click.group()
def cli():
    """ARC25 Training CLI - train vision models on ARC challenges."""
    pass


@attrs.frozen
class BatchSizeTuning:
    model: ModelSelection = ModelSelection
    image_sizes: frozenset[int] = frozenset([30])
    start: int = 1
    resolution: float = 0.05


@cli.command()
@attrs_to_click_options
def tune_batch_size(task: BatchSizeTuning):
    """
    Tune batch size for training by binary search.

    Finds the maximum batch size that fits in memory for each image size.
    """
    assert task.model.type == "mae"

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
        **mae.configs[task.model.config],
        dtype=getattr(jnp, task.model.dtype),
        rngs=nnx.Rngs(42),
    )

    lo = task.start

    for image_size in sorted(task.image_sizes, reverse=True):
        print(f"\n*** Tuning batch size for image size {image_size}x{image_size}")
        training_ds = dataset.BucketedDataset.make(
            src_dataset,
            [(image_size, image_size)],
        )

        cur = 2 ** max(0, lo - 1).bit_length()
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
    size_bins: frozenset[int] = frozenset([12, 20, 30])
    model: ModelSelection = ModelSelection()
    training: mae_trainer.MAETaskConfig = mae_trainer.MAETaskConfig()
    wandb_secret_name: str | None = attrs.field(
        default=None,
        metadata=dict(
            help="Name of the secret in GCP Secret Manager containing WandB API key"
        ),
    )
    gcp_project_id: str | None = attrs.field(
        default=None,
        metadata=dict(help="GCP project ID (defaults to GCP_PROJECT_ID env var)"),
    )
    checkpoint_base_uri: str | None = attrs.field(
        default=None,
        metadata=dict(
            help="Base URI for checkpoints (defaults to AIP_CHECKPOINT_DIR env var)"
        ),
    )


@cli.command()
@attrs_to_click_options
def full_pretraining(task: Pretraining):
    """
    Run full pretraining with WandB logging and GCS checkpointing.

    WandB API key is fetched from GCP Secret Manager using Application Default Credentials.
    Checkpoints are saved to GCS bucket specified by AIP_STORAGE_URI or checkpoint-base-uri.
    """
    print(f"full_pretraining({task})")
    assert task.model.type == "mae"

    data_file = data_root / "repack/re-arc.cbor.xz"
    print(f"Loading data from {data_file} ({data_file.stat().st_size} bytes)")
    src_dataset = dataset.ImagesDataset.load_compressed_cbor(data_file)
    config = task.training

    run_name = task.run_name

    if task.wandb_secret_name is not None:
        # Get WandB API key from GCP Secret Manager
        print(f"Fetching WandB API key from Secret Manager: {task.wandb_secret_name}")
        wandb_key = gcp.get_secret(
            secret_name=task.wandb_secret_name,
            project_id=task.gcp_project_id,
        )
    else:
        wandb_key = None

    # Set up checkpoint directory
    checkpoint_dir = gcp.get_checkpoint_dir(
        run_name=run_name,
        base_uri=task.checkpoint_base_uri,
    )
    if checkpoint_dir is not None:
        is_gcs = str(checkpoint_dir).startswith("/gcs/")
        storage_type = "GCS (via /gcs/ mount)" if is_gcs else "local filesystem"
        print(f"Checkpoint directory ({storage_type}): {checkpoint_dir}")

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

    if wandb_key is not None:
        import wandb

        # Login to WandB
        wandb.login(
            key=wandb_key,
            verify=True,
        )

    model = mae.MaskedAutoencoder(
        **mae.configs[task.model.config],
        dtype=getattr(jnp, task.model.dtype),
        rngs=nnx.Rngs(config.seed),
    )

    train_state, stats = mae_trainer.MAETrainer.main(
        model=model,
        config=config,
        dataset=training_ds,
        eval_dataset=eval_ds,
        wandb_project="arc-vision-v2-mae" if wandb_key is not None else None,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    cli()
