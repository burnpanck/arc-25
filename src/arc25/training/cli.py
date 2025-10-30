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
import etils.epath
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
from ..serialisation import serialisable
from ..vision2 import arc_solver, mae
from . import arc_solver as arc_solver_trainer
from . import dataset, gcp
from . import mae as mae_trainer
from . import saving
from .config import ImageTrainConfigBase, describe_config_json

proj_root = Path(os.environ.get("ARC25_APP_ROOT", Path(__file__).parents[3])).resolve()
data_root = proj_root / "data"


@serialisable
@attrs.frozen
class ModelSelection:
    type: Literal["mae", "arc-solver"] = "mae"
    config: Literal["tiny", "small", "legacy"] = "small"
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


@serialisable
@attrs.frozen
class Training:
    """Unified training configuration for both MAE and ArcSolver."""

    run_name: str
    model: ModelSelection = ModelSelection()

    # Polymorphic training configs - exactly one should be set based on model.type
    mae_training: mae_trainer.MAETaskConfig | None = None
    arc_solver_training: arc_solver_trainer.ArcSolverConfig | None = None

    # Common parameters
    size_bins: frozenset[int] = frozenset([12, 20, 30])
    starting_checkpoint: str | None = attrs.field(
        default=None,
        metadata=dict(help="Path to the checkpoint to start from."),
    )
    # starting-checkpoint type: What kind of model to expect, and what to do with it;
    # - "mae" and "arc-solver" update all model weights, expecting a full matching checkpoint.
    # - "with-encoder" and "encoder" only update the encoder, with the former looking for the encoder one layer within the model.
    starting_checkpoint_type: (
        Literal["with-encoder", "encoder", "mae", "arc-solver"] | None
    ) = None
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
def train(task: Training):
    """
    Run training with WandB logging and GCS checkpointing.

    Supports both MAE pretraining and ArcSolver fine-tuning.
    WandB API key is fetched from GCP Secret Manager using Application Default Credentials.
    Checkpoints are saved to GCS bucket specified by AIP_STORAGE_URI or checkpoint-base-uri.
    """
    print(f"train({task})")

    # Validate polymorphic config
    match task.model.type:
        case "mae":
            assert task.mae_training is not None, "mae_training must be set for MAE"
            assert (
                task.arc_solver_training is None
            ), "arc_solver_training must be None for MAE"
            config = task.mae_training
            wandb_project = "arc-vision-v2-mae"
        case "arc-solver":
            assert (
                task.arc_solver_training is not None
            ), "arc_solver_training must be set for ArcSolver"
            assert task.mae_training is None, "mae_training must be None for ArcSolver"
            config = task.arc_solver_training
            wandb_project = "arc-vision-v2-solver"
        case _:
            raise ValueError(f"Unknown model type: {task.model.type}")

    # Common setup
    data_file = data_root / "repack/re-arc.cbor.xz"

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
    if task.checkpoint_base_uri:
        # User-specified base, append run_name
        checkpoint_dir = etils.epath.Path(task.checkpoint_base_uri) / run_name
    elif aip_dir := os.environ.get("AIP_CHECKPOINT_DIR"):
        # Vertex AI dir is already unique per run
        checkpoint_dir = etils.epath.Path(aip_dir)
    else:
        checkpoint_dir = None

    if checkpoint_dir is not None:
        print(f"Checkpoint directory: {checkpoint_dir}")
        sys.stdout.flush()

    if wandb_key is not None:
        import wandb

        # Login to WandB
        wandb.login(
            key=wandb_key,
            verify=True,
        )

    # Model-specific training
    match task.model.type:
        case "mae":
            # MAE pretraining
            print(f"Loading data from {data_file} ({data_file.stat().st_size} bytes)")
            sys.stdout.flush()
            src_dataset = dataset.ImagesDataset.load_compressed_cbor(data_file)

            size_cuts = list(task.size_bins)

            full_eval_split, train_split = src_dataset.split_by_challenge(
                np.random.default_rng(seed=42),
                n_min=100,
            )
            eval_split, _ = full_eval_split.split_by_challenge(
                np.random.default_rng(seed=77),
                n_min=24,
            )

            training_ds, eval_ds = [
                dataset.BucketedDataset.make(
                    s,
                    set(itertools.product(size_cuts, size_cuts)),
                )
                for s in [train_split, eval_split]
            ]

            print(f"Creating MaskedAutoencoder model")
            sys.stdout.flush()

            model = mae.MaskedAutoencoder(
                **mae.configs[task.model.config],
                dtype=getattr(jnp, task.model.dtype),
                rngs=nnx.Rngs(config.seed),
            )

            trainer_cls = mae_trainer.MAETrainer
            trainer_kw = dict(
                dataset=training_ds,
                eval_dataset=eval_ds,
            )

        case "arc-solver":
            # ArcSolver training
            print(f"Loading data from {data_file} ({data_file.stat().st_size} bytes)")
            sys.stdout.flush()

            src_dataset = dataset.ImagesDataset.load_compressed_cbor(
                data_root / "repack/re-arc.cbor.xz",
                filter=lambda iop, ex: iop.input.shape == iop.output.shape,
            )

            size_buckets = list(task.size_bins)
            num_latent_programs = len(src_dataset.challenges)

            num_solution_attempts = config.num_solution_attempts

            print(
                f"Creating ARCSolver model with {num_latent_programs}×{num_solution_attempts} latent programs"
            )
            sys.stdout.flush()

            model = arc_solver.ARCSolver(
                **arc_solver.configs[task.model.config],
                dtype=getattr(jnp, task.model.dtype),
                num_latent_programs=num_latent_programs * num_solution_attempts,
                rngs=nnx.Rngs(config.seed),
            )

            eval_split, train_split = src_dataset.split_by_challenge(
                np.random.default_rng(seed=42),
                n_min=100,
            )

            trainer_cls = arc_solver_trainer.ArcSolverTrainer
            trainer_kw = dict(
                dataset=train_split,  # Pass raw dataset (with inputs and outputs)
                eval_dataset=eval_split,
                bucket_shapes=set(itertools.product(size_buckets, size_buckets)),
            )

            # Load encoder checkpoint (required for ArcSolver)
            if task.starting_checkpoint is None:
                raise RuntimeError(f"For ArcSolver, an encoder checkpoint is required")

        case _:
            raise ValueError()

    # Load encoder checkpoint if provided (for multi-stage training)
    if task.starting_checkpoint is not None:
        chkp_path = etils.epath.Path(task.starting_checkpoint)
        schpt = task.starting_checkpoint_type or task.model.type

        print(f"Loading {schpt} from checkpoint: {chkp_path}")
        sys.stdout.flush()

        starting_checkpoint = saving.load_model(chkp_path)
        match schpt:
            case "with-encoder":
                nnx.update(model.encoder, starting_checkpoint.state.model.encoder)
            case "encoder":
                raise NotImplementedError
                nnx.update(model.encoder, starting_checkpoint.state.model)
            case "mae" | "arc-solver":
                assert task.model.type == schpt
                nnx.update(model, starting_checkpoint.state.model)
            case _:
                raise ValueError(schpt)
        print("Encoder loaded successfully")

    run_metadata = dict(
        task_setup={
            k: v
            for k, v in attrs.asdict(task, recurse=False).items()
            if k not in {"mae_training", "arc_solver_training"}
        },
    )

    sys.stdout.flush()
    trainer, stats = trainer_cls.main(
        model=model,
        config=config,
        wandb_project=wandb_project if wandb_key is not None else None,
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        run_metadata=run_metadata,
        **trainer_kw,
    )


if __name__ == "__main__":
    os.environ["EPATH_USE_TF"] = "false"

    cli()
