import datetime
import os
import subprocess
import sys

import json5
import numpy as np
from google.cloud import aiplatform, aiplatform_v1

from arc25.lib.click_tools import _get_config_class, _get_fields
from arc25.training.arc_solver import ArcSolverConfig
from arc25.training.cli import ModelSelection, Training
from arc25.training.mae import MAETaskConfig

dry_run = True

training = "mae"
training = "arc-solver"
model_config = "small"

accelerator = "L4"
accelerator_count = 8

use_spot = False

if dry_run:
    # model_config = "tiny"
    accelerator = "cpu"
    accelerator_count = 1

now = datetime.datetime.now().astimezone(datetime.timezone.utc)

run_name = f"{now:%Y%m%d-%H%M}-vertex-ai-{training}-{model_config}-{accelerator_count}x{accelerator}"
run_name = "20251030-1638-vertex-ai-arc-solver-small-4xv6e"
print(f"Run: {run_name}")

checkpoint = dict(
    tiny="gs://576e2361-arc-agi-2/aiplatform-custom-training-2025-10-23-13:37:52.100/"
    "checkpoints/20251023-1137-vertex-ai-mae-tiny-4xL4/"
    "20251023-1137-vertex-ai-mae-tiny-4xL4-chkp-007568-final.msgpack.xz",
    small_prev="gs://576e2361-arc-agi-2/checkpoints/"
    "20251025-1452-vertex-ai-mae-small-4xL4/"
    "20251025-1452-vertex-ai-mae-small-4xL4-chkp-004096.msgpack.xz",
    small="gs://576e2361-arc-agi-2/checkpoints/"
    "20251029-1911-vertex-ai-mae-small-4xv6e/"
    "20251029-1911-vertex-ai-mae-small-4xv6e-chkp-005376.msgpack.xz",
)[model_config]


base_config = dict(
    reference_image_size=15,
    ref_batch=256 if accelerator != "cpu" else 16,
    max_num_ref_batches=None,
    mode="flat",
    remat=True,
    unroll=None,
)
mae_config = dict(
    test_ratio=0.25,
    nonmask_fraction=0.2,
    randomise_fraction=0.2,
)
arc_solver_config = dict(
    loss_focus=float(np.log(10)),
    loss_focus_eps=0.01,
    loss_focus_limit=float(np.log(10)),
    loss_focus_beta=0.95,
)

match training, model_config:
    case ("mae", "tiny"):
        training_config = MAETaskConfig(
            seed=42,
            batch_size=512 if accelerator != "cpu" else 64,
            base_cell_cost=0,
            minibatch_size=dict(L4=128, cpu=8)[accelerator] * accelerator_count,
            eval_batch_size=dict(L4=256, cpu=8)[accelerator] * accelerator_count,
            learning_rate=1e-5,
            max_num_epochs=5,
            warmup_steps=64,
            checkpoint_every_steps=512,
            eval_every_ref_batch=256,
            **base_config,
            **mae_config,
        )
    case ("arc-solver", "tiny"):
        num_solution_attempts = 4
        training_config = ArcSolverConfig(
            seed=42,
            batch_size=2048 if accelerator != "cpu" else 128,
            base_cell_cost=0,
            minibatch_size=dict(L4=128, v5e=48, cpu=8)[accelerator] * accelerator_count,
            eval_batch_size=dict(L4=128, v5e=64, cpu=8)[accelerator]
            * accelerator_count
            // num_solution_attempts,
            learning_rate=1e-5,
            max_num_epochs=20,
            warmup_steps=128,
            checkpoint_every_steps=128,
            eval_every_ref_batch=256,
            num_solution_attempts=num_solution_attempts,
            **base_config,
            **arc_solver_config,
        )
    case ("mae", "small"):
        training_config = MAETaskConfig(
            seed=43,
            batch_size=1024 if accelerator != "cpu" else 128,
            base_cell_cost=dict(L4=0, v6e=64, cpu=0)[accelerator],
            minibatch_size=dict(L4=16, v6e=40, cpu=8)[accelerator] * accelerator_count,
            eval_batch_size=dict(L4=64, v6e=128, cpu=8)[accelerator]
            * accelerator_count,
            learning_rate=2e-5,
            max_num_epochs=10,
            warmup_steps=128,
            checkpoint_every_steps=256,
            eval_every_ref_batch=256,
            **base_config,
            **mae_config,
        )
    case ("arc-solver", "small"):
        num_solution_attempts = 4
        training_config = ArcSolverConfig(
            seed=43,
            batch_size=1024 if accelerator != "cpu" else 128,
            base_cell_cost=dict(L4=0, v6e=64, cpu=0)[accelerator],
            minibatch_size=dict(L4=64, v5e=24, v6e=64, cpu=16)[accelerator]
            * accelerator_count,
            eval_batch_size=dict(L4=32, v6e=48, cpu=8)[accelerator]
            * accelerator_count
            // num_solution_attempts,
            learning_rate=1e-5,
            max_num_epochs=10,
            warmup_steps=128,
            checkpoint_every_steps=256,
            # on 1xv6e and 4 attempts, eval takes about 320s, and we have ~32 ex/s.
            # At batch size 256 ex/refbatch, that is 8s/refbatch.
            # Thus, eval breakeven is at 40 refbatches
            eval_every_ref_batch=512,
            num_solution_attempts=num_solution_attempts,
            **base_config,
            **arc_solver_config,
        )
    case _:
        raise NotImplementedError(f"{training=} {model_config=}")


config = Training(
    run_name=run_name,
    checkpoint_base_uri=f"gs://576e2361-arc-agi-2/checkpoints/",
    size_bins=[12, 20, 30] if accelerator != "cpu" else [30],
    model=ModelSelection(
        type=training,
        config=model_config,
    ),
    wandb_secret_name="wandb-api-key",
    starting_checkpoint=checkpoint,
    starting_checkpoint_type={"arc-solver": "with-encoder"}.get(training, training),
    **{
        MAETaskConfig: dict(mae_training=training_config),
        ArcSolverConfig: dict(arc_solver_training=training_config),
    }[type(training_config)],
)

accelerator_type = dict(
    L4="gpu",
    H100="gpu",
    RTX6000="gpu",
    v6e="tpu",
    cpu="cpu",
)[accelerator]

# see https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus

match accelerator_type:
    case "tpu":
        kw = dict(
            tpu_topology={
                1: "1x1",
                4: "2x2",
                8: "2x4",
            }[accelerator_count],
            boot_disk_type="hyperdisk-balanced",  # needed for v6e
        )
    case "gpu":
        kw = (
            dict(
                scheduling_strategy=aiplatform_v1.types.custom_job.Scheduling.Strategy.SPOT,
            )
            if use_spot
            else dict()
        )
    case "cpu":
        kw = dict()
    case _:
        raise KeyError(accelerator_type)

tune_task = (
    ["tune-batch-size"] + [f"--resolution=0.01"] + [f"--image-sizes={n}" for n in [20]]
)


def to_json(obj):
    if nested := _get_config_class(type(obj)):
        return {f.name: to_json(getattr(obj, f.name)) for f in _get_fields(nested)}
    match obj:
        case dict():
            return {k: to_json(v) for k, v in obj.items()}
        case tuple() | list():
            return type(obj)(to_json(v) for v in obj)
        case set() | frozenset():
            return list(to_json(v) for v in obj)
        case _:
            return obj


pretrain_task = ["train", "--config-json", json5.dumps(to_json(config))]
print(pretrain_task)

args = pretrain_task

env = dict(
    XLA_PYTHON_CLIENT_MEM_FRACTION="1.00",
    GCP_PROJECT_ID="deep-time-358505",
    JAX_COMPILATION_CACHE_DIR="/gcs/41bd4de0-jax-cache",
    #    JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all",
    #    JAX_LOG_COMPILES="1", # attention: huge amount of logs
)

if dry_run:
    # TODO: we should probably launch docker instead !?
    subprocess.check_call(
        [sys.executable, "-m", "arc25.training.cli"] + args,
        env=os.environ | env,
    )
    sys.exit(0)

job = aiplatform.CustomContainerTrainingJob(
    display_name=run_name,
    location="europe-west4",
    project="deep-time-358505",
    container_uri=f"europe-west4-docker.pkg.dev/deep-time-358505/arc-agi/arc25-{accelerator_type}:latest",
    staging_bucket="gs://576e2361-arc-agi-2",
)

job.run(
    service_account="arc-agi-2-training@deep-time-358505.iam.gserviceaccount.com",
    machine_type=dict(
        v6e=f"ct6e-standard-{accelerator_count}t",
        L4=f"g2-standard-{12*accelerator_count}",
        H100="a3-highgpu-1g",
        RTX6000="g4-standard-48",
    )[accelerator],
    accelerator_type=dict(
        L4="NVIDIA_L4",
        H100="NVIDIA_H100_80GB",
        RTX6000="NVIDIA_RTX_PRO_6000",
    ).get(accelerator, "ACCELERATOR_TYPE_UNSPECIFIED"),
    accelerator_count=accelerator_count,
    sync=True,
    args=args,
    environment_variables=env,
    #   tensorboard="projects/440754660445/locations/europe-west4/tensorboards/4532002211739205632",
    **kw,
)
