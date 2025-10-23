import datetime
import os
import subprocess
import sys

import json5
from google.cloud import aiplatform, aiplatform_v1

from arc25.lib.click_tools import _get_fields, _is_config_class
from arc25.training.cli import ModelSelection, Pretraining
from arc25.training.mae import MAETaskConfig

dry_run = False

now = datetime.datetime.now().astimezone(datetime.timezone.utc)

accelerator = "L4"
accelerator_count = 4

model_config = "tiny"
run_name = (
    f"{now:%Y%m%d-%H%M}-vertex-ai-mae-{model_config}-{accelerator_count}x{accelerator}"
)
print(f"Run: {run_name}")

config = Pretraining(
    run_name=run_name,
    size_bins=[12, 20, 30],
    model=ModelSelection(
        config=model_config,
    ),
    training=MAETaskConfig(
        seed=42,
        batch_size=512,
        minibatch_size=128 * accelerator_count,
        reference_image_size=15,
        base_cell_cost=10,
        ref_batch=256,
        learning_rate=1e-5,
        max_num_epochs=5,
        max_num_ref_batches=None,
        warmup_steps=64,
        checkpoint_every_steps=256,
        knn_validation_every_ref_batch=128,  # eval dataset is about 24 reference batches big
        mode="flat",
        remat=True,
        unroll=None,
        test_ratio=0.25,
        nonmask_fraction=0.2,
        randomise_fraction=0.2,
    ),
    wandb_secret_name="wandb-api-key",
)

accelerator_type = dict(
    L4="gpu",
    H100="gpu",
    RTX6000="gpu",
    v6e="tpu",
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
        kw = dict(
            scheduling_strategy=aiplatform_v1.types.custom_job.Scheduling.Strategy.SPOT,
        )
    case _:
        raise KeyError(accelerator_type)

tune_task = (
    ["tune-batch-size"] + [f"--resolution=0.01"] + [f"--image-sizes={n}" for n in [20]]
)


def to_json(obj):
    if _is_config_class(type(obj)):
        return {f.name: to_json(getattr(obj, f.name)) for f in _get_fields(type(obj))}
    match obj:
        case dict():
            return {k: to_json(v) for k, v in obj.items()}
        case tuple() | list():
            return type(obj)(to_json(v) for v in obj)
        case set() | frozenset():
            return list(to_json(v) for v in obj)
        case _:
            return obj


pretrain_task = ["full-pretraining", "--config-json", json5.dumps(to_json(config))]
print(pretrain_task)

args = pretrain_task

env = dict(
    XLA_PYTHON_CLIENT_MEM_FRACTION="1.00",
    GCP_PROJECT_ID="deep-time-358505",
    #    JAX_LOG_COMPILES="1", # attention: huge amount of logs
)

if dry_run:
    # TODO: we should probably launch docker instead !?
    subprocess.check_call(
        [sys.executable, "-m", "arc25.training.cli"] + args,
        env=dict(**os.environ, **env),
    )
    sys.exit(0)

job = aiplatform.CustomContainerTrainingJob(
    display_name=run_name,
    location="europe-west4",
    project="deep-time-358505",
    container_uri=f"europe-west4-docker.pkg.dev/deep-time-358505/arc-agi/arc25:{accelerator_type}",
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
