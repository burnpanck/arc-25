from google.cloud import aiplatform, aiplatform_v1

accelerator = "L4"

accelerator_type = dict(
    L4="gpu",
    H100="gpu",
    RTX6000="gpu",
    v6e="tpu",
)[accelerator]

job = aiplatform.CustomContainerTrainingJob(
    display_name=f"arc-agi-vision2-batch-tuning-{accelerator}-maxmem",
    location="europe-west4",
    project="deep-time-358505",
    container_uri=f"europe-west4-docker.pkg.dev/deep-time-358505/arc-agi/arc25:{accelerator_type}",
    staging_bucket="gs://576e2361-arc-agi-2",
)

match accelerator_type:
    case "tpu":
        kw = dict(
            tpu_topology="1x1",
            boot_disk_type="hyperdisk-balanced",  # needed for v6e
        )
    case "gpu":
        kw = dict(
            scheduling_strategy=aiplatform_v1.types.custom_job.Scheduling.Strategy.SPOT,
        )
    case _:
        raise KeyError(accelerator_type)

job.run(
    service_account="arc-agi-2-training@deep-time-358505.iam.gserviceaccount.com",
    machine_type=dict(
        v6e="ct6e-standard-1t",
        L4="g2-standard-12",
        H100="a3-highgpu-1g",
        RTX6000="g4-standard-48",
    )[accelerator],
    accelerator_type=dict(
        L4="NVIDIA_L4",
        H100="NVIDIA_H100_80GB",
        RTX6000="NVIDIA_RTX_PRO_6000",
    ).get(accelerator, "ACCELERATOR_TYPE_UNSPECIFIED"),
    accelerator_count=1,
    sync=True,
    args=["tune-batch-size"]
    + [f"--resolution=0.01"]
    + [f"--image-sizes={n}" for n in [20]],
    environment_variables=dict(
        XLA_PYTHON_CLIENT_MEM_FRACTION="1.00",
    ),
    #   tensorboard="projects/440754660445/locations/europe-west4/tensorboards/4532002211739205632",
    **kw,
)
