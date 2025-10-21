from google.cloud import aiplatform, aiplatform_v1

job = aiplatform.CustomContainerTrainingJob(
    display_name="arc-agi-vision2-batch-tuning-tpu-v6e",
    location="europe-west4",
    project="deep-time-358505",
    container_uri="europe-west4-docker.pkg.dev/deep-time-358505/arc-agi/arc25@sha256:9e9e6dead23ccbcb2a6a0f5bb3bc6d500dae3f05b2981624fa193b17823074fd",
    staging_bucket="gs://576e2361-arc-agi-2",
)

job.run(
    service_account="arc-agi-2-training@deep-time-358505.iam.gserviceaccount.com",
    machine_type="ct6e-standard-1t",
    tpu_topology="1x1",
    boot_disk_type="hyperdisk-balanced",  # needed for v6e
    accelerator_count=1,
    args=["tune-batch-size"] + [f"--image-sizes={n}" for n in [12, 16, 20, 24, 30]],
    #   tensorboard="projects/440754660445/locations/europe-west4/tensorboards/4532002211739205632",
    #   scheduling_strategy=aiplatform_v1.types.custom_job.Scheduling.Strategy.SPOT,
)
