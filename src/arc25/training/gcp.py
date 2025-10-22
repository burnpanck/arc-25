"""Google Cloud Platform integration for training."""

import os
from pathlib import Path
from typing import Literal

from google.cloud import secretmanager


def get_secret(
    secret_name: str,
    project_id: str | None = None,
    version: str | Literal["latest"] = "latest",
) -> str:
    """
    Fetch a secret from Google Secret Manager using Application Default Credentials.

    Args:
        secret_name: Name of the secret (e.g., "wandb-api-key")
        project_id: GCP project ID (defaults to GCP_PROJECT_ID env var)
        version: Secret version (defaults to "latest")

    Returns:
        The secret value as a string

    Raises:
        ValueError: If project_id is not provided and GCP_PROJECT_ID env var is not set
        google.api_core.exceptions.GoogleAPIError: If secret retrieval fails
    """
    if project_id is None:
        project_id = os.environ.get("GCP_PROJECT_ID")
        if project_id is None:
            raise ValueError(
                "project_id must be provided or GCP_PROJECT_ID environment variable must be set"
            )

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"

    response = client.access_secret_version(request=dict(name=name))
    return response.payload.data.decode("UTF-8")


def get_wandb_key(
    secret_name: str = "wandb-api-key",
    project_id: str | None = None,
) -> str:
    """
    Fetch WandB API key from Google Secret Manager.

    Args:
        secret_name: Name of the secret containing the WandB API key
        project_id: GCP project ID (defaults to GCP_PROJECT_ID env var)

    Returns:
        The WandB API key
    """
    return get_secret(secret_name, project_id)


def gcs_uri_to_path(uri: str | Path | None) -> Path | None:
    """
    Convert a GCS URI to a local filesystem path using Vertex AI's /gcs/ mount.

    In Vertex AI, Cloud Storage buckets are mounted at /gcs/, allowing direct
    filesystem access without using the Cloud Storage client library.

    Args:
        uri: Either a GCS URI (gs://bucket/path) or already a local path

    Returns:
        Path object pointing to /gcs/bucket/path (for GCS URIs) or the original path

    Examples:
        >>> gcs_uri_to_path("gs://my-bucket/checkpoints/run-001")
        Path('/gcs/my-bucket/checkpoints/run-001')
        >>> gcs_uri_to_path("/local/path")
        Path('/local/path')
    """
    match uri:
        case str():
            if uri.startswith("gs://"):
                # Convert gs://bucket/path to /gcs/bucket/path
                return Path("/gcs") / uri[5:]
            return Path(uri)
        case Path():
            return uri
        case None:
            return None
        case _:
            raise TypeError(type(uri).__name__)


def get_checkpoint_dir(
    run_name: str,
    base_uri: str | None = None,
) -> Path | None:
    """
    Get the checkpoint directory for a training run.

    For GCS URIs, converts to /gcs/ mount point path.
    For local paths, returns the path as-is.

    Args:
        run_name: Name of the training run
        base_uri: Base URI for checkpoints. Can be:
            - GCS URI (gs://bucket/path) - converted to /gcs/bucket/path
            - Local path (/path/to/dir) - used as-is
            - None (defaults to AIP_CHECKPOINT_DIR env var, or local data/checkpoints/)

    Returns:
        Path object for the checkpoint directory

    Examples:
        >>> # With Vertex AI environment variable
        >>> os.environ["AIP_CHECKPOINT_DIR"] = "gs://my-bucket/runs"
        >>> get_checkpoint_dir("run-001")
        Path('/gcs/my-bucket/runs/run-001')

        >>> # With explicit GCS URI
        >>> get_checkpoint_dir("run-001", "gs://my-bucket/checkpoints")
        Path('/gcs/my-bucket/checkpoints/run-001')

        >>> # With local path
        >>> get_checkpoint_dir("run-001", "/data/checkpoints")
        Path('/data/checkpoints/run-001')
    """
    if base_uri is None:
        # Try Vertex AI environment variable first
        base_uri = os.environ.get("AIP_CHECKPOINT_DIR")
        if base_uri is None:
            return None

    # Convert GCS URI to /gcs/ path if needed
    base_path = gcs_uri_to_path(base_uri)
    return base_path / run_name
