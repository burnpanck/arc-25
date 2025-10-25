"""Google Cloud Platform integration for training."""

import io
import os
from typing import Literal

import etils.epath


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
    from google.cloud import secretmanager

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


def save_and_upload_to_gcs(save_fn, gcs_path: str, filename: str) -> None:
    """
    Save a checkpoint to GCS using the Cloud Storage client library.

    Temporary function kept for compatibility verification.
    TODO: Remove once saving infrastructure is verified to work with etils.epath.

    Args:
        save_fn: Callable that accepts a file-like object and writes checkpoint data
        gcs_path: GCS URI (gs://bucket/path)
        filename: Name of the file being uploaded (for logging)

    Raises:
        ValueError: If gcs_path is not a GCS URI
    """
    from google.cloud import storage

    gcs_uri = str(gcs_path)
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Path must be a GCS URI (gs://), got: {gcs_uri}")

    # Parse bucket and blob path from gs:// URI
    bucket_name, blob_path = gcs_uri[5:].split("/", 1)

    # Save to BytesIO buffer
    buffer = io.BytesIO()
    save_fn(buffer)
    buffer.seek(0)  # Reset to beginning for upload

    # Upload using Cloud Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    print(f"Uploading {filename} to {gcs_uri}...")
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    print(f"Upload complete: {gcs_uri}")
