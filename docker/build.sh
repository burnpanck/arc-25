#!/bin/bash
set -e

# Parse arguments
ACCELERATOR=${1:-gpu}
IMAGE_NAME=${2:-arc25}
IMAGE_TAG=${3:-latest}
PUSH_TO_GCP=${4:-false}

# GCP configuration with defaults
GCP_PROJECT_ID=${GCP_PROJECT_ID:-arc-prize-2025}
GCP_REGION=${GCP_REGION:-europe-west4}
GCP_REPOSITORY=${GCP_REPOSITORY:-arc25}

# Platform for cross-compilation (default to amd64 for cloud deployment)
PLATFORM=${PLATFORM:-linux/amd64}

if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "tpu" ]]; then
    echo "Usage: $0 [gpu|tpu] [image_name] [image_tag] [push]"
    echo "Example: $0 gpu arc25 latest true"
    echo ""
    echo "Environment variables:"
    echo "  GCP_PROJECT_ID=<project>              - GCP project ID (default: arc-prize-2025)"
    echo "  GCP_REGION=<region>                   - Artifact Registry region (default: europe-west4)"
    echo "  GCP_REPOSITORY=<repo>                 - Artifact Registry repository (default: arc25)"
    echo "  PLATFORM=<platform>                   - Target platform (default: linux/amd64)"
    exit 1
fi

# Get the project root (parent of docker/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

echo "Building for accelerator: $ACCELERATOR"
echo "Project root: $PROJECT_ROOT"
echo "Docker context: $DOCKER_DIR"

# Change to project root for PDM commands
cd "$PROJECT_ROOT"

# Export dependencies to requirements.txt
echo "Exporting dependencies..."
pdm export --prod --no-hashes -o "$DOCKER_DIR/requirements.txt"

# Build wheel
echo "Building wheel..."
pdm build --no-sdist --dest "$DOCKER_DIR/dist"

# Build Docker image
cd "$DOCKER_DIR"
echo "Building Docker image for platform ${PLATFORM}..."
echo "Note: Local builds use basic x86-64 for QEMU compatibility (no AVX optimizations)"
docker buildx build \
    --platform "${PLATFORM}" \
    --build-arg ACCELERATOR="$ACCELERATOR" \
    --build-arg PYTHON_VERSION=3.13 \
    --build-arg PYTHON_CFLAGS="-march=x86-64 -mtune=generic" \
    --load \
    -t "${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}" \
    -t "${IMAGE_NAME}:${ACCELERATOR}" \
    .

# Clean up generated files and symlinks (but NOT Dockerfile or build.sh!)
echo "Cleaning up..."
rm -f requirements.txt
rm -f *.whl

echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}"
echo "Image: ${IMAGE_NAME}:${ACCELERATOR}"

# Optionally push to Google Artifact Registry
if [[ "$PUSH_TO_GCP" == "true" ]]; then
    ARTIFACT_REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}"
    REMOTE_IMAGE="${ARTIFACT_REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}"
    REMOTE_IMAGE_LATEST="${ARTIFACT_REGISTRY_URL}/${IMAGE_NAME}:${ACCELERATOR}"

    echo ""
    echo "Pushing to Google Artifact Registry..."
    echo "Registry: ${ARTIFACT_REGISTRY_URL}"

    # Configure docker for Artifact Registry
    gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

    # Tag for remote
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}" "${REMOTE_IMAGE}"
    docker tag "${IMAGE_NAME}:${ACCELERATOR}" "${REMOTE_IMAGE_LATEST}"

    # Push
    echo "Pushing ${REMOTE_IMAGE}..."
    docker push "${REMOTE_IMAGE}"
    echo "Pushing ${REMOTE_IMAGE_LATEST}..."
    docker push "${REMOTE_IMAGE_LATEST}"

    echo ""
    echo "Successfully pushed to Artifact Registry!"
    echo "Image: ${REMOTE_IMAGE}"
    echo "Image: ${REMOTE_IMAGE_LATEST}"
fi
