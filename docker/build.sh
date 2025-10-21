#!/bin/bash
set -e

# Parse arguments
ACCELERATOR=${1:-gpu}
IMAGE_NAME=${2:-arc25}
IMAGE_TAG=${3:-latest}
PUSH_TO_GCP=${4:-false}

# GCP configuration with defaults
GCP_PROJECT_ID=${GCP_PROJECT_ID:-deep-time-358505}
GCP_REGION=${GCP_REGION:-europe-west4}
GCP_REPOSITORY=${GCP_REPOSITORY:-arc-agi}

# Platform for cross-compilation (default to amd64 for cloud deployment)
PLATFORM=${PLATFORM:-linux/amd64}

if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "tpu" && "$ACCELERATOR" != "workbench" ]]; then
    echo "Usage: $0 [gpu|tpu|workbench] [image_name] [image_tag] [push]"
    echo "Example: $0 gpu arc25 latest true"
    echo "Example: $0 workbench arc25-workbench latest true"
    echo ""
    echo "Environment variables:"
    echo "  GCP_PROJECT_ID=<project>              - GCP project ID (default: deep-time-358505)"
    echo "  GCP_REGION=<region>                   - Artifact Registry region (default: europe-west4)"
    echo "  GCP_REPOSITORY=<repo>                 - Artifact Registry repository (default: arc-agi)"
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

# Create build context directory
BUILD_CONTEXT="$DOCKER_DIR/buildctxt"
echo "Preparing build context at $BUILD_CONTEXT..."
rm -rf "$BUILD_CONTEXT"
mkdir -p "$BUILD_CONTEXT"

# Export dependencies directly to build context
echo "Exporting dependencies..."
pdm export -G vision --prod --no-hashes -o "$BUILD_CONTEXT/requirements.txt"

# Build wheel directly to build context (note: pdm build wipes the target directory)
echo "Building wheel..."
pdm build --no-sdist --dest "$BUILD_CONTEXT/dist"

# Copy data and notebooks to build context
echo "Copying data and notebooks to build context..."
pdm run python -m arc25.deploy prepare-docker-context "$BUILD_CONTEXT"

# Select dockerfile and build args based on accelerator
if [[ "$ACCELERATOR" == "workbench" ]]; then
    DOCKERFILE="vertex-ai-workbench.dockerfile"
    BUILD_ARGS=(
        --build-arg "PYTHON_VERSION=3.13"
    )
    echo "Building Vertex AI Workbench image for platform ${PLATFORM}..."
else
    DOCKERFILE="Dockerfile"
    BUILD_ARGS=(
        --build-arg "ACCELERATOR=$ACCELERATOR"
        --build-arg "PYTHON_VERSION=3.13"
        --build-arg "PYTHON_CFLAGS=-march=x86-64 -mtune=generic"
    )
    echo "Building Docker image for platform ${PLATFORM}..."
    echo "Note: Local builds use basic x86-64 for QEMU compatibility (no AVX optimizations)"
fi

# Configure GCP and prepare tags if pushing
if [[ "$PUSH_TO_GCP" == "true" ]]; then
    ARTIFACT_REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}"
    REMOTE_IMAGE="${ARTIFACT_REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}"
    REMOTE_IMAGE_LATEST="${ARTIFACT_REGISTRY_URL}/${IMAGE_NAME}:${ACCELERATOR}"

    echo ""
    echo "Will push to Google Artifact Registry: ${ARTIFACT_REGISTRY_URL}"

    # Configure docker for Artifact Registry
    gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

    # Build and push directly to avoid --load issues with cross-platform builds
    echo ""
    echo "Building and pushing image..."
    set -x
    docker buildx build \
        --platform "${PLATFORM}" \
        "${BUILD_ARGS[@]}" \
        --push \
        -f "$DOCKER_DIR/${DOCKERFILE}" \
        -t "${REMOTE_IMAGE}" \
        -t "${REMOTE_IMAGE_LATEST}" \
        "$BUILD_CONTEXT"
    set +x
    echo ""
    echo "Successfully pushed to Artifact Registry!"
    echo "Image: ${REMOTE_IMAGE}"
    echo "Image: ${REMOTE_IMAGE_LATEST}"
else
    # Local build only - try to load into local docker daemon
    echo ""
    echo "Building image locally..."
    set -x
    docker buildx build \
        --platform "${PLATFORM}" \
        "${BUILD_ARGS[@]}" \
        --load \
        -f "$DOCKER_DIR/${DOCKERFILE}" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}" \
        -t "${IMAGE_NAME}:${ACCELERATOR}" \
        "$BUILD_CONTEXT"
    set +x

    # Verify the image was created
    echo ""
    echo "Verifying image was created..."
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}"

    if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}" >/dev/null 2>&1; then
        echo "ERROR: Image ${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR} was not created!"
        exit 1
    fi

    echo ""
    echo "Build complete!"
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}-${ACCELERATOR}"
    echo "Image: ${IMAGE_NAME}:${ACCELERATOR}"
fi

# Clean up build context
echo ""
echo "Cleaning up build context..."
rm -rf "$BUILD_CONTEXT"
