#!/bin/bash
set -e

# Parse arguments
BUILD_TARGET=${1:-training}  # base, training, or both
ACCELERATOR=${2:-gpu}
IMAGE_NAME=${3:-arc25}
IMAGE_TAG=${4:-latest}
PUSH_TO_GCP=${5:-false}

# GCP configuration with defaults
GCP_PROJECT_ID=${GCP_PROJECT_ID:-deep-time-358505}
GCP_REGION=${GCP_REGION:-europe-west4}
GCP_REPOSITORY=${GCP_REPOSITORY:-arc-agi}

# Platform for cross-compilation (default to amd64 for cloud deployment)
PLATFORM=${PLATFORM:-linux/amd64}

if [[ "$BUILD_TARGET" != "base" && "$BUILD_TARGET" != "training" && "$BUILD_TARGET" != "both" ]]; then
    echo "Usage: $0 [base|training|both] [gpu|tpu|workbench] [image_name] [image_tag] [push]"
    echo "Example: $0 base gpu arc25-base base true"
    echo "Example: $0 training gpu arc25 latest true"
    echo "Example: $0 both gpu arc25 latest true"
    echo ""
    echo "Environment variables:"
    echo "  GCP_PROJECT_ID=<project>              - GCP project ID (default: deep-time-358505)"
    echo "  GCP_REGION=<region>                   - Artifact Registry region (default: europe-west4)"
    echo "  GCP_REPOSITORY=<repo>                 - Artifact Registry repository (default: arc-agi)"
    echo "  PLATFORM=<platform>                   - Target platform (default: linux/amd64)"
    exit 1
fi

if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "tpu" && "$ACCELERATOR" != "workbench" ]]; then
    echo "Error: ACCELERATOR must be gpu, tpu, or workbench"
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

# Build base image function
build_base() {
    local accel=$1
    local base_img_name="${IMAGE_NAME}-${accel}-base"
    local img_tag=$2

    echo "=========================================="
    echo "Building BASE image for ${accel}"
    echo "=========================================="

    BUILD_ARGS=(
        --build-arg "ACCELERATOR=${accel}"
        --build-arg "PYTHON_VERSION=3.13"
        --build-arg "PYTHON_CFLAGS=-march=x86-64-v3 -mtune=generic"
    )

    if [[ "$PUSH_TO_GCP" == "true" ]]; then
        ARTIFACT_REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}"
        REMOTE_IMAGE="${ARTIFACT_REGISTRY_URL}/${base_img_name}:${img_tag}"
        REMOTE_IMAGE_LATEST="${ARTIFACT_REGISTRY_URL}/${base_img_name}:latest"

        echo "Will push to: ${REMOTE_IMAGE}"
        gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

        docker buildx build \
            --platform "${PLATFORM}" \
            "${BUILD_ARGS[@]}" \
            --push \
            -f "$DOCKER_DIR/Dockerfile.base" \
            -t "${REMOTE_IMAGE}" \
            -t "${REMOTE_IMAGE_LATEST}" \
            "$DOCKER_DIR"
    else
        docker buildx build \
            --platform "${PLATFORM}" \
            "${BUILD_ARGS[@]}" \
            --load \
            -f "$DOCKER_DIR/Dockerfile.base" \
            -t "${base_img_name}:${img_tag}" \
            -t "${base_img_name}:latest" \
            "$DOCKER_DIR"
    fi

    # Return the tag for use by training build
    echo "${img_tag}"
}

# Build training image function
build_training() {
    local accel=$1
    local img_tag=$2
    local base_tag=${3:-latest}  # Use provided base tag or default to "latest"

    # Training images are named with accelerator: arc25-gpu, arc25-tpu, etc.
    local img_name="${IMAGE_NAME}-${accel}"

    echo "=========================================="
    echo "Building TRAINING image for ${accel}"
    echo "Image name: ${img_name}"
    echo "Using base image tag: ${base_tag}"
    echo "=========================================="

    # Create build context directory
    BUILD_CONTEXT="$DOCKER_DIR/buildctxt"
    echo "Preparing build context at $BUILD_CONTEXT..."
    rm -rf "$BUILD_CONTEXT"
    mkdir -p "$BUILD_CONTEXT"

    # Export dependencies directly to build context
    echo "Exporting dependencies..."
    pdm export -G vision -G train --no-hashes -o "$BUILD_CONTEXT/requirements.txt"

    # Build wheel directly to build context (note: pdm build wipes the target directory)
    echo "Building wheel..."
    pdm build --no-sdist --dest "$BUILD_CONTEXT/dist"

    # Copy data and notebooks to build context
    echo "Copying data and notebooks to build context..."
    pdm run python -m arc25.deploy prepare-docker-context "$BUILD_CONTEXT"

    # Select dockerfile based on accelerator
    if [[ "${accel}" == "workbench" ]]; then
        DOCKERFILE="vertex-ai-workbench.dockerfile"
        BUILD_ARGS=(
            --build-arg "PYTHON_VERSION=3.13"
        )
    else
        DOCKERFILE="Dockerfile"
        ARTIFACT_REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}"
        BUILD_ARGS=(
            --build-arg "ACCELERATOR=${accel}"
            --build-arg "PYTHON_VERSION=3.13"
            --build-arg "BASE_IMAGE_REGISTRY=${ARTIFACT_REGISTRY_URL}"
            --build-arg "BASE_IMAGE_TAG=${base_tag}"
        )
    fi

    if [[ "$PUSH_TO_GCP" == "true" ]]; then
        ARTIFACT_REGISTRY_URL="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}"
        REMOTE_IMAGE="${ARTIFACT_REGISTRY_URL}/${img_name}:${img_tag}"
        REMOTE_IMAGE_LATEST="${ARTIFACT_REGISTRY_URL}/${img_name}:latest"

        echo "Will push to: ${REMOTE_IMAGE}"
        gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

        docker buildx build \
            --platform "${PLATFORM}" \
            "${BUILD_ARGS[@]}" \
            --push \
            -f "$DOCKER_DIR/${DOCKERFILE}" \
            -t "${REMOTE_IMAGE}" \
            -t "${REMOTE_IMAGE_LATEST}" \
            "$BUILD_CONTEXT"
    else
        docker buildx build \
            --platform "${PLATFORM}" \
            "${BUILD_ARGS[@]}" \
            --load \
            -f "$DOCKER_DIR/${DOCKERFILE}" \
            -t "${img_name}:${img_tag}" \
            -t "${img_name}:latest" \
            "$BUILD_CONTEXT"
    fi

    # Clean up build context
    echo "Cleaning up build context..."
    rm -rf "$BUILD_CONTEXT"
}

# Execute builds based on target
case "$BUILD_TARGET" in
    base)
        build_base "$ACCELERATOR" "$IMAGE_TAG"
        ;;
    training)
        build_training "$ACCELERATOR" "$IMAGE_TAG"
        ;;
    both)
        # Build base first and capture its tag
        base_tag=$(build_base "$ACCELERATOR" "$IMAGE_TAG")
        # Use the exact base tag just built for training image
        build_training "$ACCELERATOR" "$IMAGE_TAG" "$base_tag"
        ;;
esac

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
