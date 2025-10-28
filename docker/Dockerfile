ARG BASE_IMAGE_REGISTRY=europe-west4-docker.pkg.dev/deep-time-358505/arc-agi
ARG BASE_IMAGE_TAG=latest
ARG ACCELERATOR=gpu

# Use pre-built base image with pyenv and Python
FROM ${BASE_IMAGE_REGISTRY}/arc25-${ACCELERATOR}-base:${BASE_IMAGE_TAG}

ARG ARC25_APP_ROOT="/app"
ENV ARC25_APP_ROOT="${ARC25_APP_ROOT}"

# Set working directory
WORKDIR "${ARC25_APP_ROOT}"

# Set entrypoint based on accelerator
# GPU uses nvidia entrypoint, TPU uses python directly
RUN if [ "$ACCELERATOR" = "gpu" ]; then \
        printf '#!/bin/sh\nexec /opt/nvidia/nvidia_entrypoint.sh python -m arc25.training.cli "$@"\n' > "${ARC25_APP_ROOT}/entrypoint.sh"; \
    else \
        printf '#!/bin/sh\nexec python -m arc25.training.cli "$@"\n' > "${ARC25_APP_ROOT}/entrypoint.sh"; \
    fi && chmod +x "${ARC25_APP_ROOT}/entrypoint.sh"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

# Install Python dependencies (this layer changes rarely)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "jax[${JAX_EXTRAS}]" -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install the arc25 wheel (changes when code changes)
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy data and notebooks (prepared by build script via arc25.deploy prepare-docker-context)
COPY data/ "${ARC25_APP_ROOT}/data/"
COPY notebooks/ "${ARC25_APP_ROOT}/notebooks/"
