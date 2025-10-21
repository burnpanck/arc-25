ARG ACCELERATOR=gpu
ARG PYTHON_VERSION=3.13

# Base images for different accelerators
FROM nvcr.io/nvidia/cuda-dl-base:25.09-cuda13.0-devel-ubuntu24.04 AS base-gpu
ENV ACCELERATOR=gpu
ENV JAX_EXTRAS=cuda13-local

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python", "-m", "arc25.training.cli"]
CMD []

FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.8.1_3.13_tpuvm AS base-tpu
ENV ACCELERATOR=tpu
ENV JAX_EXTRAS=tpu

ENTRYPOINT ["python", "-m", "arc25.training.cli"]
CMD []

# Select the appropriate base
FROM base-${ACCELERATOR}
ARG PYTHON_VERSION

# Set working directory
WORKDIR /root

# Install system dependencies for building Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN curl https://pyenv.run | bash

# Performance-optimized Python build flags (can be overridden via build args)
ARG PYTHON_CONFIGURE_OPTS="--enable-optimizations --with-lto --enable-shared"
ARG PYTHON_CFLAGS="-march=x86-64-v3 -mtune=generic"
# Faster PGO build (comment out for maximum performance, use full test suite)
ARG PROFILE_TASK="-m test.regrtest --pgo -j0"


# Build and install Python
RUN FULL_VERSION=$(pyenv install --list | grep -E "^\s*${PYTHON_VERSION}\.[0-9]+$" | tail -1 | tr -d ' ') && \
    echo "PYTHON_VERSION=\"${PYTHON_VERSION}\", FULL_VERSION=\"${FULL_VERSION}\"" && \
    PYTHON_CONFIGURE_OPTS="${PYTHON_CONFIGURE_OPTS}" PYTHON_CFLAGS="${PYTHON_CFLAGS}" PROFILE_TASK="${PROFILE_TASK}" pyenv install ${FULL_VERSION} && \
    pyenv global ${FULL_VERSION} && \
    pyenv rehash

# Install Python dependencies (this layer changes rarely)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir "jax[${JAX_EXTRAS}]" -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install the arc25 wheel (changes when code changes)
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

ARG ARC25_APP_ROOT="/app"
ENV ARC25_APP_ROOT="${ARC25_APP_ROOT}"
# Set working directory
WORKDIR "${ARC25_APP_ROOT}"

# Copy data and notebooks (prepared by build script via arc25.deploy prepare-docker-context)
COPY data/ "${ARC25_APP_ROOT}/data/"
COPY notebooks/ "${ARC25_APP_ROOT}/notebooks/"
