# Base images for different accelerators
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/workbench-container:latest

ARG PYTHON_VERSION=3.13
ARG ENV_NAME=py-${PYTHON_VERSION}
ARG KERNEL_NAME=python-${PYTHON_VERSION}

ENV MAMBA_ROOT_PREFIX=/opt/micromamba

RUN micromamba create -n ${ENV_NAME} -c conda-forge python=${PYTHON_VERSION} -y

SHELL ["micromamba", "run", "-n", "py-3.13", "/bin/bash", "-c"]

RUN micromamba install -c conda-forge pip -y
RUN pip install ipykernel
RUN python -m ipykernel install --prefix /opt/micromamba/envs/${ENV_NAME} --name ${ENV_NAME} --display-name ${KERNEL_NAME}
# Creation of a micromamba kernel automatically creates a python3 kernel
# that must be removed if it's in conflict with the new kernel.
RUN rm -rf "/opt/micromamba/envs/${ENV_NAME}/share/jupyter/kernels/python3"

# Install Python dependencies (this layer changes rarely)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install the arc25 wheel (changes when code changes)
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
