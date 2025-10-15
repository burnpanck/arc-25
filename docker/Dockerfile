ARG ACCELERATOR=gpu
ARG PYTHON_VERSION=3.13

# Base images for different accelerators
FROM python:${PYTHON_VERSION}-slim AS base-gpu
ENV ACCELERATOR=gpu

FROM python:${PYTHON_VERSION}-slim AS base-tpu
ENV ACCELERATOR=tpu

# Select the appropriate base
FROM base-${ACCELERATOR} AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (this layer changes rarely)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install the arc25 wheel (changes when code changes)
COPY *.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy data and notebooks
COPY data/ /app/data/
COPY notebooks/ /app/notebooks/

# Install Jupyter for Vertex AI Workbench
RUN pip install --no-cache-dir jupyter jupyterlab

# Expose Jupyter port
EXPOSE 8080

# Default command for Jupyter (can be overridden for training jobs)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
