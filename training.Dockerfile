# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG SWEEP_ID

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_API_KEY=1d46416e290617f0005c9b98c3592a0350c5fa01 \
    SWEEP_ID=${SWEEP_ID} \
    SWEEP_ITERATIONS=15
    

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

COPY . .
RUN rm -rf /src/inference /src/notebooks

RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first
RUN chmod +x /app/scripts/install_pytorch.sh
RUN ./app/install_pytorch.sh

# Install other Python dependencies
RUN pip ".[train]"

RUN wandb login ${WANDB_API_KEY}
RUN echo "WANDB_API_KEY: ${WANDB_API_KEY}"

# Health check endpoint
RUN chmod +x /app/scripts/torch_check.sh
RUN ./app/torch_check.sh && echo "Health check passed"

# Entry point for running the sweep
# WANDB_API_KEY and SWEEP_ID should be passed as environment variables
CMD ["python", "run_sweep.py"]

