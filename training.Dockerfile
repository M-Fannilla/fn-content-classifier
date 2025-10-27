# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_API_KEY=1d46416e290617f0005c9b98c3592a0350c5fa01 \
    SWEEP_ID=1zw9r5wo \
    SWEEP_ITERATIONS=15
    

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
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

# Upgrade pip

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (for L4 GPU - CUDA 12.1)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Install other Python dependencies
RUN pip install timm numpy pandas Pillow scikit-learn matplotlib seaborn scipy tqdm iterative-stratification wandb weave pyyaml

RUN wandb login ${WANDB_API_KEY}
RUN echo "WANDB_API_KEY: ${WANDB_API_KEY}"
# Copy project files
COPY training .

# Create directories for outputs
RUN mkdir -p /models /outputs /wandb

# Set environment variables for Cloud Run
ENV PORT=8080

# Health check endpoint (optional, for Cloud Run)
RUN echo '#!/bin/bash\npython -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA devices: {torch.cuda.device_count()}\")"' > /app/health_check.sh && \
    chmod +x /app/health_check.sh

# Entry point for running the sweep
# WANDB_API_KEY and SWEEP_ID should be passed as environment variables
CMD ["python", "run_sweep.py"]

