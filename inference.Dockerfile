FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and clean up in single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY inference/requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference/ /app/inference/

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 7860