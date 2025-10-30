FROM python:3.11.7-slim-bullseye AS builder

WORKDIR /app

# Install system dependencies and clean up in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[inference]"

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "-m", "classifier.inference.main"]