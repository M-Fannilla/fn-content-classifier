FROM python:3.11.7-slim-bullseye AS builder

WORKDIR /app

COPY /src /app/src
COPY /pyproject.toml /app/pyproject.toml

# Install system dependencies and clean up in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/venv

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir ".[inference]"
RUN find . -type d -name __pycache__ -exec rm -r {} +

FROM python:3.11.7-slim-bullseye AS release

WORKDIR /app

COPY --from=builder /app /app
COPY /models /app/models
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/venv

ENTRYPOINT ["python", "-m", "classifier.inference.main"]