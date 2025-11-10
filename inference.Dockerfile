FROM python:3.11.7-slim-bullseye AS builder

WORKDIR /app

COPY /src /app/src
COPY /pyproject.toml /app/pyproject.toml

# ----- Install venv and others -----
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
 && rm -rf /var/lib/apt/lists/*

# ----- Create virtual environment -----
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/venv

# ----- Install dependencies -----
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir ".[inference]"

# ----- Clean up -----
RUN find . -type d -name __pycache__ -exec rm -r {} +

# Model Image
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim AS model-imaage

# Release Stage
FROM python:3.11.7-slim-bullseye AS release

WORKDIR /app

COPY --from=builder /app /app
COPY --from=model-downloader /app/models /app/models
ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/venv

ENTRYPOINT ["python", "-m", "classifier.inference.main"]


# docker buildx build \
#  --mount type=bind,source="$HOME/.config/gcloud",target=/root/.config/gcloud,readonly \
#  -f inference.Dockerfile \
#  --platform=linux/amd64 \
#  -t my-image .