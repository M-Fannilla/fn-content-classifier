# syntax=docker/dockerfile:1
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim AS downloader

# ----- Build Args -----
ARG ACCESS_TOKEN
ARG PROJECT=fannilla-dev
ARG LOCATION=europe-west4
ARG MODEL_REPOSITORY=fn-ai-models
ARG AI_MODEL_DIR=/workspace/models/onnx

ARG ACTIONS_DIR=${AI_MODEL_DIR}/actions
ARG BODYPARTS_DIR=${AI_MODEL_DIR}/bodyparts

# ----- Non-interactive token auth -----
ENV CLOUDSDK_AUTH_ACCESS_TOKEN=$ACCESS_TOKEN
ENV CLOUDSDK_CORE_PROJECT=$PROJECT

# ----- Optional useful tools -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ----- Create models dir -----
RUN mkdir -p ${ACTIONS_DIR}
RUN mkdir -p ${BODYPARTS_DIR}

# ----- Download models -----
RUN gcloud artifacts generic download \
      --project="${PROJECT}" \
      --location="${LOCATION}" \
      --repository="${MODEL_REPOSITORY}" \
      --package=classifier-actions \
      --version=current \
      --destination="${AI_MODEL_DIR}/actions"

RUN gcloud artifacts generic download \
      --project="${PROJECT}" \
      --location="${LOCATION}" \
      --repository="${MODEL_REPOSITORY}" \
      --package=classifier-bodyparts \
      --version=current \
      --destination="${AI_MODEL_DIR}/bodyparts"

# List files so we can inspect output during build logs
RUN echo "=== Downloaded model files ===" && \
    find ${AI_MODEL_DIR} -maxdepth 3 -type f -print

# Final stage just keeps the models to inspect manually
FROM alpine:3.19
WORKDIR /app
COPY --from=downloader /workspace/models/onnx ./models

CMD ["sh"]