# syntax=docker/dockerfile:1.6

FROM gcr.io/google.com/cloudsdktool/cloud-sdk:slim AS model-downloader

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy ADC -> /gcloud
RUN --mount=type=bind,from=gcloud_config,source=.,target=/gcloud_ro,readonly \
    echo "→ Copying ADC config to /gcloud" && \
    mkdir -p /gcloud && cp -r /gcloud_ro/* /gcloud/

# ✅ Make sure ALL future gcloud calls use it
ENV CLOUDSDK_CONFIG=/gcloud

# Now these commands will work
RUN gcloud auth list && \
    gcloud config set project fannilla-dev && \
    gcloud config set artifacts/location europe-west4 && \
    gcloud config set artifacts/repository fn-ai-models

COPY Makefile Makefile

# ✅ This now uses the correct auth config
RUN make download-ai-models

CMD ["sh"]

#DOCKER_BUILDKIT=1 docker buildx build \
#  --build-arg CLOUDSDK_CONFIG=$HOME/.config/gcloud \
#  --platform=linux/amd64 \
#  -t model-downloader
#  -f gcloud.Dockerfile
#  .