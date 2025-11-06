# Project and Artifact Configuration
# ==================================
ENVIRONMENT ?= dev
PROJECT ?= fannilla-$(ENVIRONMENT)
LOCATION ?= europe-west4
TAG ?= fn-content-classifier:latest
ARTIFACT_IMAGE ?= europe-docker.pkg.dev/$(PROJECT)/shared-repository/$(TAG)
# ==================================

# AI MODELS AND SPECIFICATIONS
# ==================================
LATEST_VERSION ?= current
AI_MODEL_DIR ?= models/onnx
MODEL_REPOSITORY ?= fn-ai-models
MODEL_VERSION ?= $(shell date +%Y%m%d-%H%M)
# ==================================

build-inference:
	docker build \
		--platform=linux/amd64 \
		-t ${TAG} \
		-f inference.Dockerfile \
		--progress=plain \
		.

tag-inference:
	docker tag $(TAG) $(ARTIFACT_IMAGE)

push-to-gcr: build-inference tag-inference
	docker push $(ARTIFACT_IMAGE)

inference-run:
	docker run $(TAG)

define __delete_version_if_exists__
	@if gcloud artifacts versions list \
		--project=$(PROJECT) \
		--location=$(LOCATION) \
		--repository=$(MODEL_REPOSITORY) \
		--package=classifier-$(1) \
		--format="value(VERSION)" | grep -q "^$(LATEST_VERSION)$$"; then \
		echo "Deleting existing version: classifier-$(1):$(LATEST_VERSION)"; \
		gcloud artifacts versions delete $(LATEST_VERSION) \
			--project=$(PROJECT) \
			--location=$(LOCATION) \
			--repository=$(MODEL_REPOSITORY) \
			--package=classifier-$(1) \
			--quiet; \
	else \
		echo "No existing '$(LATEST_VERSION)' version for classifier-$(1), skipping delete"; \
	fi
endef

define push-to-ai-repo
	gcloud artifacts generic upload \
		--project=$(PROJECT) \
		--source-directory=$(AI_MODEL_DIR)/onnx/$(1) \
		--location=$(LOCATION) \
		--repository=$(MODEL_REPOSITORY) \
		--package=classifier-$(1) \
		--version=$(MODEL_VERSION) \
		--source-directory=models/onnx/$(1)

	$(call __delete_version_if_exists__,$(1))

	gcloud artifacts generic upload \
		--project=$(PROJECT) \
		--source-directory=$(AI_MODEL_DIR)/onnx/$(1) \
		--location=$(LOCATION) \
		--repository=$(MODEL_REPOSITORY) \
		--package=classifier-$(1) \
		--version=$(LATEST_VERSION) \
		--source-directory=models/onnx/$(1)
endef

push-ai-models:
	$(call push-to-ai-repo,actions)
	$(call push-to-ai-repo,bodyparts)

define gcp-models-download
	dest="$(AI_MODEL_DIR)/$(1)"; \
	mkdir -p "$$dest"; \
	gcloud artifacts generic download \
	  --project="$(PROJECT)" \
	  --location="$(LOCATION)" \
	  --repository="$(MODEL_REPOSITORY)" \
	  --package="classifier-$(1)" \
	  --version=current \
	  --destination="$$dest";
endef

download-ai-models:
	$(call gcp-models-download,actions)
	$(call gcp-models-download,bodyparts)

#ACCESS_TOKEN=$(gcloud auth print-access-token)
#DOCKER_BUILDKIT=1 docker build \
#  --build-arg ACCESS_TOKEN="$ACCESS_TOKEN" \
#  --platform=linux/amd64 \
#  -f gcloud.Dockerfile \
#  -t test-download-models .