# Makefile for building and pushing Docker image to Google Cloud Artifact Registry

# Variables
PROJECT_ID := fannilla-dev
REGION := europe-west1
REPOSITORY := shared-repository
IMAGE_NAME := content-classifier
TAG := training
REGISTRY := $(REGION)-docker.pkg.dev
FULL_IMAGE_NAME := $(REGISTRY)/$(PROJECT_ID)/$(REPOSITORY)/$(IMAGE_NAME):$(TAG)
LATEST_IMAGE_NAME := $(REGISTRY)/$(PROJECT_ID)/$(REPOSITORY)/$(IMAGE_NAME):latest

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: help configure build tag push build-and-push clean

# Help target
help:
	@echo "Available targets:"
	@echo "  make configure       - Configure Docker to authenticate with GCP Artifact Registry"
	@echo "  make build          - Build the Docker image"
	@echo "  make tag            - Tag the image with latest"
	@echo "  make push           - Push the Docker image to Artifact Registry"
	@echo "  make build-and-push - Build and push the Docker image in one command"
	@echo "  make clean          - Remove local Docker images"
	@echo ""
	@echo "Image will be pushed to: $(FULL_IMAGE_NAME)"

# Configure Docker to authenticate with GCP Artifact Registry
configure:
	@echo "Configuring Docker authentication for GCP Artifact Registry..."
	gcloud auth configure-docker $(REGISTRY)

# Build the Docker image
build:
	@echo "Building Docker image: $(FULL_IMAGE_NAME)"
	docker build -t $(FULL_IMAGE_NAME) .
	@echo "Build complete!"

# Tag the image with latest
tag:
	@echo "Tagging image as latest..."
	docker tag $(FULL_IMAGE_NAME) $(LATEST_IMAGE_NAME)
	@echo "Tagged as $(LATEST_IMAGE_NAME)"

# Push the Docker image to Artifact Registry
push: tag
	@echo "Pushing Docker image to Artifact Registry..."
	docker push $(FULL_IMAGE_NAME)
	@echo "Push complete!"
	@echo "Image available at: $(FULL_IMAGE_NAME)"

# Build and push in one command
build-and-push: build push

# Build, tag, and push both training and latest tags
all: build push push-latest

# Clean up local Docker images
clean:
	@echo "Removing local Docker images..."
	-docker rmi $(FULL_IMAGE_NAME)
	-docker rmi $(LATEST_IMAGE_NAME)
	@echo "Cleanup complete!"

# Show image info
info:
	@echo "Project ID:    $(PROJECT_ID)"
	@echo "Region:        $(REGION)"
	@echo "Repository:    $(REPOSITORY)"
	@echo "Image Name:    $(IMAGE_NAME)"
	@echo "Tag:           $(TAG)"
	@echo "Full Image:    $(FULL_IMAGE_NAME)"
	@echo "Latest Image:  $(LATEST_IMAGE_NAME)"

