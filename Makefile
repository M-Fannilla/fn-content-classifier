ENVIRONMENT ?= dev
PROJECT ?= fannilla-$(ENVIRONMENT)
TAG ?= fn-content-classifier:latest
ARTIFACT_IMAGE ?= europe-docker.pkg.dev/$(PROJECT)/shared-repository/$(TAG)

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


# 495mb old
#