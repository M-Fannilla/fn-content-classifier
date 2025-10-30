INFERENCE_IMAGE_NAME?=classifier


build-inference:
	docker build . -t $(INFERENCE_IMAGE_NAME) -f inference.Dockerfile

inference-run:
	docker run $(INFERENCE_IMAGE_NAME)