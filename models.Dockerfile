FROM python:3.11.7-slim-bullseye

WORKDIR /app
COPY ./models/onnx /app/models/onnx