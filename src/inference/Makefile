.PHONY: help build up down logs restart clean test install run

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies locally"
	@echo "  make run       - Run Gradio app locally"
	@echo "  make build     - Build Docker image"
	@echo "  make up        - Start Docker container"
	@echo "  make down      - Stop Docker container"
	@echo "  make restart   - Restart Docker container"
	@echo "  make logs      - View Docker logs"
	@echo "  make clean     - Remove containers and images"
	@echo "  make test      - Test the running app"

install:
	pip install -r requirements.txt

run:
	python gradio_app.py

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Gradio app is running at http://localhost:7860"

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker rmi fn-classifier-gradio-app 2>/dev/null || true

test:
	@echo "Testing Gradio app..."
	@curl -s http://localhost:7860 > /dev/null && echo "✓ App is running" || echo "✗ App is not running"
