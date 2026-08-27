IMAGE_NAME := neumonia
IMAGE_TAG  := 1.0.0
MODEL_DIR  := $(CURDIR)/models

.PHONY: install lint format test test-all smoke check-warnings run \
        docker-build docker-run docker-run-linux verify clean

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest -m "not requires_model and not gui" -q

test-all:
	uv run pytest -q

smoke:
	uv run python scripts/smoke_test.py --image tests/data/sample_synthetic.dcm --dry-run

check-warnings:
	uv run python scripts/check_warnings.py

run:
	uv run python src/detector_neumonia.py

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run --rm -v $(MODEL_DIR):/app/models:ro $(IMAGE_NAME):$(IMAGE_TAG) \
		python scripts/smoke_test.py --image tests/data/sample_synthetic.dcm

docker-run-linux:
	xhost +local:docker
	docker run --rm -e DISPLAY=$$DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(MODEL_DIR):/app/models:ro \
		$(IMAGE_NAME):$(IMAGE_TAG)

verify: lint
	uv run ruff format --check .
	$(MAKE) test
	$(MAKE) check-warnings

clean:
	rm -rf .venv .pytest_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name "__pycache__" -exec rm -rf {} +