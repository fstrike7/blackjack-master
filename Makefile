SHELL := /bin/bash

PYTHON ?= python
PKG_MANAGER ?= uv
IMAGE ?= blackjack-learner
EPISODES ?= 1000

ifeq ($(PKG_MANAGER),pip)
	INSTALL_DEPS = $(PYTHON) -m pip install -r requirements.txt
	INSTALL_DEV  = $(PYTHON) -m pip install -r requirements.txt && \
	               $(PYTHON) -m pip install -r requirements-dev.txt
else
	PKG_MANAGER := uv
	INSTALL_DEPS = uv pip sync pyproject.toml
	INSTALL_DEV  = uv pip sync pyproject.toml && uv pip install -r requirements-dev.txt
endif

.PHONY: install install-dev test format lint run train visualize play docker-build docker-run compose-up compose-down clean

install:
	@echo "Installing dependencies with $(PKG_MANAGER)..."
	$(INSTALL_DEPS)

install-dev:
	@echo "Installing dependencies (including dev) with $(PKG_MANAGER)..."
	$(INSTALL_DEV)

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest

format:
	@echo "Formatting with Ruff..."
	ruff format .

lint:
	@echo "Linting with Ruff..."
	ruff check .

run: train

train:
	@echo "Training agent for $(EPISODES) episodes..."
	$(PYTHON) main.py train --episodes $(EPISODES)

visualize:
	@echo "Visualizing results..."
	$(PYTHON) main.py visualize

play:
	@echo "Launching interactive gameplay..."
	$(PYTHON) main.py play

docker-build:
	@echo "Building Docker image $(IMAGE)..."
	docker build -t $(IMAGE) .

docker-run: docker-build
	@echo "Running Docker image $(IMAGE)..."
	docker run --rm -it -v "$(PWD)":/app $(IMAGE) train --episodes $(EPISODES)

compose-up:
	docker compose up --build

compose-down:
	docker compose down

clean:
	@echo "Removing temporary files..."
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true
