SHELL := /bin/sh

.DEFAULT_GOAL := help

COMPOSE_FILE ?= compose.sm61-v031.yaml
VLLM_ENV_FILE ?= .env.v031
COMPOSE := VLLM_ENV_FILE=$(VLLM_ENV_FILE) docker compose -f $(COMPOSE_FILE)
BUILD_LOG := logs/build/latest.log
RUNTIME_LOG := logs/runtime/latest.log
VLLM_URL ?= http://vllm.test
API_KEY ?= EMPTY
MODEL_NAME ?= Qwen3.5-4B-GGUF

.PHONY: help refresh-source-cache source-cache extract-torch build up down build-log runtime-log logs status smoke \
	sm61-source-cache sm61-extract-torch sm61-build sm61-up sm61-down sm61-build-log fetch-torch \
	prepare-host prepare-model \
	dflash-bench dflash-smoke

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9][a-zA-Z0-9_.-]*:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

refresh-source-cache: ## Warm the git source cache used by the sm61 build
	sh ./scripts/refresh-sm61-git-cache.sh

source-cache: ## Build the cached source image
	sh ./scripts/build-sm61-source-cache.sh

sm61-source-cache: source-cache

extract-torch: ## Extract custom torch from a post-torch image layer
	sh ./scripts/extract-sm61-torch-artifact.sh

sm61-extract-torch: extract-torch

fetch-torch: ## Download the prebuilt sm_61 torch wheel (skips the hours-long build)
	sh ./scripts/fetch-torch-artifact.sh

prepare-host: ## Check the host (GPU, docker, disk) and create dirs, network, .env.v031
	sh ./scripts/prepare-host.sh

prepare-model: ## Download the reference GGUF model, tokenizer and config.json
	sh ./scripts/prepare-model.sh

build: ## Build local/vllm-openai:sm61
	sh ./scripts/build-sm61-image.sh

sm61-build: build

up: ## Start the sm61 compose stack
	$(COMPOSE) up -d

sm61-up: up

down: ## Stop the sm61 compose stack
	$(COMPOSE) down

sm61-down: down

build-log: ## Tail the latest build log
	tail -f $(BUILD_LOG)

sm61-build-log: build-log

runtime-log: ## Tail the latest runtime log
	tail -f $(RUNTIME_LOG)

logs: runtime-log

status: ## Query the idle-unload proxy status without waking the model
	$(COMPOSE) exec -T vllm python3 /workspace/scripts/idle_status.py http://127.0.0.1:8000

smoke: ## Run the API smoke test against the configured endpoint
	API_KEY="$(API_KEY)" MODEL_NAME="$(MODEL_NAME)" ./scripts/smoke-test.sh "$(VLLM_URL)"

dflash-smoke: ## Start the stack with a temporary DFlash-only env and run the smoke test
	API_KEY="$(API_KEY)" MODEL_NAME="$(MODEL_NAME)" ./scripts/dflash-bench.sh --modes dflash-only --skip-quality-bench "$(VLLM_URL)"

dflash-bench: ## Run the baseline/DFlash/TurboQuant benchmark matrix with temporary env variants
	API_KEY="$(API_KEY)" MODEL_NAME="$(MODEL_NAME)" ./scripts/dflash-bench.sh "$(VLLM_URL)"
