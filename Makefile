.DEFAULT_GOAL := help

COMPOSE := docker compose
COMPOSE_TEST := $(COMPOSE) -f docker-compose.test.yml

.PHONY: help install up down clean logs build harness-shell lint typecheck test test-unit test-e2e test-contract example bench-setup bench-smoke bench-run bench-report bench-check

# Benchmark sub-project lives in ./benchmarks (its own uv project). These are
# thin delegators; all logic is in benchmarks/Makefile.
MAKE_BENCH := $(MAKE) --no-print-directory -C benchmarks

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the Python environment (uv)
	uv sync

build: ## Build the diagnostic sandbox image
	$(COMPOSE) build

up: ## Spin up the diagnostic sandbox in detached mode
	$(COMPOSE) up -d

down: ## Stop and remove the sandbox (harness volume is preserved)
	$(COMPOSE) down

clean: ## Stop everything AND delete the /harness volume (destroys learned rules)
	$(COMPOSE) down -v

logs: ## Tail sandbox logs (follow)
	$(COMPOSE) logs -f

harness-shell: ## Open the restricted bash shell inside the running sandbox
	$(COMPOSE) exec sandbox bash

lint: ## Lint with ruff
	uv run ruff check .

typecheck: ## Type-check with mypy (strict)
	uv run mypy src

test: ## Run the full offline test suite (unit + e2e)
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest tests/unit -v

test-e2e: ## Run the end-to-end pull-loop + concurrency scenarios
	uv run pytest tests/e2e_pull_loop_test.py tests/e2e_concurrency_test.py -v

test-contract: ## Run the live contract tests (needs pandaprobe CLI + creds)
	PANDAPROBE_LIVE=1 uv run pytest tests/contract -v

example: ## Run the offline self-heal example (fake CLI, no network)
	uv run python examples/offline_self_heal.py

bench-setup: ## Set up the benchmarks/ sub-project (uv sync, harbor, appworld data)
	$(MAKE_BENCH) setup

bench-smoke: ## Run the benchmark smoke test (cheap, both arms, all benchmarks)
	$(MAKE_BENCH) smoke

bench-run: ## Run one benchmark arm (pass BENCHMARK= ARM= MODEL= SEED= ...)
	$(MAKE_BENCH) run BENCHMARK=$(BENCHMARK) ARM=$(ARM) MODEL=$(MODEL) SEED=$(SEED) BACKEND=$(BACKEND) K=$(K) LIMIT=$(LIMIT)

bench-report: ## Regenerate the benchmark summary/ artifacts
	$(MAKE_BENCH) report

bench-check: ## Lint + typecheck + unit-test the benchmarks/ code
	$(MAKE_BENCH) check
