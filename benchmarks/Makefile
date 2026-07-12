.DEFAULT_GOAL := help

# Every target is sugar over a documented `uv run pandabench-*` command; the
# Makefile carries no logic of its own.

# Per-run knobs (overridable): make appworld ARM=harness MODEL=claude-sonnet-5 SEED=1 K=4 LIMIT=2
ARM     ?= baseline
MODEL   ?= gemini-3.1-flash-lite
SEED    ?= 1
BACKEND ?=
K       ?= 4
LIMIT   ?=
BENCH   ?= appworld
BENCHMARK ?= $(BENCH)
LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)
BACKEND_FLAG := $(if $(BACKEND),--backend $(BACKEND),)

.PHONY: help setup sync smoke run appworld terminal tau2 matrix report calibrate check lint typecheck test preflight

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## uv sync with all extras + dev group
	uv sync --all-extras --group dev

setup: sync ## Full setup: deps, harbor tool, isolated AppWorld env, preflight
	uv tool install harbor || echo "harbor install skipped (already installed or unavailable)"
	bash scripts/setup_appworld.sh || echo "appworld setup skipped (see scripts/setup_appworld.sh)"
	uv run pandabench-run --preflight || true

preflight: ## Validate Docker, CLIs, credentials, and a 1-token ping per provider
	uv run pandabench-run --preflight

smoke: ## 2 tasks x 1 trial x both arms x cheap model, all benchmarks (< ~10 min)
	uv run pandabench-run --smoke

run: ## Run one arm of one benchmark (BENCHMARK= ARM= MODEL= SEED= [BACKEND=] [K=] [LIMIT=])
	uv run pandabench-run --benchmark $(BENCHMARK) --arm $(ARM) --model $(MODEL) \
		--seed $(SEED) $(BACKEND_FLAG) -k $(K) $(LIMIT_FLAG)

appworld terminal tau2: ## Run one benchmark (target name = benchmark); honors ARM/MODEL/SEED/...
	uv run pandabench-run --benchmark $@ --arm $(ARM) --model $(MODEL) \
		--seed $(SEED) $(BACKEND_FLAG) -k $(K) $(LIMIT_FLAG)

matrix: ## Run the full study matrix from configs/study.yaml (prints cost estimate, confirms)
	uv run pandabench-run --matrix configs/study.yaml

report: ## Regenerate summary/ (CSVs + report.md + plots) from runs/
	uv run pandabench-report

calibrate: ## Checkpoint 1: metric<->failure calibration for one benchmark (BENCH=appworld)
	uv run pandabench-calibrate --benchmark $(BENCH)

check: lint typecheck test ## Lint + typecheck + offline unit tests

lint: ## Lint with ruff
	uv run ruff check .

typecheck: ## Type-check with mypy (strict)
	uv run mypy

test: ## Run offline unit tests (client + benchmark envs mocked; no network)
	uv run pytest
