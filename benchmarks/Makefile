.DEFAULT_GOAL := help

# Targets invoke the documented benchmark CLIs. Post-processing uses the
# installed .venv directly so reporting does not touch the uv package cache.

# Per-run knobs (overridable): make appworld ARM=harness MODEL=claude-sonnet-5 SEED=1 K=4 LIMIT=2
#   ARM      baseline | harness
#   MODEL    a models.yaml key (see `configs/models.yaml`)
#   SEED     shuffles task order, identically in both arms; use several (1 2 3) for replicates
#   K        trials per task (pass@1 = first trial; pass^k = all K)
#   LIMIT    run only the first N tasks (unset = the whole dataset)
#   DATASET  dataset override (unset = the benchmark's configured dataset)
#   MAXTURNS per-task agent-turn cap (unset = study.yaml max_turns, 100 for all benchmarks)
#   BACKEND  Claude only: bedrock | anthropic (default: bedrock)
#   RUN_ID   reuse an existing run id to RESUME it (skips completed task-trials)
ARM      ?= baseline
MODEL    ?= gemini-3.1-flash-lite
SEED     ?= 1
BACKEND  ?=
K        ?= 4
LIMIT    ?=
DATASET  ?=
MAXTURNS ?=
BENCH    ?= appworld
BENCHMARK ?= $(BENCH)
RUN_ID   ?=
REPORT_OUT ?= results/summary/all_runs
LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)
DATASET_FLAG := $(if $(DATASET),--dataset $(DATASET),)
MAXTURNS_FLAG := $(if $(MAXTURNS),--max-turns $(MAXTURNS),)
BACKEND_FLAG := $(if $(BACKEND),--backend $(BACKEND),)
RUN_ID_FLAG := $(if $(RUN_ID),--run-id $(RUN_ID),)

.PHONY: help setup sync smoke run appworld terminal terminal_bench tau2 matrix report report-openweight calibrate check lint typecheck test preflight

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## uv sync with all extras + dev group
	uv sync --all-extras --group dev

setup: sync ## Full setup: deps (including harbor), isolated AppWorld env, preflight
	bash scripts/setup_appworld.sh || echo "appworld setup skipped (see scripts/setup_appworld.sh)"
	uv run pandabench-run --preflight || true

preflight: ## Validate Docker, CLIs, credentials, and a 1-token ping per provider
	uv run pandabench-run --preflight

smoke: ## 2 tasks x 1 trial x both arms x cheap model, all benchmarks (< ~10 min)
	uv run pandabench-run --smoke

run: ## Run one arm of one benchmark (BENCHMARK= ARM= MODEL= SEED= [BACKEND=] [K=] [LIMIT=] [DATASET=] [MAXTURNS=] [RUN_ID=])
	uv run pandabench-run --benchmark $(BENCHMARK) --arm $(ARM) --model $(MODEL) \
		--seed $(SEED) $(BACKEND_FLAG) -k $(K) $(LIMIT_FLAG) $(DATASET_FLAG) $(MAXTURNS_FLAG) $(RUN_ID_FLAG)

# The `terminal` target keeps its short name but must pass the CLI's actual
# choice, `terminal_bench` — `--benchmark terminal` is rejected.
appworld terminal_bench tau2: ## Run one benchmark (target name = benchmark); honors ARM/MODEL/SEED/K/LIMIT/DATASET/MAXTURNS/RUN_ID
	uv run pandabench-run --benchmark $@ --arm $(ARM) --model $(MODEL) \
		--seed $(SEED) $(BACKEND_FLAG) -k $(K) $(LIMIT_FLAG) $(DATASET_FLAG) $(MAXTURNS_FLAG) $(RUN_ID_FLAG)

terminal: terminal_bench ## Alias for terminal_bench

matrix: ## Full study matrix — NOT yet wired; prints how to run per-arm loops (see RUNNING.md §6)
	uv run pandabench-run --matrix configs/study.yaml

report: ## Regenerate summary/all_runs/ from every run (REPORT_OUT= to override)
	MPLCONFIGDIR=results/summary/.matplotlib XDG_CACHE_HOME=results/summary/.cache \
		.venv/bin/pandabench-report --out $(REPORT_OUT) $(if $(RELAX),--relax $(RELAX),)

report-openweight: ## Regenerate summary/openweight/ from the audited openweight runs
	MPLCONFIGDIR=results/summary/.matplotlib XDG_CACHE_HOME=results/summary/.cache \
		.venv/bin/python scripts/report_openweight.py

calibrate: ## Checkpoint 1: metric<->failure calibration for one benchmark (BENCH=appworld)
	uv run pandabench-calibrate --benchmark $(BENCH)

check: lint typecheck test ## Lint + typecheck + offline unit tests

lint: ## Lint with ruff
	uv run ruff check .

typecheck: ## Type-check with mypy (strict)
	uv run mypy

test: ## Run offline unit tests (client + benchmark envs mocked; no network)
	uv run pytest
