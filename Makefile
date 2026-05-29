.DEFAULT_GOAL := help

COMPOSE := docker-compose
COMPOSE_TEST := $(COMPOSE) -f docker-compose.test.yml
MIGRATIONS_DIR := migrations

.PHONY: help up down clean logs logs-db db-migration db-migrate db-rollback db-check test test-unit test-integration

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Spin up the database and services in detached mode
	$(COMPOSE) up -d

down: ## Stop and remove containers and networks (data volume is preserved)
	$(COMPOSE) down

clean: ## Stop everything AND delete the database volume (destroys all data)
	$(COMPOSE) down -v

logs: ## Tail logs from all running services (follow)
	$(COMPOSE) logs -f

logs-db: ## Tail database logs only (follow)
	$(COMPOSE) logs -f db

db-migration: ## Auto-generate a migration from model changes. Usage: make db-migration msg="add foo"
	$(MAKE) -C $(MIGRATIONS_DIR) migration msg="$(msg)"

db-migrate: ## Apply all pending database migrations
	$(MAKE) -C $(MIGRATIONS_DIR) migrate

db-rollback: ## Roll back the most recently applied migration
	$(MAKE) -C $(MIGRATIONS_DIR) rollback

db-check: ## Fail if the SQLAlchemy models have drifted from the migrations
	$(MAKE) -C $(MIGRATIONS_DIR) check

test: test-unit test-integration ## Run the full test suite (unit + integration)

test-unit: ## Run unit tests (no database required)
	uv run --project $(MIGRATIONS_DIR) --group test pytest tests/unit -v

test-integration: ## Run integration tests against a throwaway test database (port 5433)
	$(COMPOSE_TEST) up -d --wait
	POSTGRES_PORT=5433 POSTGRES_DB=panda_harness_test \
		uv run --project $(MIGRATIONS_DIR) --group test pytest tests/integration -v; \
	status=$$?; \
	$(COMPOSE_TEST) down -v; \
	exit $$status
