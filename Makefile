.DEFAULT_GOAL := help

COMPOSE := docker-compose
MIGRATIONS_DIR := migrations

.PHONY: help up down clean db-migration db-migrate db-rollback db-check test

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Spin up the database and services in detached mode
	$(COMPOSE) up -d

down: ## Stop and remove containers and networks (data volume is preserved)
	$(COMPOSE) down

clean: ## Stop everything AND delete the database volume (destroys all data)
	$(COMPOSE) down -v

db-migration: ## Auto-generate a migration from model changes. Usage: make db-migration msg="add foo"
	$(MAKE) -C $(MIGRATIONS_DIR) migration msg="$(msg)"

db-migrate: ## Apply all pending database migrations
	$(MAKE) -C $(MIGRATIONS_DIR) migrate

db-rollback: ## Roll back the most recently applied migration
	$(MAKE) -C $(MIGRATIONS_DIR) rollback

db-check: ## Fail if the SQLAlchemy models have drifted from the migrations
	$(MAKE) -C $(MIGRATIONS_DIR) check

test: ## Run unit and integration checks across workspace components
	cd $(MIGRATIONS_DIR) && uv run python -c "import alembic, psycopg2; print('migrations toolchain OK')"
	@if [ -d tests ] && ls tests/*.py >/dev/null 2>&1; then \
		uv run --project $(MIGRATIONS_DIR) pytest tests; \
	else \
		echo "No Python integration tests found under tests/ yet; skipping."; \
	fi
	@for svc in services/memory-service services/harness-engine; do \
		if [ -f $$svc/go.mod ]; then \
			echo "Running Go tests in $$svc"; \
			(cd $$svc && go test ./...); \
		else \
			echo "No Go module in $$svc yet; skipping."; \
		fi; \
	done
