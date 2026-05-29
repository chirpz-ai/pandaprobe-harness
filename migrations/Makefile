.DEFAULT_GOAL := help

.PHONY: migration migrate rollback check help

# -- Database -----------------------------------------------------------------

migration:  ## Auto-generate a new Alembic migration.  Usage: make migration msg="add memory_nodes index"
	@test -n "$(msg)" || { echo "Usage: make migration msg=\"short description\""; exit 1; }
	POSTGRES_HOST=localhost uv run alembic revision --autogenerate -m "$(msg)"

migrate:  ## Run Alembic migrations (head) against local Postgres
	POSTGRES_HOST=localhost uv run alembic upgrade head

rollback:  ## Roll back the most recently applied migration
	POSTGRES_HOST=localhost uv run alembic downgrade -1

check:  ## Fail if the SQLAlchemy models have drifted from the migrations
	POSTGRES_HOST=localhost uv run alembic check

# -- Help ---------------------------------------------------------------------

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
