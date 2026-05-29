"""Alembic entrypoint for the PandaProbe Harness storage layer.

Kept intentionally thin (mirroring the core ``pandaprobe`` backend): connection
configuration lives in ``settings.py`` and the schema source of truth lives in
``models.py``. Alembic requires this file to be named ``env.py``.

``target_metadata`` is bound to the SQLAlchemy models so that
``alembic revision --autogenerate`` (``make db-migration``) can diff the models
against the live database.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from models import Base
from settings import get_sync_database_url

# Alembic Config object, providing access to values within alembic.ini.
config = context.config

# Inject the resolved connection string before the engine is constructed.
config.set_main_option("sqlalchemy.url", get_sync_database_url())

# Configure Python logging from the alembic.ini settings.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Schema source of truth for autogenerate.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode, emitting SQL to script output.

    No live DB connection is created; Alembic renders the migration as SQL
    using only the configured URL.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
