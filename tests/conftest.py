"""Shared pytest configuration and fixtures for the PandaProbe Harness suite.

The database code under test lives in the ``migrations`` uv project
(``settings.py`` and ``models.py``). Tests run from the repository root via that
project's environment, so this conftest puts ``migrations/`` on ``sys.path`` to
make ``import settings`` / ``import models`` resolve cleanly.

Integration fixtures assume the throwaway test database from
``docker-compose.test.yml`` is reachable (see ``make test-integration``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"

# Make the migrations project importable (settings, models, env).
if str(MIGRATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(MIGRATIONS_DIR))


@pytest.fixture(scope="session")
def database_url() -> str:
    """Resolve the synchronous DB URL exactly as Alembic would at runtime."""
    from settings import get_sync_database_url

    return get_sync_database_url()


@pytest.fixture(scope="session")
def alembic_config():
    """An Alembic ``Config`` pinned to the migrations project (absolute paths)."""
    from alembic.config import Config

    cfg = Config(str(MIGRATIONS_DIR / "alembic.ini"))
    # Resolve script location absolutely so migrations are found regardless of
    # the current working directory.
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


@pytest.fixture()
def migrated_engine(alembic_config, database_url):
    """Upgrade the test DB to head, yield an Engine, then downgrade to base.

    Each test that requests this fixture runs against a freshly-migrated schema
    and leaves the database empty afterwards.
    """
    import sqlalchemy as sa
    from alembic import command

    command.upgrade(alembic_config, "head")
    engine = sa.create_engine(database_url, future=True)
    try:
        yield engine
    finally:
        engine.dispose()
        command.downgrade(alembic_config, "base")
