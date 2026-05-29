"""Connection settings for the PandaProbe Harness migration runtime.

Configuration is separated from the Alembic entrypoint, mirroring the
``env.py`` + ``settings.py`` split used by the core ``pandaprobe`` backend.
Alembic requires its entrypoint file to be named ``env.py``, so ``env.py``
stays thin and imports the resolved URL from here.

The synchronous (psycopg2) SQLAlchemy URL is resolved with this precedence:
  1. ``DATABASE_URL`` env var, verbatim, if set.
  2. Otherwise composed from the individual ``POSTGRES_*`` parts / defaults.

Because migrations execute from the host (see the root ``Makefile``),
``POSTGRES_HOST`` defaults to ``localhost``.
"""

from __future__ import annotations

import os

# Defaults match docker-compose.yml. POSTGRES_HOST defaults to localhost so the
# host-run migration tooling reaches the published container port.
DEFAULTS: dict[str, str] = {
    "POSTGRES_USER": "panda_admin",
    "POSTGRES_PASSWORD": "panda_secret",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "panda_harness",
}


def get_sync_database_url() -> str:
    """Return the synchronous SQLAlchemy URL for Alembic to connect with."""
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit

    user = os.environ.get("POSTGRES_USER", DEFAULTS["POSTGRES_USER"])
    password = os.environ.get("POSTGRES_PASSWORD", DEFAULTS["POSTGRES_PASSWORD"])
    host = os.environ.get("POSTGRES_HOST", DEFAULTS["POSTGRES_HOST"])
    port = os.environ.get("POSTGRES_PORT", DEFAULTS["POSTGRES_PORT"])
    db = os.environ.get("POSTGRES_DB", DEFAULTS["POSTGRES_DB"])
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
