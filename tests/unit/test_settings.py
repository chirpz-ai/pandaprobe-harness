"""Unit tests for migrations.settings.get_sync_database_url (no database)."""

from __future__ import annotations

import settings

_POSTGRES_ENV = (
    "DATABASE_URL",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
)


def _clear_env(monkeypatch):
    for key in _POSTGRES_ENV:
        monkeypatch.delenv(key, raising=False)


def test_explicit_database_url_takes_precedence(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@example:9999/custom")
    # Even with conflicting parts present, DATABASE_URL must win verbatim.
    monkeypatch.setenv("POSTGRES_USER", "ignored")
    assert settings.get_sync_database_url() == "postgresql+psycopg2://u:p@example:9999/custom"


def test_defaults_match_docker_compose(monkeypatch):
    _clear_env(monkeypatch)
    assert (
        settings.get_sync_database_url()
        == "postgresql+psycopg2://panda_admin:panda_secret@localhost:5432/panda_harness"
    )


def test_url_composed_from_parts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("POSTGRES_USER", "alice")
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "harness_x")
    assert (
        settings.get_sync_database_url()
        == "postgresql+psycopg2://alice:s3cret@db.internal:6543/harness_x"
    )


def test_host_override_only(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("POSTGRES_HOST", "db")
    url = settings.get_sync_database_url()
    assert "@db:5432/" in url
    assert url.startswith("postgresql+psycopg2://panda_admin:panda_secret@")
