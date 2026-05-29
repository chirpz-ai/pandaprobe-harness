"""SQLAlchemy schema definitions for the PandaProbe Harness storage layer.

This module is the single source of truth for the database schema. Alembic's
``--autogenerate`` (see ``make db-migration``) diffs ``Base.metadata`` here
against the live database to produce new migration scripts, mirroring the
model-driven workflow used by the core ``pandaprobe`` backend.

The definitions below intentionally match migration ``0001`` exactly so that
autogenerate reports no drift against an already-migrated database.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base carrying the project's table metadata."""


class MemoryNode(Base):
    """A single node in the version-controlled memory DAG (Envelope Pattern).

    Relational columns hold the structural metadata used for fast timeline
    traversal; the framework-specific state is offloaded into the JSONB
    ``payload`` column.
    """

    __tablename__ = "memory_nodes"

    node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "memory_nodes.node_id",
            name="fk_memory_nodes_parent_id",
            ondelete="CASCADE",
        ),
        nullable=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    is_head: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    framework_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    __table_args__ = (
        # Partial index -> O(1) active-HEAD lookup per session.
        Index(
            "idx_session_head",
            "session_id",
            postgresql_where=text("is_head = true"),
        ),
        # B-Tree index for hierarchical parent traversal.
        Index("idx_parent_id", "parent_id"),
    )
