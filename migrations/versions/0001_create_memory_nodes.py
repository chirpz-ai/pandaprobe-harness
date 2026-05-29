"""create memory_nodes table (Envelope Pattern)

Revision ID: 0001
Revises:
Create Date: 2026-05-29

Establishes the version-controlled DAG backing the agent's short-term memory.
Each row is a node in the conversation timeline; relational columns carry the
structural metadata required for fast traversal while the framework-specific
state is offloaded into a single JSONB ``payload`` column (Envelope Pattern).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "memory_nodes"
SESSION_HEAD_INDEX = "idx_session_head"
PARENT_ID_INDEX = "idx_parent_id"


def upgrade() -> None:
    op.create_table(
        TABLE_NAME,
        sa.Column(
            "node_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "is_head",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "framework_type",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        # Self-referencing edge: root nodes hold NULL parent_id.
        sa.ForeignKeyConstraint(
            ["parent_id"],
            [f"{TABLE_NAME}.node_id"],
            name="fk_memory_nodes_parent_id",
            ondelete="CASCADE",
        ),
    )

    # Partial index: at most one HEAD per session, so this yields an O(1)
    # lookup of the active pointer for any conversational session.
    op.create_index(
        SESSION_HEAD_INDEX,
        TABLE_NAME,
        ["session_id"],
        unique=False,
        postgresql_where=sa.text("is_head = true"),
    )

    # B-Tree index to optimize backward traversal of the execution tree.
    op.create_index(
        PARENT_ID_INDEX,
        TABLE_NAME,
        ["parent_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(PARENT_ID_INDEX, table_name=TABLE_NAME)
    op.drop_index(SESSION_HEAD_INDEX, table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)
