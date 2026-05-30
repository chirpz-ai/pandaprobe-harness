"""Unit tests for the SQLAlchemy schema in migrations.models (no database)."""

from __future__ import annotations

from models import Base

EXPECTED_COLUMNS = {
    "node_id",
    "parent_id",
    "session_id",
    "is_head",
    "framework_type",
    "payload",
    "created_at",
}


def _table():
    return Base.metadata.tables["memory_nodes"]


def test_table_registered():
    assert "memory_nodes" in Base.metadata.tables


def test_columns_and_nullability():
    table = _table()
    assert {c.name for c in table.columns} == EXPECTED_COLUMNS
    assert table.c.node_id.primary_key is True
    assert table.c.session_id.nullable is False
    assert table.c.framework_type.nullable is False
    assert table.c.payload.nullable is False
    assert table.c.is_head.nullable is False
    # Root nodes hold NULL parent_id.
    assert table.c.parent_id.nullable is True


def test_self_referencing_foreign_key_cascades():
    table = _table()
    fks = list(table.foreign_keys)
    assert len(fks) == 1
    fk = fks[0]
    assert fk.column.table.name == "memory_nodes"
    assert fk.column.name == "node_id"
    assert fk.ondelete == "CASCADE"


def test_indexes_defined():
    table = _table()
    by_name = {ix.name: ix for ix in table.indexes}
    assert "idx_session_head" in by_name
    assert "idx_parent_id" in by_name


def test_session_head_index_is_partial():
    table = _table()
    index = next(ix for ix in table.indexes if ix.name == "idx_session_head")
    where = index.dialect_options["postgresql"].get("where")
    assert where is not None
    assert "is_head" in str(where)
