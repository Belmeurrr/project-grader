"""Parity tests between SQLAlchemy models and the hand-written initial migration.

Hand-written migrations drift silently from models. These tests parse the
migration file via AST, extract every `op.create_table` call and its
columns, and assert that the result matches `Base.metadata`.

When you add a column to a model, the parity test fails until you either
update the migration or add a new revision."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from grader.db.models import Base

MIGRATION_PATH = (
    Path(__file__).parent.parent
    / "alembic"
    / "versions"
    / "20260428_0001_initial_schema.py"
)


def _parse_migration_tables() -> dict[str, set[str]]:
    """Walk the migration AST and return {table_name: {column_names}}."""
    tree = ast.parse(MIGRATION_PATH.read_text())
    tables: dict[str, set[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "op"
            and func.attr == "create_table"
        ):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        table_name = first.value
        column_names: set[str] = set()
        for arg in node.args[1:]:
            if not isinstance(arg, ast.Call):
                continue
            arg_func = arg.func
            is_column = (
                (isinstance(arg_func, ast.Attribute) and arg_func.attr == "Column")
                or (isinstance(arg_func, ast.Name) and arg_func.id == "Column")
            )
            if not is_column or not arg.args:
                continue
            col_arg = arg.args[0]
            if isinstance(col_arg, ast.Constant) and isinstance(col_arg.value, str):
                column_names.add(col_arg.value)
        tables[table_name] = column_names

    return tables


def test_every_model_table_is_in_migration() -> None:
    migration_tables = _parse_migration_tables()
    model_tables = {t.name for t in Base.metadata.tables.values()}
    missing = model_tables - migration_tables.keys()
    assert not missing, f"tables in models but missing from migration: {missing}"


def test_no_extra_tables_in_migration() -> None:
    migration_tables = _parse_migration_tables()
    model_tables = {t.name for t in Base.metadata.tables.values()}
    extra = migration_tables.keys() - model_tables
    assert not extra, f"tables in migration but missing from models: {extra}"


@pytest.mark.parametrize(
    "table_name",
    sorted({t.name for t in Base.metadata.tables.values()}),
)
def test_every_model_column_is_in_migration(table_name: str) -> None:
    migration_tables = _parse_migration_tables()
    assert table_name in migration_tables, (
        f"table {table_name} missing from migration"
    )
    model_columns = {c.name for c in Base.metadata.tables[table_name].columns}
    migration_columns = migration_tables[table_name]

    missing = model_columns - migration_columns
    extra = migration_columns - model_columns
    assert not missing, (
        f"{table_name}: columns in model but missing from migration: {missing}"
    )
    assert not extra, (
        f"{table_name}: columns in migration but missing from model: {extra}"
    )


def test_migration_creates_pgvector_extension() -> None:
    src = MIGRATION_PATH.read_text()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in src


def test_migration_creates_hnsw_index_on_embedding() -> None:
    src = MIGRATION_PATH.read_text()
    assert "USING hnsw" in src
    assert "vector_cosine_ops" in src


def test_migration_has_working_downgrade() -> None:
    """Cheap structural check: downgrade must drop every table it creates."""
    tree = ast.parse(MIGRATION_PATH.read_text())
    upgrades_created: set[str] = set()
    downgrades_dropped: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in ("upgrade", "downgrade"):
            continue
        target = upgrades_created if node.name == "upgrade" else downgrades_dropped
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            f = inner.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
                continue
            if f.value.id != "op":
                continue
            if f.attr in ("create_table", "drop_table"):
                if inner.args and isinstance(inner.args[0], ast.Constant):
                    target.add(inner.args[0].value)

    missing = upgrades_created - downgrades_dropped
    assert not missing, f"tables created but never dropped on downgrade: {missing}"
