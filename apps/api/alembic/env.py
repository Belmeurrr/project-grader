"""Alembic environment.

Loads the SQLAlchemy metadata from grader.db.models and the database URL from
grader.settings (Pydantic). We run migrations synchronously even though the
app uses asyncpg — Alembic's online runner doesn't need async, and we want a
plain psycopg URL here so we don't pull asyncpg into migration tooling.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from grader.db.models import Base
from grader.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _sync_database_url() -> str:
    """Return a sync (psycopg) variant of the configured DATABASE_URL.

    Settings default to asyncpg; rewrite to psycopg for Alembic to avoid
    pulling asyncpg into migration runs and to keep DDL behavior predictable.
    """
    url = get_settings().database_url
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


config.set_main_option("sqlalchemy.url", _sync_database_url())

target_metadata = Base.metadata


def render_item(type_: str, obj: object, autogen_context) -> str | bool:
    """Custom renderer so pgvector's Vector(...) column type emits correctly."""
    from pgvector.sqlalchemy import Vector

    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({obj.dim})"
    return False


def include_object(object_, name, type_, reflected, compare_to) -> bool:
    """Skip the alembic_version table itself in autogenerate."""
    if type_ == "table" and name == "alembic_version":
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=render_item,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
