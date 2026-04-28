# Runbook — Database migrations

Project Grader uses Alembic against a Postgres database with the `pgvector`
extension. Migrations live in `apps/api/alembic/versions/`.

## Local dev

```bash
# 1. bring up Postgres (the pgvector image)
docker compose -f infra/docker/docker-compose.dev.yml up -d postgres

# 2. install API deps (one-time)
cd apps/api && uv sync

# 3. apply all migrations
uv run alembic upgrade head

# 4. verify the schema is in place
uv run python -c "from grader.db.models import Base; print(sorted(t.name for t in Base.metadata.tables.values()))"
```

The `DATABASE_URL` is read from `.env` (see `apps/api/.env.example`). Alembic
internally rewrites `postgresql+asyncpg://` to `postgresql+psycopg://` so DDL
runs through the sync driver — keep the env file using asyncpg for app code.

## Creating a new migration

```bash
cd apps/api
uv run alembic revision --autogenerate -m "short message"
```

Then review the generated file under `alembic/versions/`:

- The `pgvector` `Vector(...)` type is rendered via `render_item` in `env.py`;
  if you see a missing import, check `env.py:render_item`.
- For new HNSW or IVFFlat vector indexes, write the `CREATE INDEX … USING hnsw`
  by hand inside `op.execute(...)` — autogen does not yet emit pgvector index
  syntax.
- Always provide a working `downgrade()`. The CI gate is "any migration that
  upgrades cleanly must also downgrade cleanly" against an empty database.

## Production rollout

1. Migrations run as a one-off ECS task before the web tier is deployed.
2. The deploy pipeline blocks if the migration job fails.
3. Long-running data backfills go in a separate migration that uses
   `op.execute("SET statement_timeout = 0")` and operates in batches; never
   block deploys on backfills.
4. Roll forward, never roll back — write a forward fix instead. Keep
   `downgrade()` correct for development reset, not for production rollback.

## Resetting local dev

```bash
docker compose -f infra/docker/docker-compose.dev.yml down -v
docker compose -f infra/docker/docker-compose.dev.yml up -d postgres
cd apps/api && uv run alembic upgrade head
```
