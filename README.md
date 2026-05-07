# Project Grader

Personal-use AI trading-card grader and authenticator. Multi-shot capture, per-criterion grading models with calibrated uncertainty, and a 7-detector counterfeit ensemble. Web app today; mobile is on the "eventually" list.

> **Scope note:** this project started life as a commercial-SaaS scaffold and was re-scoped (2026-05) to grade the user's own collection. Stripe / marketplace / kiosks / formal SLOs are dropped. Work is weighted by *"does this make the grader more accurate / reliable / usable for one person grading their own cards?"* Living scope + roadmap lives in [PROJECT_STATE.md](PROJECT_STATE.md) and [TODO.md](TODO.md).

## What's working

End-to-end grading runs today on a fresh dev box (with caveats below):

- ✅ 8-shot capture wizard (front, back, 4 corners, flash, tilt) with per-shot quality gating, camera + file-input fallback
- ✅ Detection + dewarp (YOLOv11-seg + Canny fallback) → canonical 750×1050
- ✅ Identification (DinoV2 + pgvector NN) — *but only if a reference catalog is present*
- ✅ Centering grade (geometric, ~99% by construction)
- ✅ Edges grade v1 (MAD anomaly)
- ✅ All 7 counterfeit detectors wired: rosette FFT, holographic-parallax, k-NN reference, typography (RapidOCR), substrate (paired-flash b\*), embedding-anomaly, CIELAB color
- ✅ Public cert page (`/cert/[id]`) — TAG-inspired surface: heatmap, DINGS-style defect list, population stat, Card Vision opacity slider, OG image, QR share, print view
- ✅ Production hardening: Sentry (opt-in), slowapi rate limits, 25 MiB upload cap, orphan-task reconciler, broker-failure 503

What's stubbed: the corners, surface, identification, and detection ML heads are skeletons — they live behind heuristics until labeled training data lands. See [PROJECT_STATE.md § State snapshot](PROJECT_STATE.md#state-snapshot) for the full matrix.

## Repo layout

```
apps/
  api/      FastAPI + SQLAlchemy + Celery, Postgres + pgvector
  web/      Next.js 14 (app router), Clerk auth (with dev fallback), Vitest + RTL
ml/         Python package; pipelines, training trainers, evaluation, scripts
infra/      docker-compose dev stack
docs/       architecture, ADRs, roadmap
```

`apps/api` depends on `ml/` via uv editable install.

## Prerequisites

- Node 20+ (npm is fine — pnpm-lock at the root, but `apps/web` has its own `package-lock.json` so npm works)
- Python 3.12+
- `uv` (https://docs.astral.sh/uv/) — invoke as `python -m uv` if not on PATH
- Docker
- AWS CLI (only for prod / Terraform work; not needed locally — MinIO substitutes for S3)

## Quick start (local dev)

The previous one-liner version was incomplete — these are the steps that actually work on a fresh box.

### 1. Bring up infra

```bash
docker compose -f infra/docker/docker-compose.dev.yml up -d
# Postgres+pgvector :5432, Redis :6379, MinIO :9000 (console :9001, login grader/gradergrader)
```

### 2. Create env files

`apps/api/.env`:

```ini
ENV=dev
LOG_LEVEL=INFO
DATABASE_URL=postgresql+asyncpg://grader:grader@localhost:5432/grader
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
S3_BUCKET=project-grader-dev
S3_REGION=us-east-1
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=grader
S3_SECRET_ACCESS_KEY=gradergrader
S3_FORCE_PATH_STYLE=true
CORS_ORIGINS=["http://localhost:3000"]
```

`apps/web/.env.local`:

```ini
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DEV_CLERK_ID=dev-user
```

### 3. Bootstrap the schema

> ⚠️ `alembic upgrade head` currently fails on a fresh DB with `type "game" already exists` (enum double-creation in the initial migration; tests bypass alembic via `Base.metadata.create_all` so this has been latent). Until the migration is fixed, bootstrap directly via SQLAlchemy models:

```bash
python -m uv run --project apps/api --directory apps/api python -c "
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from grader.db.models import Base

async def main():
    engine = create_async_engine('postgresql+asyncpg://grader:grader@localhost:5432/grader')
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"'))
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
asyncio.run(main())
"
```

### 4. Create the MinIO bucket

```bash
python -m uv run --project apps/api --directory apps/api python -c "
import boto3
from botocore.client import Config
s3 = boto3.client('s3', endpoint_url='http://localhost:9000',
                  aws_access_key_id='grader', aws_secret_access_key='gradergrader',
                  region_name='us-east-1',
                  config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
try: s3.create_bucket(Bucket='project-grader-dev')
except s3.exceptions.BucketAlreadyOwnedByYou: pass
print('buckets:', [b['Name'] for b in s3.list_buckets()['Buckets']])
"
```

### 5. Start the API, the worker, and the web

Three terminals (or background-job each):

```bash
# API on :8000
python -m uv run --project apps/api --directory apps/api uvicorn grader.main:app --host 127.0.0.1 --port 8000

# Celery worker (use --pool=solo on Windows; default prefork hangs there)
python -m uv run --project apps/api --directory apps/api celery -A grader.workers.celery_app worker --loglevel=info --pool=solo

# Web on :3000
cd apps/web && npm install --legacy-peer-deps && npm run dev
```

### 6. Open the wizard

Go directly to **http://localhost:3000/grade** — the home page (`/`) is a marketing landing with no nav links, so typing the URL is currently the only entry point. Walk through the 8 shots; submit; you'll be redirected to `/cert/<id>` when the pipeline finishes.

## Known dev-setup gotchas

- **Alembic migration is broken on fresh DBs** (see § 3 above). Tests bypass it via `Base.metadata.create_all`, so it's been undetected. Real fix: `create_type=False` on the enum constructors in `apps/api/alembic/versions/20260428_0001_initial_schema.py`. Tracked in [TODO.md](TODO.md).
- **Clerk dev fallback** in `apps/web/middleware.ts` and `apps/web/app/layout.tsx` was previously broken under `@clerk/nextjs` v5+ — comments claimed "no-op when key missing" but Clerk threw / showed a keyless overlay. Both files now genuinely short-circuit when `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is unset, preserving the documented fallback. Production behavior unchanged.
- **Identification returns `None` without a reference catalog.** The dev box needs `~/manufacturer_refs/reference_embeddings.npz` (produced by the daily flywheel on the Mac). Without it, the cert renders an amber "preliminary" banner with no card name; centering / edges / detectors that don't depend on a catalog still run.
- **Windows + Celery**: use `--pool=solo`. The default `prefork` pool doesn't fork on Windows.

## Running tests

```bash
# ml/ tests (Windows venv)
cd ml && .venv/Scripts/python.exe -m pytest -q

# apps/api unit tests, no Postgres needed
python -m uv run --project apps/api pytest apps/api/tests -q -m "not requires_postgres"

# apps/api full (Postgres on localhost:5432)
python -m uv run --project apps/api pytest apps/api/tests -q

# web Vitest
cd apps/web && npm test
```

## More

- [PROJECT_STATE.md](PROJECT_STATE.md) — single-file orientation snapshot for picking up a fresh session
- [TODO.md](TODO.md) — active work list
- [docs/roadmap.md](docs/roadmap.md) — full operational + shipped log
- [docs/adr/](docs/adr/) — architectural decisions
