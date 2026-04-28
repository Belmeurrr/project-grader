# Project Grader

AI trading card grader and authenticator. Multi-platform (web + mobile), software-only at v1, designed to grow into a real grading service.

## What this is

A production-grade card grading and authenticity system covering Sports, Pokemon, Magic: The Gathering, Yu-Gi-Oh, One Piece, Lorcana, and other major TCGs. The two differentiators vs existing AI graders (CardGrade.io, AGS, TCG AI Pro, TAG):

1. **Higher accuracy** via multi-shot capture, strict capture-quality gating, and per-criterion grading models with calibrated uncertainty (target ≥ 96% top-1 PSA accuracy on held-out PSA-confirmed cards).
2. **Counterfeit detection as a first-class feature** — a 7-detector ensemble including print-rosette FFT analysis that works on cards with no per-card reference.

See `docs/adr/0001-tech-stack.md` for architectural decisions and `docs/architecture/` for system design.

## Repo layout

```
apps/        # deployable units (web, mobile, api)
packages/    # shared TS libs (design tokens, api client, types, ui)
ml/          # ML pipelines, training, serving, data
infra/       # terraform, k8s, docker
docs/        # architecture, ADRs, runbooks
```

## Prerequisites

- Node 20.11+
- pnpm 9+
- Python 3.12+
- `uv` (https://docs.astral.sh/uv/)
- Docker (for local Postgres/Redis)
- AWS CLI (for infra work)

## Quick start

```bash
# install JS deps
pnpm install

# install Python deps for the API
cd apps/api && uv sync && cd ../..

# run local Postgres + Redis
docker compose -f infra/docker/docker-compose.dev.yml up -d

# run API in dev
cd apps/api && uv run uvicorn grader.main:app --reload --port 8000

# run web in dev
pnpm --filter web dev
```

## Phase 1 MVP scope

Web only. Pokemon + MTG only. 6-shot capture (front, back, 4 corner zooms). Centering + corners + edges grading. PSA-style final grade. Counterfeit detectors: rosette, color, typography, embedding-anomaly. TCGplayer comps. Clerk auth + Stripe.

## Status

Greenfield. Scaffolding in progress.
