# ADR 0001 — Tech Stack

- **Status:** Accepted
- **Date:** 2026-04-28
- **Decision drivers:** software-only multi-platform v1; aspiring grading service; primary differentiators are higher accuracy and counterfeit detection.

## Context

Greenfield project. Team will be small (3 senior engineers + 1 ML + 1 designer at MVP). The system must combine a TS web/mobile front end with a Python ML stack, support GPU inference at p95 ≤ 20s, and produce evidence trails sufficient to defend grades in disputes once the grading-service phases launch.

## Decisions

### API and runtime

- **Python 3.12 + FastAPI + Pydantic v2** for the API service. Same language as the ML stack; eliminates the serialization tax and lets request handlers call inference code directly when needed.
- **Async I/O** via `asyncio` + `httpx`. `uvicorn` workers behind an ALB.
- **uv** for Python package management (faster + reproducible vs Poetry/pip-tools).

### Data stores

- **Aurora Postgres** as primary OLTP store. JSONB columns for unstructured per-card metadata.
- **`pgvector`** (HNSW index) for the card-identification embedding index. Switch to a dedicated vector DB only if we exceed ~50M vectors.
- **Redis (ElastiCache)** as cache + Celery broker. db 0 cache, db 1 broker.
- **S3** for raw images, crops, masks, and visualization artifacts. Lifecycle: hot 30 days → Intelligent-Tiering.

### Async / workers

- **Celery + Redis broker** for grading job orchestration. v1 fan-out (one job → 6–12 model calls) does not justify Temporal yet; revisit at Phase 3 when slabbing workflows become long-running and durable state matters.

### ML serving

- **NVIDIA Triton Inference Server** on `g5.xlarge` (A10G) workers. Multi-model batching across PyTorch / ONNX / TensorRT in one process — important because the pipeline has 6–10 models.
- **MLflow** (S3-backed) for model registry and experiment tracking.
- **DVC** for dataset versioning. Every MLflow model carries a `dataset_version` tag.

### Web front end

- **Next.js 15 (App Router) + TypeScript + Tailwind + shadcn/ui**. Chosen over Vite for SEO on card-detail / share pages, server actions for upload-URL signing, and ISR for the public cert-lookup pages.
- **TanStack Query + Zustand** for state (no Redux).
- **Server-Sent Events** for one-way submission progress (simpler than WebSockets).

### Mobile front end

- **React Native + Expo + `react-native-vision-camera`**. Native (Swift + Kotlin) doubles eng cost without enough payoff at v1; Flutter pulls us off the shared TS design system.
- **On-device ML for capture guidance only.** iOS via Core ML (`.mlmodel`), Android via TFLite (`.tflite`). Same source PyTorch model, two export targets. Heavy grading models stay server-side so we can iterate without app updates.

### Shared / monorepo

- **Turborepo + pnpm workspaces.**
- **orval-generated TS API client** from FastAPI's OpenAPI schema (single source of truth).
- **Shared design tokens only** between web and mobile. Don't try to share components.

### Infra and ops

- **AWS, Terraform-managed.** ECS Fargate for the stateless API; EKS only for GPU workers. Don't go full Kubernetes day one.
- **Cloudflare** in front for CDN/edge. Signed S3 upload URLs go client → S3 directly.
- **GitHub Actions** for CI/CD.

### Auth, payments, flags

- **Clerk** for auth (better React Native support than Auth0, cheaper at our scale).
- **Stripe** for payments.
- **Statsig** for feature flags. Server-side flags are critical for ML model canary rollouts (1% → 10% → 100%).

### Observability

- **Datadog APM** + **Sentry** errors + **Grafana** for ML metrics (drift, latency p95, confidence histograms).
- All logs to CloudWatch then shipped to Datadog.

## Consequences

- Two-language monorepo (Python + TS). Mitigated by clean app boundaries — Python lives in `apps/api/` and `ml/`, TS lives in `apps/web/`, `apps/mobile/`, `packages/`. Cross-language contract is the FastAPI OpenAPI schema.
- AWS lock-in for Aurora, ElastiCache, S3, Triton on EKS. Acceptable trade for SageMaker/Bedrock optionality later.
- Triton has a learning curve but pays off when running 6–10 models per request.
- `pgvector` will need monitoring as catalog grows. Plan to stand up a dedicated read-replica for vector queries before exceeding ~10M cards.
- Celery's failure model (visibility timeouts, broker partition drops) is the operational risk. Worth an early runbook in `docs/runbooks/`.

## Alternatives considered (briefly)

- **Node API**: rejected. ML glue would re-implement Python serialization across the wire repeatedly.
- **Vite for web**: rejected. SEO + ISR matter on cert and card pages.
- **Native iOS/Android**: rejected for v1 cost. Reconsider when on-device ML perf becomes the bottleneck.
- **Temporal over Celery**: rejected for v1 simplicity; appropriate for Phase 3+.
- **MongoDB**: rejected. Relational integrity (audit log → submissions → grades → users) is core to the business; JSONB on Postgres handles the unstructured cases.
- **Cloudflare R2 over S3**: rejected. Egress savings outweighed by AWS-native ML tooling integration.
