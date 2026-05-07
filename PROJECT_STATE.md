# Project state — handoff snapshot (2026-05-07)

Single-file orientation doc for picking up a fresh chat session. Pairs with [TODO.md](TODO.md) (active list) and [docs/roadmap.md](docs/roadmap.md) (full operational + shipped log).

---

## Scope

**Personal-use card-grading SaaS.** The user is grading their own cards through this system, not running a commercial service. Drop Stripe / marketplace / kiosks / slabbing / public accuracy benchmark / formal SLO+PagerDuty. Weight work by *"does this make the grader more accurate / reliable / usable for one person."*

Memory: `~/.claude/projects/c--Users-Belme-Documents-GitHub-project-grader/memory/project_personal_use.md`.

Working directory: `c:\Users\Belme\Documents\GitHub\project-grader`. Branch: `main`. Working tree clean as of `e6c0cca`.

---

## Architecture at a glance

```
apps/
  api/    FastAPI + SQLAlchemy + Celery, async, Postgres + pgvector
  web/    Next.js 14 (app router), Clerk auth, Vitest + RTL
ml/       Python package; pipelines, training trainers, evaluation, scripts
infra/    docker-compose dev stack
.github/  GitHub Actions CI (3 jobs: ml, api-unit, web-typecheck)
```

`apps/api` depends on `ml/` via uv editable install (`[tool.uv.sources] grader-ml = { path = "../../ml", editable = true }`).

User flow:

1. `/grade` (web) → wizard captures 8 shot kinds (front, back, 4 corners, flash, tilt) → presigned-POST to S3 → quality gate
2. `POST /submissions/{id}/submit` → Celery `process_submission` task
3. Pipeline: detect+dewarp → identification (DinoV2 + pgvector) → centering → edges → 7-detector counterfeit ensemble → grade compose
4. `/cert/{id}` (public, ISR-cached) renders the result with full TAG-inspired surface

---

## Working features

### Capture & ingestion
- 8-shot wizard with camera + file-input fallback, per-shot quality feedback, retake flow
- Real `@clerk/nextjs` JWT auth on web with dev-mode `NEXT_PUBLIC_DEV_CLERK_ID` fallback
- Rate limits via slowapi: 60/min/IP for cert reads, 5–60/min/user for writes
- 25 MiB upload cap via `generate_presigned_post` `content-length-range` condition
- Server-side defense-in-depth: `head_object` ContentLength check on register, 413 on exceed

### Pipeline (per submission)
- Quality gate (per-shot blur/glare/perspective thresholds in `ml/pipelines/quality/report.py:THRESHOLDS_BY_SHOT`)
- Card detection + dewarp (YOLOv11-seg + Canny/contour fallback) → canonical 750×1050
- Identification: pHash/dHash + DinoV2 ViT-B/14 embedder against pgvector reference catalog
- Centering grading (geometric, ~99% by construction)
- Edges grading v1 (MAD anomaly, worse-face dominates)
- **7/7 counterfeit detectors**:
  - #1 k-NN top-3 reference (`ml/pipelines/counterfeit/knn_reference/`) — replaces planned siamese
  - #3 holographic-parallax (`ml/pipelines/counterfeit/holographic/`) — Farnebäck flow + foil mask
  - #4 FFT print-rosette
  - #5 typography (`ml/pipelines/counterfeit/typography/`) — RapidOCR + Levenshtein vs identified card name
  - #6 substrate (`ml/pipelines/counterfeit/substrate/`) — paired-flash differential b\* on white-border ROI
  - #7 embedding-anomaly (centroid distance)
  - + CIELAB color profile
- Recalibration tool (`ml/evaluation/counterfeit_recalibration/`) covers all 7; ratchets thresholds when a labeled corpus exists

### Cert page (`/cert/[id]` — public, no-auth, ISR-cached)
- Identified-card block (manufacturer / variant / name / set / number)
- Per-criterion grades + amber "preliminary" banner when corners/surface haven't trained yet
- Confidence band ("Final 8.5, likely 7.7–9.3 at 85%")
- Damage-heatmap region overlay (`<DamageHeatmap>`)
- DINGS-style itemized defect list (`<DefectList>`, severity-color dots)
- Population stat ("1 of 47 graded · #3 highest · #12 chronologically")
- **Card Vision opacity slider** (`<CardVisionSlider>`, crossfades standard ↔ flash) — backed by 1-hour presigned-GET URLs
- Per-detector authenticity breakdown (all 7)
- Rich OG meta + dynamic 1200×630 OG image at `/cert/[id]/opengraph-image` — renders rich previews on Discord/X/Slack
- Cert share card: QR code (encodes canonical URL), copy-cert-ID, copy-URL, native share + dropdown
- `/cert/[id]/print` — auto-triggers `window.print()` on load
- "Request manual review" mailto stub at `reviews@grader.app`
- Cache-Control: `max-age=2400` (40 min, no SWR — keeps presign TTL margin)

### Production hardening
- Submit broker-failure: revert PROCESSING → CAPTURING + 503 `{reason: broker_unavailable, retry_after: 30}`
- Orphan-task reconciler beat: `SELECT FOR UPDATE SKIP LOCKED`, 2-min cadence, 5-min stale cutoff
- Sentry SDK (FastAPI + Celery + SQLAlchemy integrations) opt-in via `SENTRY_DSN`
- HTTP timer middleware → `http.request` structlog line per request (single-field p95 source)
- Per-stage `bound_contextvars` → `pipeline.stage.completed` log per heavy stage
- `email_verified` Clerk claim gate (rejects self-issued unverified accounts)
- ML deps cached per Celery worker process via `@worker_process_init.connect` (saves 8–15s/submission warm)
- `presigned_post_for_shot` swallows `(ClientError, BotoCoreError)` → typed `StoragePresignError` → 503 `{reason: storage_unavailable}`
- Shot-kind spoofing closed: `register_shot` requires `kind` in body, validates against s3_key

### Data flywheel (4-step `daily_cycle.sh`)
1. `psa_daily_ingest.py` — drains PSA Public API daily budget into `~/psa_data/`
2. `embed_references.py` — embeds new manufacturer references into `~/manufacturer_refs/reference_embeddings.npz`
3. `accumulate_psa_exemplars.py` — appends matched-variant embeddings as authentic exemplars (idempotent by cert_id)
4. `sync_catalog_db.py` — JSONL + npz → `card_sets` + `card_variants` upserts (computes phash on the fly; idempotent via `INSERT ... ON CONFLICT DO UPDATE`); skips with rc=0 when `DATABASE_URL` is unset

Per-step rc surfaced in `~/psa_data/launchd.stdout.log`. Plist hands off to `daily_cycle.sh`.

### ML training (skeletons committed, real training data-blocked)
- Corners (`ml/training/trainers/corners.py`) — EfficientNet-V2-S regression head, refuses below 200 samples
- Surface (`ml/training/trainers/surface.py`) — SegFormer-style decoder; `all_background_mask_loader` is the only stub
- Identification (`ml/training/trainers/identification.py`) — DinoV2 + triplet loss; supervised metric learning **unblocked** via name+set positive sampler in `ml/training/datasets/psa_identification.py`
- Detection (`ml/training/trainers/detection.py`) — YOLOv11-seg orchestrator over `ultralytics.YOLO.train`; manifest builder ready

### Test / DevOps
- 471 ml tests, ~150 apps/api tests (~108 `requires_postgres` + ~91 unit-only)
- ~10+ web Vitest + RTL tests
- Hypothesis property tests: typography monotonicity, embedding-anomaly monotonicity
- JSONB schema-parity test (catches silent JSON↔JSONB drift)
- GitHub Actions CI: 3 jobs, pinned actions, `concurrency.cancel-in-progress`

---

## State snapshot

| Component | Status |
|---|---|
| Counterfeit ensemble | ✅ 7/7 wired |
| Cert page surface | ✅ Full TAG-inspired (heatmap, DINGS, pop, Card Vision, OG, QR, print) |
| Production hardening | ✅ Sentry, rate limits, upload cap, reconciler, broker-failure 503 |
| Auth | ✅ Real Clerk SDK + `email_verified` gate |
| CI | ✅ GitHub Actions 3-job pipeline |
| Daily flywheel (code) | ✅ 4-step `daily_cycle.sh` |
| Daily flywheel (running on Mac) | ⚠️ Plist redeploy + `rapidocr-onnxruntime` install queued |
| Corners trainer | ⚠️ Skeleton; calendar-blocked on corpus ≥ 200 |
| Surface trainer | ⚠️ Skeleton; blocked on hand-labeled defect masks |
| Identification trainer | ⚠️ Skeleton; supervised metric learning unblocked, awaits real run |
| Detection trainer | ⚠️ Skeleton; manifest regen + GPU run pending |
| Counterfeit threshold recalibration | ⚠️ Tool ready; needs ~300 labeled cards |

---

## Active TODO (mirrors [TODO.md](TODO.md))

### Now — code-tractable, high personal utility

- **TCGplayer pricing comps on the cert (M)** — free-tier API. `<PricingSection>` on cert page below population stat. Hide when `identified_card is None`. ~2-3 days.
- **Capture-guidance tuning (S, ongoing)** — you're the test user. Tune `THRESHOLDS_BY_SHOT` in `ml/pipelines/quality/report.py` based on retake/rejection rates as you grade your own cards.

### Soon — operational / data-bound

- **Mac-side flywheel ops** — `launchctl unload && cp && launchctl load` the new plist; `pip install --user 'rapidocr-onnxruntime>=1.4'` so typography fires; smoke `launchctl start`.
- **Counterfeit recalibration with real labeled corpus** — operational; needs you to source/curate. Tool runs in authentic-only mode against `~/psa_data/scraped.jsonl` already; two-sided unlocks with a `--csv` of counterfeits.
- **Surface defect masks** — incremental. As you grade cards with visible defects, drop `<image>.mask.png` next to the front. ~100 unblocks real surface training.

### Calendar-blocked (PSA daily ingest is running)

- Real corners trainer (corpus ≥ 200, ~3-5 days from PSA ingest start)
- Real identification trainer (sufficient name+set keys)
- Real detection trainer (manifest regen + GPU)
- PSA paid API tier upgrade — only if 100/day becomes the bottleneck

### Sit-with-me (no rush)

- Preliminary-grade UX stance — keep amber banner or ship `min(centering, edges)` as labeled "preliminary final"
- JWT email-claim policy — overwrite or freeze on first-seen
- `reference_embeddings.npz` deployment path — local-disk fine until/unless self-hosting on AWS

### Eventually — keep on the list

- Mobile app (RN/Expo)
- eBay sold-listings comp widget on cert (L)
- Terraform dev/prod env
- Latency SLO + Datadog dashboards
- Public accuracy benchmark — only if sharing publicly

### Dropped (per personal-use scope)

Stripe, marketplace, kiosks, slabbing, formal SLO+PagerDuty, multi-tenant rate-limit Redis backend, public monthly accuracy benchmark.

---

## Key file pointers

### Cert / authentication
- `apps/api/grader/routers/cert.py` — public cert endpoint
- `apps/api/grader/routers/submissions.py` — owner-side submit + shot endpoints
- `apps/api/grader/services/auth.py` — Clerk JWT verification + dev mode
- `apps/api/grader/services/storage.py` — presigned POST/GET, `head_shot`
- `apps/api/grader/schemas/submissions.py` — Pydantic types (CertificatePublic, RegionScore, etc.)
- `apps/web/lib/cert.ts` — TS mirrors of the API types
- `apps/web/app/cert/[id]/page.tsx` — main cert page
- `apps/web/components/cert/{DefectList,CardVisionSlider,CertShareCard}.tsx`
- `apps/web/app/cert/[id]/opengraph-image.tsx` — dynamic OG image
- `apps/web/app/cert/[id]/print/page.tsx` — print-friendly view

### Pipeline / counterfeit
- `apps/api/grader/workers/pipeline_runner.py` — stage orchestration
- `apps/api/grader/workers/grading_pipeline.py` — Celery task entry, deps cached per worker
- `apps/api/grader/workers/reconciler.py` — orphan-task beat
- `apps/api/grader/services/counterfeit.py` — service wrappers for all 7 detectors
- `ml/pipelines/counterfeit/{rosette,color,embedding_anomaly,typography,holographic,knn_reference,substrate}/` — detector implementations
- `ml/pipelines/counterfeit/ensemble.py` — thresholds + verdict combiner
- `ml/evaluation/counterfeit_recalibration/` — threshold tuning tool

### Data flywheel
- `ml/scripts/daily_cycle.sh` — 4-step launchd entry point
- `ml/scripts/{psa_daily_ingest,embed_references,accumulate_psa_exemplars,sync_catalog_db}.py`
- `ml/scripts/com.projectgrader.psa-ingest.plist` — launchd job definition

### ML training
- `ml/training/trainers/{corners,surface,identification,detection}.py`
- `ml/training/datasets/psa_identification.py` — name+set positive sampler

### Tests
- `apps/api/tests/{test_cert_endpoint,test_submit_endpoint,test_pipeline_runner,test_reconciler,test_observability,test_counterfeit_service}.py`
- `apps/api/tests/test_migration_parity.py` — JSONB schema-parity check
- `apps/web/__tests__/{grade-page,cert-share-card,defect-list,card-vision-slider,cert-og-image}.test.tsx`
- `ml/tests/test_*property*.py` — hypothesis monotonicity tests

### CI / config
- `.github/workflows/ci.yml`
- `apps/api/pyproject.toml` — `requires_postgres` marker, uv editable install of `grader-ml`
- `apps/web/vitest.config.ts`

---

## How to run things

```bash
# ml/ tests (Windows venv)
cd ml && .venv/Scripts/python.exe -m pytest -q

# apps/api tests, no Postgres needed
python -m uv run --project apps/api pytest apps/api/tests -q -m "not requires_postgres"

# apps/api tests with Postgres on localhost:5432
python -m uv run --project apps/api pytest apps/api/tests -q

# apps/web type-check (Windows; pnpm may need npm install --legacy-peer-deps fallback)
cd apps/web && pnpm tsc --noEmit

# apps/web tests
cd apps/web && pnpm test

# Full daily flywheel smoke (from repo root, on Mac)
PROJECT_ROOT="$PWD" ml/scripts/daily_cycle.sh

# Counterfeit recalibration in authentic-only mode
cd ml && python -m evaluation.counterfeit_recalibration --psa-authentics ~/psa_data/scraped.jsonl
```

---

## Memory pointers (cross-session)

In `C:\Users\Belme\.claude\projects\c--Users-Belme-Documents-GitHub-project-grader\memory\`:

- `MEMORY.md` — index
- `feedback_parallel_agent_workflow.md` — User absorbs 20+ commits/session via parallel agent fan-out + cherry-pick integration. Default to fan-out for ≥3 independent items; don't ask permission. Cherry-pick over PR review. Conflicts resolved by integrator (me), not punted back.
- `project_personal_use.md` — project-grader is for the user's own card-grading, not commercial. Drop monetization features; weight work by personal utility.

---

## Recent commits (last 15)

```
e6c0cca docs: TODO — re-prioritize for personal-use scope
b2c3276 docs: TODO — mark 3 TAG-inspired cert features shipped
77e61b4 feat: Card Vision opacity slider on public cert page
de1799e feat: TAG-style population stat on public cert page
ea5206c feat: DINGS-style itemized defect rationale on public cert page
b2ddfb7 docs: TODO — TAG-inspired cert backlog
6c4ecef feat: cert share infrastructure — TAG-style stickiness
1f08bbd docs: TODO — counterfeit ensemble 7/7 + audit cleanup
d3ed0ec chore: address four small audit findings deferred from earlier planning
4cfe455 feat: substrate / paper-fluorescence detector — 7/7 ensemble milestone
4a06156 feat: damage heatmap overlay on public cert page
aacdf80 chore(ci): GitHub Actions workflow + requires_postgres pytest marker
ca27507 fix(web): clear TS18048 errors on cert page primary grade
a4e45fd test: add JSONB schema-parity check, web Vitest smoke, hypothesis property tests
5db7208 docs: TODO — counterfeit ensemble 6/7 + production hardening
```

For full history: `git log --oneline` from the repo root.
