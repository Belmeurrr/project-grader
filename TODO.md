# TODO

Action-oriented checklist of what's left. Full operational snapshot in [docs/roadmap.md](docs/roadmap.md). Strategic plan (4-phase business roadmap, MVP scope, accuracy targets) lives at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md`.

Each item has enough context to pick it up cold without reloading session state.

---

## Status snapshot (2026-05-04, late session)

Counterfeit ensemble is now **7/7 wired** end-to-end (FFT rosette + color profile + embedding-anomaly + typography + holographic-parallax + k-NN top-3 reference + substrate). Production-readiness landed: orphan reconciler beat, 25 MiB upload cap, slowapi rate limits, Sentry + p95-able HTTP timer + per-stage pipeline logs, `email_verified` Clerk claim gate, MissingGreenlet test-fixture bug fixed, damage-heatmap on cert page, GitHub Actions CI + `requires_postgres` marker, JSONB schema-parity test, web Vitest smoke, hypothesis property tests. Manufacturer reference flywheel is live for MTG (Scryfall paginated + bulk) and Pokemon (PokemonTCG.io). Daily data flywheel is a **4-step** launchd chain (ingest → re-embed references → accumulate exemplars → catalog DB-side sync). All four ML-head trainer skeletons committed; identification trainer just got supervised metric learning unblocked (no new data needed). Public cert page now renders Phase-1 partial-grade certs without crashing (was a P0). Submit endpoint surfaces broker outages instead of stranding. Web frontend wired to real Clerk JWTs (with dev-mode fallback for tests). Capture wizard now offers optional flash + tilt shots so future detectors get archived data immediately. apps/api → ml/ imports go through a proper editable install (no more `parents[4]` footgun). Counterfeit-threshold recalibration tool covers all 4 detectors. The system is ~35% to production MVP; ML heads stay data-blocked on PSA corpus growth + counterfeit-slice sourcing.

### What's shipped (production-ready)

| Component | State | Notes |
|---|---|---|
| Quality gating | ✅ Production | Per-shot-kind blur/glare/perspective thresholds |
| Card detection + dewarp | ✅ Production | YOLOv11-seg + Canny+contour fallback; canonical 750×1050 |
| Card identification | ✅ Production | pHash/dHash + DinoV2 embedder; pgvector NN |
| Centering grading | ✅ Production | Geometric, ~99% by construction |
| Edges grading v1 | ✅ Production | MAD anomaly; worse-face dominates |
| Counterfeit rosette | ✅ Production (synthetic) | FFT halftone signature |
| Counterfeit color | ✅ Production (synthetic) | CIELAB chroma after WB calibration |
| Counterfeit embedding-anomaly | ✅ Production (limited) | Gracefully abstains on no-refs |
| Manufacturer reference flywheel | ✅ Production | Scryfall (paginated + bulk) + PokemonTCG.io |
| PSA daily ingest + exemplar accumulator | ✅ Production | 3-step launchd chain via `daily_cycle.sh` |
| Counterfeit threshold recalibration tool | ✅ Production | `evaluation.counterfeit_recalibration` |
| Web capture wizard | ✅ MVP | 6-shot flow (front + back + 4 corners) |
| Public cert page + identified card | ✅ MVP | ISR; `(manufacturer, variant_id, name, set, no.)` surfaced |
| Celery pipeline orchestrator | ✅ Production | Soft-fail on missing stages |
| Postgres schema + migrations | ✅ Production | 8 tables; AST parity test |
| S3 / MinIO integration | ✅ Production | Presigned URLs |

### What's MVP / skeleton (works, rough edges)

| Component | State | Notes |
|---|---|---|
| Counterfeit ensemble | ✅ MVP (7/7) | Rosette + color + embedding-centroid + typography + holographic + k-NN top-3 + substrate all wired |
| Web UI | ⚠️ MVP | No design polish; no Clerk integration |
| Corners trainer | ⚠️ Skeleton | Refuses to train < 200 samples |
| Surface trainer | ⚠️ Skeleton | Default mask loader returns background |
| Identification trainer | ⚠️ Skeleton | Supervised metric learning unblocked via name+set sampler; awaits real training run |
| Detection trainer | ⚠️ Skeleton | Manifest builder ready; not trained |
| Edges grading | ⚠️ MVP | Only synthetic-calibrated confidence |

### What's stubbed / missing

| Feature | Gap | Impact |
|---|---|---|
| Corners + surface grading models | Real training data | Pipeline returns NULL grades |
| Stripe payments | Integration not wired | Blocks paid-tier feature |
| TCGplayer pricing comps | Integration not wired | No EV calculations |
| Clerk JWT swap (web) | ✅ wired (commit c37cf03); flip API `dev_auth_enabled=false` in prod env | (no longer dev-only) |
| Terraform prod infra | AWS account setup | Only dev docker-compose exists |
| Mobile app | Phase 2 | Out of MVP scope |
| Real-data threshold calibration | ~300 labeled cards | Tool ready; corpus needed |
| Catalog DB-side ingest | ✅ shipped (sync_catalog_db.py, daily_cycle step 4); awaits first prod run | (no longer empty) |
| MLflow model registry | No deployment plumbing | Weights local-only |
| GDPR data export / RTBF | Not implemented | Needed for paid tier |

---

## Scope (2026-05-06)

**Personal-use project**, not a commercial SaaS. The user is grading their own cards through this system — no paying customers, no marketplace, no scaling pressure. Items are weighted by *"does this make the grader more accurate / reliable / usable for one person grading their own cards"* rather than commercial fit. See `~/.claude/projects/.../memory/project_personal_use.md`.

---

## Now — code-tractable, high personal utility

- [x] **Fix alembic migration — enum double-creation on fresh DB** ✅ Shipped 2026-05-07. `create_type=False` on the five `postgresql.ENUM(...)` constructors in `apps/api/alembic/versions/20260428_0001_initial_schema.py` so the explicit `enum.create(bind, checkfirst=True)` loop is the only creation site. Verified end-to-end against a fresh `grader_alembic_test` DB (alembic upgrade head → 0002, 8 tables created clean). The previous workaround (`Base.metadata.create_all`) is no longer needed; the README quick-start now just runs `alembic upgrade head`.

- [x] **Home-page CTA → /grade** ✅ Shipped via the Workbench port (commit f70731e). The new `/` is the Workbench dashboard; the empty-state and hero both link to `/grade`.

- [ ] **Home-page CTA → /grade** (XS) — superseded
  - `apps/web/app/page.tsx` is pure marketing — no nav links anywhere. The only entry point to the wizard is typing `/grade` in the URL. Add a "Start grading" button below the headline that links to `/grade`. Five-minute fix; meaningful friction reduction every time you grade a card.

- [ ] **TCGplayer pricing comps on the cert** (M)
  - Free-tier API, simpler than eBay. Shows current market price for the identified variant on the cert page.
  - Lookup keyed on `(IdentifiedCard.{set, number, name}, primary.final)` → pulls "current market price" + maybe "median price by grade." Cache aggressively (24-hour TTL — these don't move that fast).
  - Adds a `<PricingSection>` component on the cert page beneath the population stat. Hide entirely when `identified_card is None` (no card to price) or when the API call fails (graceful degrade).
  - Useful for personal use: knowing "this Pikachu V is worth $X at PSA 9" is the whole point of grading your own collection.

- [ ] **Capture-guidance tuning** (S, ongoing)
  - You're the test user — walk a few of your own cards through the wizard, log retakes and rejections. If quality gates feel too aggressive (or too lax), tune `THRESHOLDS_BY_SHOT` in `ml/pipelines/quality/report.py`. Pure operational work; pairs with running real grades on real cards.

---

## Soon — operational / data-bound

- [ ] **Mac-side daily flywheel ops** (S, queued)
  - On the Mac prod box: `launchctl unload && cp && launchctl load` the new 4-step plist. Then `pip install --user 'rapidocr-onnxruntime>=1.4'` so the typography detector fires instead of UNVERIFIED-abstaining. Smoke-test via `launchctl start com.projectgrader.psa-ingest`; watch `~/psa_data/launchd.stdout.log` for per-step rc lines.
  - Without this, the daily flywheel doesn't pick up the catalog DB sync or the substrate / typography detector deps. Everything else depends on it.

- [ ] **Counterfeit recalibration with real labeled corpus** (operational, blocked on you)
  - Tool ships (`evaluation.counterfeit_recalibration`); now covers all 7 detectors. Authentic-only mode usable today against `~/psa_data/scraped.jsonl`; two-sided mode unlocks once you supply a `--csv` of counterfeits.
  - For personal use: as you grade your own collection, your authentic exemplars accumulate. If you also get a few cheap fakes from eBay (or use known-counterfeit images from forums), you can flip to two-sided mode and lock real thresholds.
  - Run, paste the patch into `ml/pipelines/counterfeit/ensemble.py`, lock + version in `model_versions`.

- [ ] **Surface trainer — incremental defect-mask labeling** (S per card, accumulates)
  - When you grade a card with visible scratches/dents, hand-paint a defect mask in LabelMe / CVAT and drop it next to the image as `<image>.mask.png`. The default `all_background_mask_loader` already accepts this drop-in. ~100 masks unblock real training; you don't have to do them all in one sitting.

---

## Calendar-blocked (PSA daily ingest is running, watch for thresholds)

- [ ] **Real corners trainer run** — needs corpus ≥ 200 samples. Current ingest rate ~30-50/day. Goal of the first run: prove the pipeline plumbs through to a torch step. Real val signal needs ~1k samples.
- [ ] **Real identification trainer run** — needs sufficient (name, set) key coverage in PSA exemplars. The supervised positive sampler ships (commit ece3c94); just needs data.
- [ ] **Real detection trainer first run** — operational, GPU + manifest regeneration via `build_detection_manifest.py`. Synthetic-only baseline before user-shot labeling.
- [ ] **PSA paid API tier upgrade** — only if 100/day becomes a bottleneck. Email `webcert@collectors.com`.

---

## Sit-with-me decisions (no rush — personal use)

- [ ] **Preliminary-grade UX stance** — currently the cert renders an amber "preliminary subgrades — final grade unavailable" banner when corners/surface haven't trained yet. Decide: ship `min(centering, edges)` as a labeled "preliminary final" or hold the line until all 4 trainers ship.
- [ ] **JWT email-claim policy** — overwrite `User.email` on re-login or freeze on first-seen. GDPR angle is mostly moot for personal use, but worth a deliberate decision.
- [ ] **`reference_embeddings.npz` deployment path** — local disk works fine on the Mac. Only matters when/if you self-host on AWS.

---

## Eventually — keep on the list, no rush

- [ ] **iOS app — Capacitor wrap of the existing wizard** — phone capture works *today* via Tailscale-served HTTPS (see README §6b: `infra/tailscale/serve-dev.ps1`); the wizard already requests rear camera and `<input capture>`. The native step is wrapping `apps/web` with [Capacitor](https://capacitorjs.com/) (or Tauri Mobile) so the same React app ships as a real iOS bundle: native camera plugin replaces `getUserMedia` (better autofocus + RAW access), files PUT to the same API. Cost: Apple Developer account ($99/yr), TestFlight or sideload — App Store review only matters if distributing publicly. RN/Expo would mean re-implementing the wizard; Capacitor avoids that. PWA "Add to Home Screen" is the cheap intermediate.
- [ ] **eBay sold-listings comp widget on cert** (L) — needs eBay Browse / Marketplace Insights API + cache layer + 130point licensing decision. Bigger lift than TCGplayer; would be a major differentiator vs TAG/PSA but personal-use value is "nice-to-have."
- [ ] **Terraform dev/prod env** — only needed if/when you decide to host the API somewhere other than your Mac. Currently docker-compose is sufficient.
- [ ] **Latency SLO + Datadog dashboards** — Sentry covers personal-use observability needs. SLOs matter when paying customers are watching.
- [ ] **Public accuracy benchmark** — only relevant if you decide to share grades publicly as a trust signal.

---

## Operational hygiene (continuous)

- [x] **Counterfeit benchmark — v1 corpus + harness** (committed 6823e48). Run: `python -m evaluation.counterfeit_benchmark`. Current: 50-sample synthetic corpus, rosette 98% accuracy / color 100% / ensemble 98%. **Still TODO**: extend with embedding-anomaly cases (needs labeled identified-variant samples), swap in real images as the corpus grows.
- [x] **Manufacturer reference re-embed cycle** — runs as step 2 of `ml/scripts/daily_cycle.sh` (faea7d6). Idempotent on `(manufacturer, variant_id)`; a no-op on days with no new ingest.
- [x] **GitHub Actions CI** — shipped (aacdf80). 3 jobs: ml, api-unit (with `requires_postgres` marker), web-typecheck.
- [x] **Consolidate `_ML_ROOT` sys.path bootstrap** — shipped (287f04c) via uv editable install of `ml/` in `apps/api/pyproject.toml`.

---

## Dropped (per personal-use scope, 2026-05-06)

- ~~Stripe payments~~ — no paying customers
- ~~Marketplace surface~~ — not building a 2-sided market
- ~~Imaging kiosks (Phase 3)~~ — single-user
- ~~Full slabbing operation (Phase 4)~~ — single-user
- ~~Formal SLO + PagerDuty~~ — Sentry covers solo use
- ~~Multi-tenant rate-limit Redis backend~~ — in-memory slowapi is fine
- ~~Public accuracy benchmark monthly publication~~ — only relevant if commercial

---

## Dependency graph for upcoming milestones

```
Daily flywheel running on Mac (ops, you)
└─ PSA corpus grows + catalog DB-side sync runs
   ├─ Corpus ≥ 200 samples → first real corners trainer run
   ├─ (name, set) coverage → first real identification trainer run
   └─ Per-variant exemplars accumulate → embedding-anomaly + k-NN reference confidence ramps

You source / label a counterfeit slice (operational)
└─ Recalibration tool flips to two-sided mode
   └─ Lock real thresholds across all 7 detectors

You hand-paint surface defect masks as you grade
└─ Surface trainer flips from skeleton to real (incremental)

(Independent code work)
TCGplayer comps on the cert (M, ~2-3 days)
Capture-guidance tuning (S, ongoing as you grade your own cards)
```

---

## Recently shipped

### 2026-05-06
- **Substrate / paper-fluorescence counterfeit detector (#6 of 7)** (`feat: substrate / paper-fluorescence detector — 7/7 ensemble milestone`, 4cfe455). **Counterfeit ensemble now 7/7 wired.** New `ml/pipelines/counterfeit/substrate/`. Algorithm: paired-flash differential b\* on the white-border ROI (outer 4% ring, corners excluded). Exposure-normalize by matching median L\* between flash and no-flash, then `delta_b = median(b*_flash[border]) - median(b*_no_flash[border])`. Logistic squash midpoint=-3, slope=0.4 (deliberately gentle — phone flash is UV-poor). Full confidence at MAD<1.0, ramps to 0.4 at MAD≥3.0, **hard cap at 0.7** to reflect the algorithmic ceiling vs UV-lamp ground truth. Five abstain reasons (`flash_not_captured`, `shape_mismatch`, `invalid_image`, `border_too_small`, exception swallow). FRONT_FULL_FLASH added to OPTIONAL_SHOTS so dewarp soft-fails on it. Wired through ensemble.py + recalibration tool + pipeline_runner stage 3.5 + `detector_scores["substrate"]` persistence + audit log fields. 15 detector + 5 service + 1 pipeline tests, 471 ml total. Documented limits: phone flash UV-poor; ceiling ~60-75% on cheap fakes; useless on super-fakes printed on brightener-free offset stock; foil-exclusion gate is out of scope (no foil detector exists yet).
- **Audit cleanup bundle** (`chore: address four small audit findings deferred from earlier planning`, d3ed0ec). Four small fixes in one commit. (a) Deleted duplicated `load_canonical_bgr` in `services/identification.py`; consumer now uses the canonical `storage.load_canonical_bgr`. (b) `services/detection.py` no longer parses S3 keys via `split("/")[1]` index — uses regex `^submissions/(?P<id>[0-9a-f-]{36})/` with explicit failure mode. (c) Shot-kind spoofing closed: `register_shot` now requires `kind` in the request body and validates it matches the kind segment in the s3_key; clients can no longer mix shot_id from one presign with s3_key from another. (d) `presigned_post_for_shot` wraps `(ClientError, BotoCoreError)`, logs structured detail, re-raises as typed `StoragePresignError`; route handler maps to 503 with `{reason: storage_unavailable, retry_after: 30}` matching the broker-failure shape.

### 2026-05-05 (late session — 5-commit production-hardening batch)
- **k-NN top-3 reference counterfeit detector (#1 of 7)** (`feat: k-NN top-3 reference counterfeit detector — 6/7 ensemble milestone`, 80a43de). Replaces the planned siamese-network detector with mean cosine distance to top-3 nearest authentic exemplars (vs. centroid distance for #7). Catches the manifold case centroid distance misses — authentic exemplars cluster on a manifold (different angles/lighting), submissions on the manifold but far from centroid score "anomalous" today; submissions inside centroid radius but far from any actual exemplar score "fine" today. New module `ml/pipelines/counterfeit/knn_reference/`. Logistic squash midpoint=0.25, slope=15.0; abstain UNVERIFIED below `n=3` references. Wired through ensemble.py (KNN_REFERENCE_AUTHENTIC=0.65, COUNTERFEIT=0.35, MIN_CONFIDENCE=0.4 placeholders), recalibration coverage, pipeline_runner stage 3.5, persisted under `detector_scores["knn_reference"]`. 23 detector tests + 6 service + 1 pipeline. **Counterfeit ensemble now 6/7 wired** — only substrate (#6) remains, and that's data-blocked.
- **Minimum-viable observability** (`feat: minimum-viable observability — Sentry + HTTP timer + pipeline-stage logs`, e1c80d4). Three additions, no new infra: (a) `sentry_sdk.init(...)` in lifespan with FastAPI + Celery + SQLAlchemy integrations, opt-in via `SENTRY_DSN` env. (b) `@app.middleware("http")` Timer emitting `http.request` structlog line per request — single field is enough for p95 in any log aggregator. (c) `bound_contextvars(stage=...)` + `pipeline.stage.completed` log per heavy stage in pipeline_runner. Free-tier Sentry covers first 10 users.
- **Upload size cap + rate limiting** (`feat: enforce 25 MiB upload size cap + slowapi rate limits on public/write endpoints`, 7d8bf1a). Switched to `generate_presigned_post` with `Conditions=[["content-length-range",1,settings.submission_max_image_bytes]]`. Updated apps/web `requestShotUploadUrl + uploadShot` to multipart-POST shape. Server-side defense-in-depth via `head_object` ContentLength check on register; 413 on exceed. slowapi (in-memory) added to apps/api with `user_or_ip_key`: `GET /cert/{id}` 60/min/IP, `POST /submissions` 10/min/user, `/submit` 5/min/user, upload-url + register 60/min/user. Drive-by `select(...).options(...)` migration on routers (different strategy than D's `populate_existing=True` — both work; mine landed first via cherry-pick).
- **MissingGreenlet test-fixture unblock** (`test: unblock MissingGreenlet fixture bug — expire identity map per request + populate_existing on db.get`, dd9d7ab). 50 apps/api tests had been failing because `db.get(..., options=[selectinload(...)])` cache-hits the in-flush identity map and silently drops eager-load options; lazy-load fires inside the request handler's async scope → MissingGreenlet. Fix combines `db_session.expire_all()` per-request override (conftest) with `populate_existing=True` on the get-callsites. ~50 tests recovered.
- **Orphan-task reconciler beat** (`feat: orphan-task reconciler beat — flips wedged PROCESSING rows to FAILED`, 4dcec3c). Complements today's broker-failure 503 (synchronous failures); reconciler handles worker-died-mid-pipeline (asynchronous failures). New `apps/api/grader/workers/reconciler.py` with `SELECT ... FOR UPDATE SKIP LOCKED` and a separate `maintenance` queue. 2-min beat cadence; 5-min stale cutoff (~10x p99). Marks stuck rows FAILED with `rejection_reason="worker_timeout_reconciled"` + audit log entry. 5 tests (Postgres available locally).

### 2026-05-05
- **P0 ship-blockers bundled** (`fix: P0 ship-blockers — schema drift + GradingScheme enum + email_verified gate`, 30ee366). Three production bugs in one commit. (a) `AuthenticityOut.detector_scores` was `dict[str, float]` but the column persists `dict[str, dict]` after the typography + holographic detectors landed — owner-side `GET /submissions/{id}` would 500 the moment a populated row existed. Relaxed to `dict[str, dict[str, Any]]`; regression test added. (b) `apps/web/lib/cert.ts` GradingScheme TS enum had `"sgc"|"cgc"` (which never appear server-side) and was missing `"trugrade"`. Mirrored to the actual API enum. (c) `_authenticate_clerk` was auto-creating User rows from JWTs without inspecting the `email_verified` claim — self-issued Clerk accounts could submit jobs. Now raises 401 `email_not_verified` if the claim is absent or false; dev-mode path unchanged.
- **ML deps cached per Celery worker process** (`perf: cache ML deps per Celery worker process — saves 8-15s/submission`, 7131f20). DinoV2 + reference embeddings + pgvector catalog were rebuilt inside every `process_submission` task body, eating ~8-15s of dead time before stage 1. Now constructed once via `@worker_process_init.connect` and cached on a module-level global; `_get_production_dependencies()` lazy-builds as a fallback for tests. Per-task state was already cleanly separated (DB session opens inside `run_pipeline_sync`); nothing needed peeling out. ~40-75% reduction in p95 latency for warmed workers — first task on a fresh worker still pays cold-start; every subsequent task pays zero.
- **Holographic-parallax counterfeit detector (#3 of 7)** (`feat: holographic-parallax counterfeit detector — 5/7 ensemble milestone`, a169349). Detector #3 of 7. New module `ml/pipelines/counterfeit/holographic/`. Dense Farnebäck optical flow between dewarped front + tilt_30 canonicals, restricted to a foil mask (CIELAB chroma > 30 AND HSV saturation > 80, opened with a 3×3 ellipse). Score is logistic-squashed flow-ratio inside-vs-outside the mask: ≥3× ratio → real holo (~0.95), ~1× ratio → flat fake (~0.05). Graceful UNVERIFIED on tilt_not_captured / no_holo_region (mask < 2%) / shape_mismatch / flow_computation_failed. Wired through ensemble.py (HOLOGRAPHIC_AUTHENTIC=0.65, COUNTERFEIT=0.35, MIN_CONFIDENCE=0.4 — placeholders, recalibration tool ratchets later), recalibration coverage, pipeline_runner stage 3.5 (TILT_30 added to OPTIONAL_SHOTS so dewarp soft-fails on it). 13 detector tests + 7 service + 1 end-to-end pipeline_runner test. **Counterfeit ensemble now 5/7 wired.** Documented limits: false-negatives on silver/gold mirror foil (low chroma); false-positives on highly-saturated full-art / Secret Lair non-foil cards.

### 2026-05-04 (late session — 8-commit batch via parallel agents)
- **Typography counterfeit detector** (`feat: typography counterfeit detector — OCR title vs identified card name`, 4194511). Detector #5 of 7. New module `ml/pipelines/counterfeit/typography/`. RapidOCR (PP-OCR weights via ONNX Runtime, no Paddle dep) on the top-10% title ROI of the canonical 750×1050 dewarped front. Case-insensitive normalized Levenshtein vs. the identified card name from the identification stage. Logistic squash midpoint 0.35, slope 12: exact match → 0.985, 1-char typo → ~0.96, totally wrong → ~0.02. Graceful UNVERIFIED on no-identified-name / invalid-image / OCR-import-missing. Wired through ensemble.py (`TYPOGRAPHY_AUTHENTIC_THRESHOLD=0.65`, `_COUNTERFEIT=0.35`, `_MIN_CONFIDENCE=0.4`), recalibration tool (`CURRENT_THRESHOLDS` + `_PATCH_CONSTANT_NAMES`), pipeline_runner stage 3.5, and persisted to `detector_scores["typography"]` on AuthenticityResult. 18 detector tests + service + pipeline tests; 112 ml tests passing total.
- **Web frontend → real Clerk JWTs** (`feat: wire @clerk/nextjs SDK into web frontend with dev-mode fallback`, c37cf03). `@clerk/nextjs@7.3.0` mounted via `<ClerkProvider>` in `app/layout.tsx`. Middleware at `apps/web/middleware.ts` gates `/grade/:path*` only — `/cert/:path*` stays public. New `useAuthedFetch()` hook in `lib/submission.ts` calls `useAuth().getToken()` and emits `Authorization: Bearer <jwt>`; falls back to `Authorization: Dev <NEXT_PUBLIC_DEV_CLERK_ID>` when `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is unset (so tests/local dev keep working without Clerk keys). API-side `dev_auth_enabled` flag intentionally not flipped — toggle in production env. `.env.example` annotated.
- **Capture wizard — flash + tilt shots** (`feat: capture wizard — optional front_full_flash + tilt_30 shots`, d66834c). Two `required: false` rows added to `WIZARD_SHOTS` after the corners. Backend (ShotKind enum, quality thresholds in `THRESHOLDS_BY_SHOT`) was already ready; archives the data for the future substrate (#6) + holographic (#3) detectors that consume them. UI-only.
- **apps/api → uv editable dep on grader-ml** (`chore: apps/api → uv editable dep on grader-ml, drop sys.path bootstrap`, 287f04c). Replaced 15 `_ML_ROOT = parents[4]/"ml"` blocks (8 service/worker + 7 test files) with a real editable install via `[tool.uv.sources] grader-ml = { path = "../../ml", editable = true }` in `apps/api/pyproject.toml`. Import paths unchanged (ml's hatch wheel exposes `pipelines`, `data`, etc. at top level). 113/24 unit-pure tests green; the 46 failing tests are pre-existing pgvector/asyncpg/no-postgres infra issues, not caused by this refactor.
- **Submit endpoint surfaces broker outages** (`fix: submit endpoint returns 503 when broker unreachable`, b123a6c). On `process_submission.delay(...)` exception, revert `sub.status = SubmissionStatus.CAPTURING` and raise HTTPException(503, `{reason: broker_unavailable, retry_after: 30}`). The early-return guard at line 220 already protects against double-enqueue. Reconciler beat intentionally NOT built — that solves a different problem (worker died mid-job). Test mocks `process_submission.delay` raising and asserts the full revert + retry path.
- **Cert-page 500 fix — Phase-1 partial grades render** (`fix: relax GradeOut subgrades to nullable — unblock Phase-1 cert page 500`, bf70ca9). `GradeOut.{centering, corners, edges, final}` were non-nullable in the public Pydantic schema, but `compute_psa_final` returns None whenever corners or surface is missing — which is *every* Phase-1 cert with skeleton trainers. `model_validate(g)` was 500'ing on every cert request. Relaxed to `float | None`, mirrored in `apps/web/lib/cert.ts`, added an amber "Preliminary subgrades — final grade unavailable until corners + surface analysis ships" banner in `apps/web/app/cert/[id]/page.tsx` when `primary.final === null`. Regression test inserted.
- **Catalog DB-side ingest** (`feat: sync_catalog_db script — flywheel JSONL + npz → card_sets / card_variants`, d609c63). New `ml/scripts/sync_catalog_db.py`: pass 1 upserts unique `(game, set_code)` into `card_sets`; pass 2 streams JSONL, looks up embedding from `reference_embeddings.npz`, computes `canonical_phash` from the front image (not previously precomputed), upserts into `card_variants` keyed `(game, set_id, card_number)`. Async SQLAlchemy + `INSERT ... ON CONFLICT DO UPDATE` for idempotency. Slot 4 of `daily_cycle.sh` (skips with rc=0 when `DATABASE_URL` is unset so dev boxes don't fail the cycle). 10 unit tests on the prep layer; SQL round-trip skipped because pgvector + asyncpg aren't in ml/.venv (covered by API-side test suite against real Postgres).
- **Identification trainer — supervised metric learning unblocked** (`feat: supervised metric learning unblocked — name+set positive sampler`, ece3c94). `_sample_positive_index` previously returned `anchor_index`, so triplets only taught augmentation invariance. Now: every sample carries a normalized `(card_name, set_name)` key (lowercase + strip + collapse-whitespace; deliberately conservative — no punctuation stripping that could merge "Mewtwo" with "Mewtwo & Mew"). `key_index` lookup; singletons (no different-cert positive) are dropped. Triplet loss now sees genuine printing-variant positives. **No new training data needed** — was just a sampler swap. 100 ml tests passing.

### 2026-05-04 (mid-session)
- **Counterfeit-detector threshold recalibration tool** — `ml/evaluation/counterfeit_recalibration/` (97e3f71). Sibling to `counterfeit_benchmark`: ingests a labeled real-image corpus (`--csv` and/or `--psa-authentics scraped.jsonl`), runs it through the same ensemble runner production sees, and recommends per-detector thresholds. Two modes auto-picked from corpus contents — `two_sided` (authentics + counterfeits) uses Youden's-J on midpoint candidates + an FPR-budget cap (default ≤ 0.5%) on the LIKELY_COUNTERFEIT side; `authentic_only` (PSA-only steady state) recalibrates just the AUTHENTIC threshold and notes that COUNTERFEIT side needs fakes. Outputs a drop-in patch block for `ensemble.py` (TWO_SIDED detectors only — partial recommendations are surfaced in the report but kept out of the patch to avoid half-applying). `--json` / `--markdown` for CI / PR comments. 21 tests covering all three modes, both loaders, and end-to-end against the synthetic benchmark corpus.
- **Daily data-flywheel wrapper** — `ml/scripts/daily_cycle.sh` (faea7d6). Replaces the previous launchd inline command with a 3-step bash chain: `psa_daily_ingest` → `embed_references` → `accumulate_psa_exemplars`. Each step runs independently (the wrapper aggregates exit codes — first non-zero rc is returned, but later steps still run on earlier failure since they have value on their own). Per-step `rc=N` lines hit `launchd.stdout.log` so a failed step is triageable. Configurable knobs: `PROJECT_ROOT`, `PYTHON`, `PSA_INGEST_DATA_DIR`, `MANUFACTURER_REFS_DATA_DIR`. Sources `$PROJECT_ROOT/.env` so secrets propagate. Smoke: `PROJECT_ROOT="$PWD" ml/scripts/daily_cycle.sh`. Plist hands off to it; previous installations keep the same `Label` so a `launchctl unload && load` cycle picks up the new behavior with no rename.
- **Windows operation docs** — `ml/scripts/README.md` (b4d7a06). The wrapper is portable bash and runs unchanged under Git Bash; only the scheduler differs. Adds Task Scheduler setup via a `daily_cycle.cmd` wrapper (schtasks `/TR` is too brittle for inlining), an ops table (run/pause/resume/delete), and the two real Windows caveats (logged-in only, sleep-misses-runs). Brief WSL2+cron alternative.
- **`/cert/{id}` surfaces `identified_card`** — `apps/api/grader/routers/cert.py` + `routers/submissions.py` (832275f). The public cert payload now includes the matched `(manufacturer, variant_id, name, set_code, collector_number)` from the identification pass alongside the verdict + grade. `GET /submissions/{id}` (owner-side, authed) added to support the wizard's status polling. Keeps the public payload sanitized (no user_id / S3 keys).

### 2026-04-30
- **Web capture flow** — `apps/web/lib/submission.ts` (typed owner-side client) + `apps/web/app/grade/page.tsx` (kickoff) + `apps/web/app/grade/[id]/page.tsx` (status-aware wizard). Wizard order: front → back → TL → TR → BR → BL (clockwise). Per-shot flow: presign → PUT to S3 → register (server runs quality gate) → quality_passed advances or surfaces reasons for retake. Camera via `getUserMedia({ facingMode: "environment" })` with file-input fallback for permission-denied / iOS-quirk paths. Submit-for-grading button enables once front passes (REQUIRED_SHOTS = (FRONT_FULL,)); other shots are optional but improve the grade. PROCESSING status polls every 2s until COMPLETED, then redirects to `/cert/[id]`. Auth uses dev-mode `Authorization: Dev <clerk_id>` from `NEXT_PUBLIC_DEV_CLERK_ID` (real Clerk SDK swap is one function); no design polish per prior guidance. Typecheck clean on the new files.
- **Bulk-data Scryfall ingest path** — `data/ingestion/scryfall_bulk.py` + tests + CLI wiring. New `ingest_bulk(...)` mirrors `ingest_query` but pulls the full `default_cards` dump from `/bulk-data` in one shot (sidesteps the ~1750-result `/cards/search` soft cap). Reuses `ingest_card`, `ScryfallIngestStats`, `LocalReferenceStore` — only the fetch strategy is new. Streamed chunked-write keeps the download itself constant-memory; parsing uses `json.load` (peak ~1-1.5 GB for default_cards, swap point for ijson noted in the docstring). CLI: `python -m scripts.manufacturer_refs_ingest --bulk [--bulk-cache-dir ...] [--keep-bulk-cache] [--max-cards N]`. 14 new tests; ingestion suite (scryfall + scryfall_bulk + pokemontcg + references_storage) all green.
- **Public cert page** — `apps/api/grader/routers/cert.py` + `apps/web/app/cert/[id]/page.tsx` (460841a + 33e9c4e). Public, cacheable read of a COMPLETED submission for the `/cert/[id]` route. `Cache-Control: public, max-age=300, stale-while-revalidate=3600`. 404 (opaque) for unknown / in-progress / failed submissions. Per-detector breakdown surfaces as a typed list so adding ensemble detectors #4-7 doesn't require a schema bump.
- **Per-variant exemplar accumulator** — `data/ingestion/psa_exemplars.py` + `scripts/accumulate_psa_exemplars.py`. Walks PSA `scraped.jsonl`, identifies each record against the catalog built from on-disk references + their embeddings, appends matched submissions' embeddings to the npz under the same `(manufacturer, variant_id)` key. Idempotent by `cert_id` via sidecar `psa_exemplars_log.jsonl`. CLI: `python -m scripts.accumulate_psa_exemplars [--psa-data-dir ...] [--refs-data-dir ...]`. 12 tests; full ml suite 342/2/0.

### 2026-04-29
- **Manufacturer reference scrapers** — `data/ingestion/scryfall.py` (faab948) + `data/ingestion/pokemontcg.py` (d24691a). CLI: `python -m scripts.manufacturer_refs_ingest --source {mtg,pokemon} --query "..."`.
- **Reference embedding pass** — `data/ingestion/reference_embeddings.py` (6caa8a1). `embed_references` walks the references JSONL, embeds with the identification embedder, writes a single npz keyed by `"<manufacturer>/<variant_id>"`. CLI: `python -m scripts.embed_references --data-dir ...`. Lookup helpers: `load_embeddings`, `lookup_references`.
- **Surface trainer skeleton** — `training/trainers/surface.py` (31b7adb). EfficientNet-V2-S encoder + SegFormer-style decoder, 7-class CrossEntropyLoss. Default `mask_loader` returns all-background until real per-pixel labels exist.
- **DinoV2 identification trainer skeleton** — `training/trainers/identification.py` (41e68f2). ViT-B/14 + triplet loss. Augmentation-based positives until variant_id linkage exists. Image size must be a multiple of 14 (hard ViT constraint).
- **YOLO detection manifest builder** — `data/catalogs/build_detection_manifest.py` (09fe2a4). Synthesizes labeled scenes from `card_in_scene` fixtures with analytically-derived bboxes. Manifest at `ml/data/catalogs/detection_dataset.yaml`.
- **Embedding-anomaly counterfeit detector wired** — `services/counterfeit.py` + `workers/pipeline_runner.py` (c5af923). Stage 3.5 now runs all three detectors. Graceful UNVERIFIED on no-references / unidentified paths so the ensemble degrades cleanly.
