# Updated TODO — Audit & Priorities (2026-05-04)

Audit-driven refresh of the action list. Builds on `TODO.md` and `docs/roadmap.md` with concrete file:line evidence and a sharper priority order. Source of the audit: direct reads of `pipeline_runner.py`, `routers/submissions.py`, `routers/cert.py`, `services/{grading,identification,counterfeit}.py`, plus full ML/data-layer survey.

---

## Where we are (one-paragraph)

The system is a **functioning end-to-end greenfield grader** with one full happy path wired (capture → S3 → quality gate → detect → dewarp → identify → centering + edges + 3-of-7 counterfeit detectors → cert page) but **every ML head is intentionally a skeleton**, blocked on PSA corpus growth. ~28k LOC. 35+ test files. 342/2/0 passing in `ml/`. Roughly **30% to production MVP**; agent estimate is **4–8 weeks** to ship at current ingest rate.

---

## Concrete gaps found by direct code inspection

These are not speculation — each has file:line evidence.

### 1. Cert never shows what card it is — and the data is already in the DB

- `apps/api/grader/routers/cert.py:98` hardcodes `identified_card=None` with TODO: *"populate once IdentifiedCard write-path lands"*.
- `apps/api/grader/routers/submissions.py:250` does the same on the owner-side getter.
- But the write path **already exists**: `services/identification.py:82` writes `submission.identified_variant_id` and `submission.identification_confidence` on the Submission row.
- Missing piece: `Submission` model (`db/models.py:148–150`) has the FK column but no `relationship` to `CardVariant`, so the response mappers never join.
- **Fix size**: ~30 LOC + tests. Highest visible-value-per-line in the repo.

### 2. Submit endpoint silently strands submissions if Celery is down

- `apps/api/grader/routers/submissions.py:228–242` sets row to `PROCESSING`, then calls `process_submission.delay(...)`.
- If the broker raises, it catches `Exception`, sets `task_id=None`, returns 202.
- Comment promises *"the worker will pick it up via the periodic reconciliation task"* — **that reconciler doesn't exist** (no scheduled beat, no orphan scan).
- Submissions stuck in `PROCESSING` forever; owner-side polling spins indefinitely.
- **Fix**: revert status to `CAPTURING` on broker error, OR build the reconciler.

### 3. Catalog has two stores that aren't synced

- On-disk reference library at `~/manufacturer_refs/<manufacturer>/<variant_id>/` populated by Scryfall + PokemonTCG ingests (~25k MTG / ~16k Pokemon printings reachable).
- SQL `card_variants` table (`db/models.py:111`) — what the pgvector identification lookup queries at submission time — is **only populated by tests**.
- No daily job walks the JSONL and upserts into Postgres with `canonical_image_embedding` / `canonical_phash` set.
- **The manufacturer reference flywheel and the production identification path are running on different data stores.** Largest unlit fuse in the system.

### 4. Counterfeit thresholds are still synthetic

- `ml/pipelines/counterfeit/ensemble.py:54–78` — every threshold (rosette / color / embedding-anomaly authentic + counterfeit + min-confidence) is tuned against `synth_*` fixture generators.
- Embedding-anomaly thresholds (0.65/0.35/0.4) explicitly noted as placeholders mirroring rosette's shape.
- Benchmark harness has 50 synthetic samples; the benchmark *is* the synthetic distribution, so it can't catch the drift.
- Until there's a real-card validation slice (~300 labeled), production verdicts could be systematically biased.

### 5. The data flywheel isn't actually running

- `scripts/accumulate_psa_exemplars.py` is built + tested but only fires manually.
- `com.projectgrader.psa-ingest.plist` runs the ingest itself, nothing else.
- Each missed day = missed bump to embedding-anomaly's `n_references`, which is the gate on its confidence ramp.
- Pure ops work, no new code.

### 6. Capture flow doesn't capture shots three detectors need

- `apps/web/lib/submission.ts:28–36` declares `front_full_flash` and `tilt_30` as valid `ShotKind`s.
- But `apps/web/app/grade/[id]/page.tsx:60–97`'s wizard order is front + back + 4 corners only.
- Surface defect detector wants flash shots; holographic parallax detector wants tilt — **both are blocked at the capture step before they're blocked on data**.
- Adding two optional shots to the wizard unblocks two future detectors with no ML work.

### 7. `Grade.final` is structurally always `None` in Phase 1

- `apps/api/grader/services/grading.py:154–169` — `compute_psa_final` returns `None` if any of (centering, corners, edges, surface) is missing.
- Corners + surface are skeletons → **every Phase-1 cert renders without a final grade by design**.
- Correct philosophy ("never fabricate") but large product implication.
- **Decision needed**: ship a labeled "preliminary grade = min(centering, edges)" with caveat, OR hold the partial-grade UX line.

### 8. `sys.path.insert(0, parents[4]/"ml")` repeated in four files

- `services/{grading,identification,counterfeit}.py` and `workers/pipeline_runner.py` each do the same fragile `_ML_ROOT = Path(__file__).resolve().parents[4] / "ml"` block at module top.
- Depth-counting footgun; not a bug today but breaks the first time anyone moves a file.
- **Fix**: single helper module imported once, OR `uv add` the `ml` package in editable mode.

### 9. Production gaps the roadmap names but worth restating as risks

- **No CI** — manual `_run_*_tests.py` runners only; nothing gates merges to `main`.
- **Real Clerk SDK not wired** — `NEXT_PUBLIC_DEV_CLERK_ID` is the only auth path on web.
- **Terraform dev env** is empty; only docker-compose exists.
- **`reference_embeddings.npz` deployment** unresolved — local-disk default won't survive ECS Fargate.
- **No MLflow registry** in use yet; `model_versions` JSONB exists but no deployment plumbing.
- **No Datadog / Sentry / Statsig** — observability is planned, not implemented.

---

## Refined priority list

Ordered by impact-per-effort, blending audit findings with the existing roadmap.

| # | Move | Why | Size |
|---|---|---|---|
| 1 | Surface `identified_card` on cert + owner pages | Data already persisted; turns generic cert into "2023 Pikachu V from Crown Zenith" | ~30 LOC + tests |
| 2 | Catalog DB-side ingest job (JSONL → `card_variants` upserts) | Identification lookup queries an empty table in real environments | 2–3 days |
| 3 | Wire `embed_references` + `accumulate_psa_exemplars` into the daily launchd plist | Operationalizes the data flywheel; embedding-anomaly compounds daily | ~0.5 day, ops only |
| 4 | Fix broker-failure path in submit endpoint (revert to CAPTURING, or build reconciler) | Production-readiness; protects against Celery hiccups | small / medium |
| 5 | Add `front_full_flash` + `tilt_30` as optional wizard steps | Unblocks surface + holographic detectors at capture-time | UI only |
| 6 | Real Clerk SDK swap on web (1-function change in `lib/submission.ts:getDevAuthHeader`) | Unblocks real users + Stripe | ~2 hours |
| 7 | GitHub Actions CI (`uv sync && pytest` for `apps/api` and `ml`) | Stops the next regression at PR time | one-shot |
| 8 | Decide "preliminary grade" stance — ship labeled `min(centering, edges)` OR document partial-grade UX | Without this, every Phase-1 cert reads as incomplete | product call |
| 9 | When PSA corpus crosses 200, run `corners.train smoke_only=true` end-to-end | The "data plumbs to torch" milestone | calendar-time |
| 10 | Build small real-card calibration set + recalibrate ensemble thresholds | Stops synthetic-vs-reality threshold drift | needs ~300 labeled cards |
| 11 | Terraform dev env + pick `reference_embeddings.npz` deployment path | Decides itself once API is being containerized for real | medium |
| 12 | Variant linkage between PSA records and Scryfall/PokemonTCG catalog | Unlocks proper supervised triplet loss for DinoV2 | ~1 week, research-heavy |
| 13 | Stripe payments + usage tracking | Enables monetization ($10/mo unlimited tier) | 3–4 days |
| 14 | Complete remaining 4/7 counterfeit detectors (typography → substrate → holographic → siamese) | Ensemble recall + FPR improvements | ~2 weeks each, mostly data-blocked |
| 15 | Flash-shot data collection + labeling (substrate detector) | Phase 2 differentiator | 4–6 weeks, hardware-dependent |

---

## Recommended next session — three tractable hours

Three small wins that unblock different threads:

- **Hour 1** — Surface `identified_card`. Add `Submission.identified_variant: Mapped[CardVariant | None] = relationship(...)` on the model, `selectinload(Submission.identified_variant).selectinload(CardVariant.set)` in both routers, populate `IdentifiedCard(name=..., set=..., year=...)` from it. Two endpoint tests pinning the new field.
- **Hour 2** — Fix broker-failure handling in `routers/submissions.py:228–242`. Pick: revert status to `CAPTURING` on `Exception`, OR add a 60s reconciler beat that scans for `PROCESSING` rows older than N minutes with no audit-log progress.
- **Hour 3** — Add launchd plist entries for `embed_references` + `accumulate_psa_exemplars` after `psa_daily_ingest`. Pure ops; data flywheel starts compounding the same day.

Items #2 (catalog DB-side ingest) and #5 (flash + tilt shots) are bigger and warrant dedicated sessions each.

---

## What's shipped (production-ready)

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
| PSA daily ingest | ✅ Production | launchd, 30–50 records/day on free tier |
| PSA exemplar accumulator | ✅ Production | Per-variant embedding expansion |
| Web capture wizard | ✅ MVP | 6-shot flow; presign + S3 PUT + quality gate |
| Public cert page | ✅ MVP | ISR, per-detector breakdown |
| Celery pipeline orchestrator | ✅ Production | Soft-fail on missing stages |
| Postgres schema + migrations | ✅ Production | 8 tables; AST parity test |
| S3 / MinIO integration | ✅ Production | Presigned URLs |

## What's MVP/skeleton (works, rough edges)

| Component | State | Notes |
|---|---|---|
| Counterfeit ensemble | ⚠️ MVP (3/7) | Rosette + color + embedding wired |
| Web UI | ⚠️ MVP | No design polish; no Clerk integration |
| Corners trainer | ⚠️ Skeleton | Refuses to train < 200 samples |
| Surface trainer | ⚠️ Skeleton | Default mask loader returns background |
| Identification trainer | ⚠️ Skeleton | Lacks variant linkage for supervised triplet loss |
| Detection trainer | ⚠️ Skeleton | Manifest builder ready, not trained |
| Edges grading | ⚠️ MVP | Only synthetic-calibrated confidence |

## What's stubbed / missing

| Feature | Gap | Impact |
|---|---|---|
| Corners + surface grading models | Real training data | Pipeline returns NULL grades |
| Typography detector (#5) | OCR dependency | Out of 7-detector ensemble |
| Substrate / paper detector (#6) | Flash-shot labeled dataset | Out of 7-detector ensemble |
| Holographic parallax (#3) | Tilt shot in capture flow | Out of 7-detector ensemble |
| Stripe payments | Integration not wired | Blocks paid-tier feature |
| TCGplayer pricing comps | Integration not wired | No EV calculations |
| Clerk JWT swap (web) | Boilerplate | Dev mode only |
| Terraform prod infra | AWS account setup | Only dev docker-compose exists |
| Mobile app | Phase 2 | Out of MVP scope |
| Real-data threshold calibration | ~300 labeled cards | All thresholds synthetic-fixture defaults |
| Catalog DB-side ingest | No Scryfall/Pokemon → SQL job | `card_variants` empty in real envs |
| MLflow model registry | No deployment plumbing | Weights local-only |
| GDPR data export / RTBF | Not implemented | Needed for paid tier |

---

## Dependency graph for next milestones

```
Corpus ≥200 samples (May 3-7)
├─ First corners trainer run (1 day)
│  └─ Surface + edges models improve (2-3 weeks real training)

Counterfeit thresholds recalibrated (2 days, ~300 real cards)
└─ Ensemble accuracy locked in

Exemplar accumulator in launchd (0.5 day)
└─ Embedding-anomaly confidence ramps automatically

Catalog DB-side ingest (2-3 days)
└─ pgvector identification accuracy unlocked

Variant linkage (1 week, research-dependent)
└─ DinoV2 supervised triplet loss → identification accuracy

Clerk SDK swap (2 hours) + Stripe (3-4 days)
└─ Real signups + monetization

Terraform prod (1-2 weeks, AWS setup required)
└─ Datadog + SLOs

Flash-shot dataset + substrate detector (4-6 weeks)
└─ 6/7 ensemble complete → ship MVP to beta
```

---

## Open product/strategic decisions

- **Preliminary final grade** — ship `min(centering, edges)` with "subject to revision" badge, or hold the partial-grade UX line until corners + surface land?
- **`reference_embeddings.npz` prod path** — bake into image / S3 + boot-time download / S3 + LRU cache / FAISS sidecar?
- **JWT email-claim policy** — overwrite `User.email` on re-login or freeze on first-seen? (GDPR/consent angle.)
- **PSA paid API tier** — free is 100/day; email `webcert@collectors.com` once daily ceiling becomes a bottleneck.
- **AWS account / region / naming / IAM strategy** — required before Terraform can be filled in.
