# TODO

Action-oriented checklist of what's left. Full operational snapshot in [docs/roadmap.md](docs/roadmap.md). Strategic plan (4-phase business roadmap, MVP scope, accuracy targets) lives at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md`.

Each item has enough context to pick it up cold without reloading session state.

---

## Status snapshot (2026-05-04)

Counterfeit ensemble is **3/7 wired** end-to-end (FFT rosette + color profile + embedding-anomaly). Manufacturer reference flywheel is live for MTG (Scryfall paginated + bulk) and Pokemon (PokemonTCG.io). All four ML-head trainer skeletons (corners, surface, identification/DinoV2, detection/YOLO) committed. Public cert page (`/cert/[id]`) and owner-side capture wizard (`/grade/[id]`) live — complete in→out surface in apps/web. Daily data flywheel runs as a 3-step launchd chain (ingest → re-embed references → accumulate exemplars). Counterfeit-threshold recalibration tool ships; awaiting a labeled counterfeit slice to flip from authentic-only to two-sided mode. The system is ~30% to production MVP; ML heads stay data-blocked on PSA corpus growth + variant linkage.

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
| Counterfeit ensemble | ⚠️ MVP (3/7) | Rosette + color + embedding wired |
| Web UI | ⚠️ MVP | No design polish; no Clerk integration |
| Corners trainer | ⚠️ Skeleton | Refuses to train < 200 samples |
| Surface trainer | ⚠️ Skeleton | Default mask loader returns background |
| Identification trainer | ⚠️ Skeleton | Lacks variant linkage for supervised triplet loss |
| Detection trainer | ⚠️ Skeleton | Manifest builder ready; not trained |
| Edges grading | ⚠️ MVP | Only synthetic-calibrated confidence |

### What's stubbed / missing

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
| Real-data threshold calibration | ~300 labeled cards | Tool ready; corpus needed |
| Catalog DB-side ingest | No JSONL → SQL job | `card_variants` empty in real envs |
| MLflow model registry | No deployment plumbing | Weights local-only |
| GDPR data export / RTBF | Not implemented | Needed for paid tier |

---

## Now — do next (1 session each)

- [ ] **First real corners trainer run** (when corpus crosses 200 samples)
  - At current ingest rate (~30-50/day), expected ~3-5 days from 2026-04-29
  - `cd ml && uv run python -m training.trainers.corners train.smoke_only=true train.epochs=2`
  - Goal: prove the data pipeline plumbs through to a torch step. NOT "model converges" (corpus too small).
  - Expected: `dataset.min_samples=200` gate passes; one mini-batch flows through the EfficientNet-V2-S backbone; checkpoint written to `outputs/corners/best.pt`
  - First useful val signal needs ~1k samples (~3-4 weeks at free-tier rate, OR email `webcert@collectors.com` for paid tier).

- [ ] **Recalibrate counterfeit thresholds** — three detectors now, all on synthetic-fixture defaults
  - All thresholds in `ml/pipelines/counterfeit/ensemble.py` are calibrated against synthetic generators (the `synth_*` fixtures).
  - Embedding-anomaly added 2026-04-29 with placeholder thresholds (0.65 / 0.35 / 0.4) mirroring the rosette + color shape — needs real-data anchors before they're trustworthy.
  - **Tool now ships** (`evaluation.counterfeit_recalibration`, 97e3f71): given a labeled corpus, recommends per-detector AUTHENTIC + COUNTERFEIT thresholds via Youden's-J + an FPR-budget cap, emits a drop-in patch block for `ensemble.py`. Runs in `authentic_only` mode against `~/psa_data/scraped.jsonl` alone (the steady state once the daily accumulator has been running) — picks the AUTHENTIC threshold, leaves COUNTERFEIT alone until fakes appear. Switches to `two_sided` once a `--csv` of counterfeits is supplied.
  - Remaining work is operational: source the counterfeit slice (manual curation or a small purchase set), run the tool, paste the patch, re-run the benchmark, lock + version in `model_versions` metadata. Benchmark harness (`python -m evaluation.counterfeit_benchmark`) stays the regression gate.

- [ ] **Fix broker-failure path on submit** — submissions silently strand if Celery is down
  - `apps/api/grader/routers/submissions.py:237-251` flips the row to `PROCESSING` then calls `process_submission.delay(...)`. On broker exception it swallows, sets `task_id=None`, returns 202.
  - The promised reconciler ("the worker will pick it up via the periodic reconciliation task") doesn't exist — no scheduled beat, no orphan scan. Submissions sit in `PROCESSING` forever; the wizard's status poll spins indefinitely.
  - Pick one: (a) revert status to `CAPTURING` on `Exception` so the user can retry, (b) build the orphan-scan reconciler beat (60s, scan PROCESSING > N min with no audit-log progress).
  - Option (a) is small + immediate; (b) is the proper fix.

- [ ] **Catalog DB-side ingest job** — JSONL → `card_variants` upserts
  - On-disk reference library at `~/manufacturer_refs/<manufacturer>/<variant_id>/` is populated by Scryfall + PokemonTCG ingests (~25k MTG / ~16k Pokemon printings reachable). The SQL `card_variants` table (`db/models.py:111`) — what the pgvector identification path queries at submission time — is **only populated by tests**.
  - In real environments the identification lookup queries an empty table. The manufacturer reference flywheel and the production identification path run on different data stores.
  - Fix: a daily upsert job that walks the references JSONL + `reference_embeddings.npz` and writes `CardVariant` rows with `canonical_image_embedding` + `canonical_phash` set. Belongs alongside the existing `daily_cycle.sh` chain.

- [ ] **Add `front_full_flash` + `tilt_30` to the capture wizard**
  - `apps/web/lib/submission.ts:28-36` already declares both as valid `ShotKind`s, but `apps/web/app/grade/[id]/page.tsx:60-97` only walks front + back + 4 corners.
  - Surface defect detector wants flash shots; holographic-parallax detector wants tilt — both are blocked at the **capture step** before any ML work.
  - Adding two optional shots to the wizard unblocks two future detectors with no ML work. Pure UI.

---

## Soon — needs your input (sit-with-me sessions)

- [ ] **Terraform dev environment**
  - File: `infra/terraform/envs/dev/main.tf`
  - Needs your AWS account ID, region, naming convention, IAM strategy
  - Resources: S3 bucket, Aurora Postgres + pgvector, ElastiCache Redis, ECR for Docker images, IAM roles
  - Don't deploy real infrastructure on autopilot

- [ ] **Decide policy on JWT email-claim updates**
  - Current behavior: `User.email` is NOT overwritten on re-login if Clerk's JWT carries a different email
  - Has GDPR/consent angle — worth a deliberate decision
  - One-line fix in `_get_or_create_user` in [services/auth.py](apps/api/grader/services/auth.py) once decided

- [ ] **Reference-embeddings deployment path** — where does the npz live in prod?
  - Local-disk default works fine for dev (`~/manufacturer_refs/reference_embeddings.npz`).
  - In prod the API workers + Celery workers all need read access. Options: (a) bake into the Docker image (~100 MB at 25k variants × 768 floats), (b) S3 + boot-time download to a shared volume, (c) S3 + per-call lazy download with LRU cache, (d) FAISS index + a sidecar service.
  - Pick when API is being containerized for real (currently only docker-compose dev).

- [ ] **"Preliminary final grade" stance**
  - `apps/api/grader/services/grading.py:154-169` — `compute_psa_final` returns `None` if any of (centering, corners, edges, surface) is missing. Corners + surface are skeletons → **every Phase-1 cert renders without a final grade by design**.
  - Correct philosophy ("never fabricate a grade we can't justify") but a real product implication.
  - Decision: ship a labeled `min(centering, edges)` "preliminary" with caveat, OR hold the partial-grade UX line until corners + surface land.

- [ ] **PSA paid API tier**
  - Free is 100 calls/day; once that becomes the bottleneck on corpus growth, email `webcert@collectors.com` to upgrade. Mostly a calendar-time decision driven by training-corpus size hitting ceilings.

---

## Blocked on data accumulation (PSA daily ingest is running)

- [ ] **Corners ML model — real training** — Skeleton committed (8ea54b4). Refuses to train below 200 samples; expected to cross threshold ~3-5 days from 2026-04-29. First useful val signal needs ~1k samples.
- [ ] **Surface ML model — real training** — Skeleton committed (31b7adb). Hardest model in the system. Needs flash-shot data (currently unavailable) + per-defect-class labels (currently unlabeled). Default `mask_loader` returns all-background placeholder masks for the skeleton; swap in a real loader once labels exist.
- [ ] **Identification (DinoV2) trainer — real training** — Skeleton committed (41e68f2 / cd7a7f0). Triplet loss with augmentation-based positives (anchor + positive = same image, two different transforms). Real training needs string-matched-positive sampling once we have variant_id linkage between PSA records and the catalog. ViT-B/14 patch constraint: `image_size` must be a multiple of 14.
- [ ] **Detection (YOLOv11-seg) — real training** — Trainer + config exist (45fd695); manifest builder shipped (09fe2a4) producing labeled scenes from `card_in_scene` fixtures. Real training would mix synthetic + real-scene labels; the manifest builder is ready for that. CLI: `python -m scripts.build_detection_manifest --out-dir ... --n-train ... --n-val ...`.

---

## Blocked on other work

- [ ] **Typography counterfeit detector** (ensemble #5) — needs PaddleOCR or similar (new dep). OCR + glyph-shape comparison to canonical font templates.
- [ ] **Substrate paper detector** (ensemble #6) — manufacturer reference library now exists, but per-manufacturer paper-fluorescence training still needs a labeled flash-shot dataset (which we don't have). Distinct from embedding-anomaly: substrate looks at the *paper itself* under flash, not at art content.
- [ ] **Holographic parallax detector** (ensemble #3) — needs the tilt shot in the capture flow. Two-angle optical flow on holo region.
- [ ] **Siamese reference detector** (ensemble #1) — would be trained off the same data feeding the DinoV2 trainer skeleton; effectively the same fine-tune with a different head. Consider folding into the identification trainer rather than building a separate one.

---

## Later — Phase 1 MVP polish

- [ ] **Stripe payments** — free 5/mo, $10/mo unlimited. Webhook handler for subscription lifecycle.
- [ ] **TCGplayer pricing comps** — pull current market price on identified-variant. Show predicted-grade-adjusted EV (`Σ P(grade) × median_price(grade)`).
- [ ] **Real production deploy** — Aurora Postgres, ElastiCache, ECS Fargate API, EKS GPU workers, Triton inference, MLflow registry. Build out via Terraform after dev env exists.

---

## Phase 2+ (out of MVP scope per the plan)

- [ ] **Mobile app** — React Native + Expo, on-device ML for capture guidance
- [ ] **eBay sold-listings comps** — needs 130point licensing decision; eBay Marketplace Insights API integration
- [ ] **Buy/sell marketplace** — built on top of identified+graded inventory
- [ ] **Imaging kiosks at partner shops** — Phase 3 hardware. Custom rig, 4-6 cameras, controlled lighting, telecentric lens
- [ ] **Full slabbing operation** — Phase 4. Facility, staff, slab line, E&O insurance

---

## Operational hygiene (continuous)

- [x] **Counterfeit benchmark — v1 corpus + harness** (committed 6823e48). Run: `python -m evaluation.counterfeit_benchmark`. Current: 50-sample synthetic corpus, rosette 98% accuracy / color 100% / ensemble 98%. **Still TODO**: extend with embedding-anomaly cases (needs labeled identified-variant samples), swap in real images as the corpus grows, then add the hard CI gate (recall ≥ 90% / FPR ≤ 2%).
- [x] **Manufacturer reference re-embed cycle** — runs as step 2 of `ml/scripts/daily_cycle.sh` (faea7d6). Idempotent on `(manufacturer, variant_id)`; a no-op on days with no new ingest.
- [ ] **Consolidate `_ML_ROOT = parents[4]/"ml"` sys.path bootstrap** — `services/{grading,identification,counterfeit}.py` and `workers/pipeline_runner.py` each repeat the same depth-counting block. Footgun on the first file move. Fix: a single helper module imported once, OR `uv add` the `ml` package in editable mode.
- [ ] **GitHub Actions CI** — `uv sync && pytest` for `apps/api` and `ml`. Type-check + lint gates. Coverage tracking. Currently merges to `main` are ungated.
- [ ] Public accuracy benchmark — predictions vs subsequent PSA grades, published monthly (trust-strategy item from plan).
- [ ] Capture-guidance acceptance test — real users on mid-range phones; rejection rate and retakes/submission. Target <2 retakes median.
- [ ] Latency SLO monitoring — p95 ≤ 20s, p99 ≤ 35s (Datadog dashboards + PagerDuty alerts).

---

## Dependency graph for upcoming milestones

```
Corpus ≥ 200 samples (~3-5 days from 2026-04-29)
└─ First corners trainer end-to-end run (1 day)
   └─ Surface + edges models improve (2-3 weeks real training)

Counterfeit slice sourced (~300 labeled cards)
└─ Recalibration tool flips authentic-only → two-sided
   └─ Lock thresholds + version in model_versions

Catalog DB-side ingest (2-3 days)
└─ pgvector identification accuracy unlocked (no longer querying empty table)

Variant linkage (PSA records ↔ Scryfall/PokemonTCG, ~1 week, research-heavy)
└─ DinoV2 supervised triplet loss → identification accuracy

Clerk SDK swap (2 hours) + Stripe (3-4 days)
└─ Real signups + monetization

Terraform prod (1-2 weeks, AWS setup required)
└─ Datadog + SLOs

Flash-shot dataset + substrate detector (4-6 weeks, hardware-dependent)
└─ 6/7 ensemble complete → ship MVP to beta
```

---

## Recently shipped

### 2026-05-04
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
