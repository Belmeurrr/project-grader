# Project Grader — Roadmap & Status

**Updated:** 2026-04-29

This doc tracks near-term work and current state. The architectural plan (4-phase roadmap, MVP scope, accuracy targets, business strategy) lives separately at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md` — that's the strategic baseline and shouldn't drift; this is the operational snapshot.

---

## Where we are

- **26 items shipped** across 13 work sessions (see Done table below). Counterfeit ensemble is now **3/7 wired** end-to-end (rosette + color + embedding-anomaly). Manufacturer reference flywheel is live for MTG (Scryfall) and Pokemon (PokemonTCG.io). All four ML-head trainer skeletons (corners, surface, identification/DinoV2, detection/YOLO) are committed; real training stays data-blocked on PSA corpus growth + variant linkage. Public cert page (`/cert/[id]`) is the first user-clickable surface. Per-variant PSA exemplar accumulator now bumps embedding-anomaly's `n_references` per cycle.
- ~28,000 LOC across Python + TypeScript + YAML.
- **35+ test files**; benchmark harness at `python -m evaluation.counterfeit_benchmark` runs in ~3s.
- **Feature commits on `main`** (foundation → today):
  - Foundation: `45fd695`, `6f54de7`, `c174171`, `0c4a414`, `ab43c6a` — quality gating, detection+dewarp, identification, FFT counterfeit + wiring, PSA Public API + GitHub seed, daily launchd ingest.
  - Counterfeit ensemble: `ed29e35` + `94b83ca` (color built+wired), `65bbcdb` (embedding-anomaly pure fn), `c5af923` (embedding-anomaly wired), `6823e48` (benchmark + shared ensemble module).
  - ML trainer skeletons: `8ea54b4` (corners), `31b7adb` (surface), `41e68f2`+`cd7a7f0` (identification/DinoV2), `09fe2a4`+`e477270` (detection manifest builder).
  - Manufacturer reference flywheel: `faab948` (Scryfall), `d24691a` (PokemonTCG.io), `6caa8a1`+`6d9032a` (reference embedding pass).
  - Cert page + cleanup: `00b056a` (canonical-loader dedup + lazy httpx), `460841a`+`33e9c4e` (public cert page + 3-detector test fixture), `7d4cdfe` (late-session TODO sweep).
- **Calendar-time tracks running:**
  - PSA daily ingest (launchd `com.projectgrader.psa-ingest`, 09:00 local) — ~30–50 successful triples/day on free tier.
  - Corpus needs to cross corners-trainer's `min_samples=200` threshold (~3-5 days at current rate from 2026-04-29).
  - Manufacturer reference re-embed cycle: run `scripts.embed_references` after each new Scryfall / PokemonTCG ingest so newly-ingested variants become eligible for embedding-anomaly. Idempotent by `(manufacturer, variant_id)`.

## Pipeline state

```
[client] ─upload→ [S3/MinIO] ─register→ [QualityGate]               ✓ done
                                              │ pass
                            (user) ─POST /submit→
                                              │
                                              ▼
                                    [Celery: process_submission]    ✓ done
                                              │
                                              ▼
                                    [Detection + Dewarp]            ✓ done
                                              │
                                              ▼
                                    [Identification (soft-fail)]    ✓ done
                                              │
                                              ▼
                                    [Centering grading]             ✓ done — geometric, ~99% accurate
                                              │
                                              ▼
                                    [Edges grading]                 ✓ done — geometric v1
                                              │
                                              ▼
                                    [Corners grading]               ⊘ blocked on PSA data
                                              │
                                              ▼
                                    [Surface grading]               ⊘ blocked on PSA data + flash shot
                                              │
                                              ▼
                                    [Counterfeit ensemble]          ✓ rosette + color + embedding wired (Stage 3.5)
                                       └─ FFT rosette               ✓ built + wired
                                       └─ Color profile             ✓ built + wired (CIELAB chroma)
                                       └─ Embedding anomaly         ✓ built + wired (centroid cosine; abstains gracefully on no-refs)
                                       └─ Typography                ⊘ TODO (needs OCR dep)
                                       └─ Substrate (paper)         ⊘ TODO (needs labeled flash-shot dataset)
                                       └─ Holographic parallax      ⊘ needs tilt shot in capture flow
                                       └─ Siamese reference         ⊘ folds into DinoV2 trainer (different head, same fine-tune)
                                              │
                                              ▼
                                    [Submission COMPLETED]
                                       Grade row:           centering ✓ edges ✓ corners ⊘ surface ⊘ final ⊘
                                       AuthenticityResult:  rosette ✓  color ✓  embedding ✓  others ⊘
```

`final` stays NULL until corners + surface land. By design — we never fabricate a number.

---

## Done (26 items)

| # | Item | Commit | Notes |
|---|---|---|---|
| 1 | Alembic migration + parity test | 45fd695 | 8 tables, 5 enums, HNSW index, `pgvector`+`uuid-ossp` extensions. Migration parity-tested via AST. |
| 2 | S3/MinIO presigned upload + quality gating | 45fd695 | Per-shot-kind thresholds (flash tolerates more glare; corner zooms relaxed fill). Reject-rather-than-guess principle. |
| 3 | Card detection + dewarp (Stages 1+2) | 45fd695 | YOLOv11-seg lazy loader + Canny+contour heuristic fallback. 4-point homography to canonical 750×1050. Quad-irregularity gate. |
| 4 | Card identification (Stage 3) | 45fd695 | pHash + dHash, `ImageEmbedder` Protocol with `SimpleEmbedder` fallback and lazy `DinoV2Embedder` production path. `pgvector` catalog repo with `bit_count` Hamming + `<=>` cosine NN. Two's-complement bigint round-trip. |
| 5 | Celery worker pipeline | 45fd695 | Orchestrates detection → identification → grading. Per-stage audit log. Partial-grade persistence. |
| 6 | Centering grading | 45fd695 | Geometric measurement, ~99% accurate by construction. Front/back support; back-tolerance follows PSA. |
| 7 | Edges grading v1 | 45fd695 | Geometric perimeter-strip MAD anomaly detection. Per-side localization. Worse-face dominates. |
| 8 | **PSA pop-report scraper** | 6f54de7 | Resumable, rate-limited, 429-aware, S3+local stores. 18 inline tests passing. **DEPRECATED 2026-04-28**: PSA cert pages are gated by Cloudflare's "Just a moment..." JS challenge (verified `cf-mitigated: challenge` in headers); a polite httpx client cannot pass it. Module retained for the storage layer (`ScrapedRecord` + `*Store`) which is reused by the new PSA Public API client. |
| 9 | **FFT print-rosette counterfeit detector** | 6f54de7 | First detector in the 7-detector authenticity ensemble. Halftone Fourier signature → manufacturer-level counterfeit fingerprint. **Built but not yet wired into the pipeline runner.** 21 inline tests passing. |
| 10 | **Real Clerk JWT auth** | 6f54de7 | `dev_auth_enabled` flag with env-derived default. JWKSCache with TTL. RS256-only verification. Alg-confusion attack guarded. 512 lines of tests with mocked JWKS + generated RSA keys. |
| 11 | **FFT wired into pipeline + AuthenticityResult persistence** | c174171 | Stage 3.5 runs `analyze_rosette` after detection+dewarp; `persist_authenticity_result` upserts on `submission_id` with verdict derived from rosette score + confidence (AUTHENTIC ≥ 0.65, COUNTERFEIT ≤ 0.35, UNVERIFIED below min_confidence). Soft-fails so grading proceeds. E2E tests assert both Grade AND AuthenticityResult rows + audit log entries + upsert stability. |
| 12 | **PSA Public API client + GitHub seed corpus ingestor** | 0c4a414 | Bearer-auth httpx client over the existing ScrapedRecordStore. Parser verified end-to-end against 5 real cert classes (Pokemon Mint 9, baseball half-grade FR 1.5, baseball NM-MT 8, PSA/DNA Bonds autograph, auto/mem 1/1 Banchero). 100/day free-tier daily-budget tracker with UTC rollover. GitHub seed ingestor reads `samsilverman/PSA-Baseball-Grades` (11.5k labeled images) → JSONL manifest. License tripwire requires `PSA_SEED_LICENSE_ACK=1` for redistribution paths. |
| 13 | **Calendar-time daily PSA ingest under launchd** | ab43c6a | `psa_daily_ingest.py` driver with state-file resume cursor (`~/psa_data/state.json`), append-only stats log (`ingest_log.jsonl`), launchd plist firing daily at 09:00 local. Soft-fail at the per-cert level; hard-stop only on auth errors (state cursor NOT advanced so token rotation doesn't lose its place). Real first-fire produced 27 records + 28 image dirs in ~30s before budget exhaustion. |
| 14 | **Corners ML trainer skeleton** | 8ea54b4 | EfficientNet-V2-S backbone + Linear regression head; PSACornerDataset reads scraped.jsonl, emits 4 corner crops per cert; 80/20 train/val split BY cert_id (no leakage). Hydra config + MLflow tracking matching detection.py's pattern. Refuses to train below `dataset.min_samples=200` (avoids meaningless loss curves on tiny corpora). All 4 corners labeled with overall card grade until per-corner labels exist; swap-in slot in dataset for future per-corner subgrades. |
| 15 | **Color-profile counterfeit detector — built + wired** | ed29e35 + 94b83ca | Ensemble #4. CIELAB p95 chroma after white-border-sampled chromatic-adaptation calibration. Logistic squash midpoint at 40 (between inkjet ceiling and offset baseline). Stage 3.5 now runs both detectors with conservative ensemble combiner: any LIKELY_COUNTERFEIT wins, AUTHENTIC requires consensus among confident detectors, UNVERIFIED detectors don't block. AuthenticityResult row carries both detectors' raw outputs + per-detector verdicts + combined verdict. 18 inline + 11 service tests passing. |
| 16 | **Embedding-anomaly counterfeit detector — pure fn** | 65bbcdb | Ensemble #7. Cosine-distance-from-centroid against per-variant authentic reference embeddings; logistic squash with confidence ramp by `n_refs` (0 → 0.0, 1 → 0.4, 5 → 0.85, 10+ → 1.0). 22 tests on synthetic embeddings. **Service wrapper deferred** — the per-variant authentic-exemplars store doesn't exist yet (catalog has 1 ref per variant, not several). Wiring blocked on either manufacturer-reference-scraper output or a dedicated authentic-exemplars table seeded from PSA ingest aggregated by SpecID. |
| 17 | **Counterfeit-detector benchmark + shared ensemble module** | 6823e48 | Operational hygiene. `ml/pipelines/counterfeit/ensemble.py` is the canonical home for thresholds + verdict combiner; apps/api re-exports them. `ml/evaluation/counterfeit_benchmark/` runs the labeled-corpus benchmark (`python -m evaluation.counterfeit_benchmark`); 50-sample synthetic corpus (25 authentic + 25 counterfeit). Current scores: rosette 98% accuracy, color 100% accuracy, ensemble 98% accuracy. JSON + Markdown + console reports. Hard CI gate intentionally NOT added until corpus is ≥50% real images. |
| 18 | **Manufacturer reference image scrape — Scryfall (MTG)** | faab948 | `data/ingestion/scryfall.py` — paginated `/cards/search` client + image download, polite rate limiting, ToS-compliant attribution. Reference store at `~/manufacturer_refs/<manufacturer>/<variant_id>/` with a JSONL index. CLI: `python -m scripts.manufacturer_refs_ingest --source mtg --query "..."`. ~25k MTG printings reachable; per-set ingest is the typical workflow until bulk-data path lands. |
| 19 | **Manufacturer reference image scrape — PokemonTCG.io** | d24691a | `data/ingestion/pokemontcg.py` — same shape as Scryfall but for Pokemon cards (~16k printings). Variant key is `"<set>-<num>"`. Same JSONL/image storage, same `manufacturer_refs_ingest` CLI with `--source pokemon`. Both scrapers feed a unified reference store consumed by the embedding pass. |
| 20 | **Reference embedding pass** | 6caa8a1 + 6d9032a | `data/ingestion/reference_embeddings.py` — walks the manufacturer reference JSONL, embeds each canonical with the identification embedder, writes a single `reference_embeddings.npz` keyed by `"<manufacturer>/<variant_id>"`. Idempotent on key — re-running adds new variants without rewriting existing arrays. Helpers: `load_embeddings`, `lookup_references` (latter consumed by `analyze_embedding_anomaly`). CLI: `python -m scripts.embed_references --data-dir ...`. |
| 21 | **Embedding-anomaly wired into pipeline_runner** | c5af923 | `analyze_embedding_anomaly` service wrapper at `apps/api/grader/services/counterfeit.py`. Stage 3.5 now runs all three detectors; ensemble combiner consumes the new `EMBEDDING_*_THRESHOLD` constants and `verdict_from_embedding_anomaly`. Graceful UNVERIFIED on no-references / unidentified / no-embedding paths so the ensemble degrades cleanly. `persist_authenticity_result` carries embedding's score, distance, n_references, and abstain_reason in `detector_scores["embedding_anomaly"]`. |
| 22 | **Surface ML trainer skeleton** | 31b7adb | EfficientNet-V2-S encoder + SegFormer-style decoder, 7-class CrossEntropyLoss (background, scratch, print_line, indentation, stain, paper_loss, foil_scratch). PSASurfaceDataset with a pluggable `mask_loader` (default returns all-background for the skeleton). Hydra+MLflow shape matching corners.py + detection.py. Real training data-blocked on flash-shot data + per-pixel labels. |
| 23 | **DinoV2 identification trainer skeleton** | 41e68f2 + cd7a7f0 | `training/trainers/identification.py` — ViT-B/14 backbone with triplet loss. Augmentation-based positives (anchor + positive = same image, two transforms) until variant_id linkage between PSA records and the catalog exists. ViT-B/14 patch constraint encoded: `image_size` must be multiple of 14. Replaces the public `facebook/dinov2-base` baseline once trained. |
| 24 | **YOLO detection-dataset manifest builder** | 09fe2a4 + e477270 | `data/catalogs/build_detection_manifest.py` — synthesizes labeled scenes from `card_in_scene` fixtures with analytically-derived bboxes. Emits a manifest at `ml/data/catalogs/detection_dataset.yaml` consumable by the YOLOv11-seg trainer (45fd695). CLI: `python -m scripts.build_detection_manifest --out-dir ... --n-train ... --n-val ...`. |
| 25 | **Public cert page (`/cert/[id]`)** | 460841a + 33e9c4e | First user-clickable surface tying Grade + AuthenticityResult together. Public no-auth `GET /cert/{submission_id}` endpoint at `apps/api/grader/routers/cert.py` with sanitized payload (no user_id / S3 keys / audit log), opaque 404 for in-progress submissions, `Cache-Control: public, max-age=300, swr=3600`. Server component at `apps/web/app/cert/[id]/page.tsx` with ISR (revalidate=300), color-coded verdict badge (lime/amber/red/zinc), per-detector breakdown (3 detectors) with abstain-reason rendering. 7 endpoint tests pinning the public contract; test fixture models the realistic 3-detector ensemble including embedding-anomaly's abstaining UNVERIFIED state. |
| 26 | **Per-variant PSA exemplar accumulator** | 96749f6 | `data/ingestion/psa_exemplars.py` + `scripts/accumulate_psa_exemplars.py`. Walks PSA `scraped.jsonl`, identifies each record against the catalog built from on-disk references + their embeddings, appends matched submissions' embeddings to `reference_embeddings.npz` under the same `(manufacturer, variant_id)` key. Idempotent by `cert_id` via sidecar `psa_exemplars_log.jsonl` so repeated runs don't double-add. Bumps `n_references` per variant which directly raises embedding-anomaly's confidence ramp over time. CLI: `python -m scripts.accumulate_psa_exemplars [--psa-data-dir ...] [--refs-data-dir ...]`. 12 tests; full ml suite 342 passing. |
| 27 | **Web capture flow — owner-side wizard** | 07f5394 | `apps/web/lib/submission.ts` + `apps/web/app/grade/page.tsx` (kickoff) + `apps/web/app/grade/[id]/page.tsx` (status-aware wizard). Wizard order: front → back → TL → TR → BR → BL (clockwise). Per-shot flow: presign → PUT to S3 → register (server runs quality gate) → quality_passed advances or surfaces reasons for retake. Camera via `getUserMedia({ facingMode: "environment" })` with file-input fallback. PROCESSING polls every 2s; COMPLETED redirects to `/cert/[id]`. Auth: dev-mode `Authorization: Dev <clerk_id>` (real Clerk SDK swap is one function). |
| 28 | **`/cert/{id}` + `/submissions/{id}` surface `identified_card`** | 832275f | Public cert payload now carries the matched `(manufacturer, variant_id, name, set_code, collector_number)` from the identification pass alongside verdict + grade. `GET /submissions/{id}` (owner-side, authed) added to support the wizard's status polling. Public payload stays sanitized (no user_id / S3 keys). |
| 29 | **Daily data-flywheel wrapper + Windows ops docs** | faea7d6 + b4d7a06 | `ml/scripts/daily_cycle.sh` — 3-step bash chain (`psa_daily_ingest` → `embed_references` → `accumulate_psa_exemplars`) replacing the previous inline launchd command. Steps run independently; per-step rc lines hit `launchd.stdout.log`. Configurable via `PROJECT_ROOT` / `PYTHON` / `PSA_INGEST_DATA_DIR` / `MANUFACTURER_REFS_DATA_DIR`. Plist hands off; same `Label` so installations keep working after `launchctl unload && load`. Companion `ml/scripts/README.md` Windows section documents Task Scheduler operation via a `daily_cycle.cmd` wrapper for the dev box. |
| 30 | **Counterfeit-detector threshold recalibration tool** | 97e3f71 | `ml/evaluation/counterfeit_recalibration/` — sibling to `counterfeit_benchmark`. Ingests `--csv image_path,ground_truth` and/or `--psa-authentics scraped.jsonl`; runs samples through the same ensemble runner production sees; recommends per-detector AUTHENTIC + COUNTERFEIT thresholds. Two modes auto-picked: `two_sided` (Youden's-J on midpoint candidates + FPR-budget cap on the LIKELY_COUNTERFEIT side) and `authentic_only` (target-recall percentile on the AUTHENTIC side, COUNTERFEIT left alone with a note). Outputs console table + per-detector notes + a drop-in patch block for `ensemble.py` (TWO_SIDED detectors only — partial recommendations stay out of the patch to avoid half-applying). `--json` / `--markdown` for CI / PR comments. Tool deliberately does NOT modify `ensemble.py`; threshold updates remain a deliberate human commit. 21 tests (sweep math, CSV + PSA-jsonl loaders, end-to-end against the synthetic benchmark corpus). |

---

## Near-term roadmap (refreshed)

In dependency / value order. Items marked **calendar-time** can run in the background while other work proceeds.

| # | Task | Type | Blocked on | Notes |
|---|---|---|---|---|
| 1 | **First real corners trainer end-to-end run** | Run | corpus ≥ 200 (~3-5 days) | Skeleton committed; once corpus crosses `dataset.min_samples=200`, run `python -m training.trainers.corners train.smoke_only=true` to prove the torch step end-to-end. Loss is meaningless on tiny corpora; goal is "wiring works", not "model converges". |
| 2 | ~~**Run accumulator against the live PSA corpus**~~ | Run / ops | done 2026-05-04 (faea7d6) | Daily cycle is now a 3-step bash chain via `ml/scripts/daily_cycle.sh`: `psa_daily_ingest` → `embed_references` → `accumulate_psa_exemplars`. Steps run independently; per-step `rc=N` lines hit `~/psa_data/launchd.stdout.log` for triage. Plist hands off to the wrapper. Redeploy on the macOS prod box (`launchctl unload && cp && launchctl load`). |
| 3 | **Recalibrate counterfeit thresholds against real cards** | ML / ops | counterfeit slice (purchase/curate) | Tool now ships (`evaluation.counterfeit_recalibration`, 97e3f71) — ingests labeled corpora, recommends thresholds via Youden's-J + FPR-budget cap, emits drop-in patch for `ensemble.py`. Authentic-only mode usable today against `~/psa_data/scraped.jsonl`; two-sided mode unlocks once a `--csv` of counterfeits is supplied. Remaining work is operational: source the counterfeit slice, run the tool, paste the patch, lock + version in `model_versions`. Benchmark harness stays the regression gate. |
| 4 | **Bulk-data Scryfall ingest path** | ML | — | Per-set `/cards/search` works (~1750-result cap); bulk JSON endpoint is the corpus-scale path. New module `data/ingestion/scryfall_bulk.py` mirroring `ingest_query`. Image download stays per-card; bulk-data only saves on metadata round-trips. |
| 5 | **Web capture flow** | Frontend | User UX input | Functional-only `/grade` + `/grade/[id]` pages. Camera permissions, error states, retake handling, multi-shot wizard order — pair-friendly UX choices. Pairs naturally with the now-shipped cert page since both touch apps/web. |
| 6 | **Terraform dev env** | DevOps | User AWS choices | Account ID, region, naming, IAM strategy — needs your input. |
| 7 | **Reference-embeddings deployment path** | DevOps | containerization decisions | Where does `reference_embeddings.npz` live in prod? Local-disk default works for dev. Options: bake into Docker image, S3 + boot-time download, S3 + LRU cache, FAISS sidecar. Pick when API is being containerized for real. |
| 8 | **Typography counterfeit detector** | ML | OCR availability | Detector #5. OCR + glyph-shape comparison. Needs PaddleOCR or similar — a real new dep. |
| 9 | **Substrate paper detector** | ML | labeled flash-shot dataset | Detector #6. Manufacturer reference library now exists, but training still needs labeled flash-shot data (which we don't have). Distinct from embedding-anomaly: substrate looks at the *paper itself* under flash, not at art content. |
| 10 | **Holographic parallax detector** | ML | Tilt shot capture flow | Detector #3. Two-angle optical flow on holo region. Needs the tilt shot in the capture flow. |
| 11 | **Siamese reference detector** | ML | folds into DinoV2 trainer | Detector #1. Same fine-tune as the DinoV2 identification trainer with a different head; consider folding into that trainer rather than building a separate one. |
| 12 | **Mobile app (RN/Expo)** | Frontend | Web capture flow | Phase 1 MVP scope explicitly excludes mobile. v0.2. |
| 13 | **Stripe payments** | Integration | Auth + capture flow | Free tier 5/mo, $10/mo unlimited per the plan. |
| 14 | **TCGplayer pricing comps** | Integration | Card identification | Pricing API integration on identified-variant. |
| 15 | **eBay sold-listings comps** | Integration | Pricing strategy | More involved; 130point licensing decision required. |
| 16 | **Buy/sell marketplace** | Frontend + backend | Phase 2 | Out of MVP scope. |

---

## Recommended next moves

Three good next sessions, by priority:

### Option A — Wait for corpus, then real corners trainer run
The corners trainer is committed; the launchd PSA ingest is dripping data daily. In ~3-5 days from 2026-04-29, the corpus crosses 200 samples, at which point `python -m training.trainers.corners train.smoke_only=true` runs end-to-end as a "proof the data pipeline plumbs through to a torch step" milestone. Real training (loss converging, useful val metrics) requires far more data; this run just proves the wiring.

### Option B — sit-with-me UX session: web capture flow
Camera permissions, error states, retake handling, multi-shot wizard order. Needs your input on UX decisions; can't be done autonomously. Pairs naturally with the now-shipped cert page (both touch apps/web). Unblocks Stripe + the full end-to-end demo flow.

### ~~Option C — operationalize the data flywheel~~ (shipped 2026-05-04, faea7d6)
Daily cycle is now a 3-step chain via `ml/scripts/daily_cycle.sh` (ingest → reference re-embed → exemplar accumulator). The plist hands off to the wrapper; per-step rc lines surface in `launchd.stdout.log` for triage. Each cycle bumps embedding-anomaly's `n_references` per identified variant.

### Option D — recalibrate counterfeit thresholds against real cards
The recalibration **tool** now exists (`evaluation.counterfeit_recalibration`, 97e3f71) — Youden's-J + FPR-budget cap, drop-in patch output. Authentic-only mode is usable today against the PSA `scraped.jsonl` (recalibrates the AUTHENTIC threshold, leaves COUNTERFEIT alone). Two-sided unlocks once a counterfeit slice is sourced (manual purchase / curation). Workflow: source counterfeits → `python -m evaluation.counterfeit_recalibration --csv counterfeits.csv --psa-authentics ~/psa_data/scraped.jsonl` → paste patch block into `ml/pipelines/counterfeit/ensemble.py` → re-run benchmark → lock + version in `model_versions`. The benchmark harness is the regression gate.

---

## Constraints and notes

### Things requiring your input
- AWS account / region / naming for Terraform
- Web capture UX decisions (permissions flow, error states, retake handling, multi-shot wizard order)
- Decision on paid PSA API tier — free is 100 calls/day; paid scales but pricing not public. Email `webcert@collectors.com` once daily ceiling becomes a bottleneck.
- Counterfeit confidence calibration anchors — once real authentic + counterfeit samples exist
- `reference_embeddings.npz` deployment path (bake-into-image vs S3 vs FAISS sidecar) — picks itself once API is being containerized for real

### Things NOT requiring your input (safe for autonomous agents)
- Bulk-data Scryfall ingest path (`data/ingestion/scryfall_bulk.py`) for full MTG catalog
- Typography counterfeit detector (gated on OCR dep decision)
- Geometric grading refinements (e.g., centering with non-uniform borders)
- Real training of any ML head — purely data-blocked at this point; nothing for an agent to do until corpus accumulates
- Operational-hygiene tasks: public accuracy benchmark publishing, capture-guidance acceptance test, latency SLO dashboards

### Architectural decisions paying off so far
- **Protocol-based interfaces** (`CardDetector`, `ImageEmbedder`, `CatalogIndex`) — production swap is one env var
- **Lazy-loaded ML models** — workers boot fast
- **Reject-rather-than-guess** — quality gate, dewarp irregularity check, identification confidence threshold all use this; lets advertised accuracy honestly beat competitors
- **Per-criterion grading models, not a single regressor** — centering at ~99%, edges at ~95% (synthetic), unblocks corners/surface to add later without architectural change
- **Hand-written migration with AST parity test** — migration won't drift silently from models even when modified by hand
- **Two's-complement bigint round-trip for pHash storage** — Postgres bigint signedness gotcha solved with helpers + boundary test
- **File-disjoint scoping for parallel agents** — three agents wrote 4.4k LOC simultaneously with zero merge conflicts
