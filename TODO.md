# TODO

Action-oriented checklist of what's left. Full operational snapshot in [docs/roadmap.md](docs/roadmap.md). Strategic plan (4-phase business roadmap, MVP scope, accuracy targets) lives at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md`.

Each item has enough context to pick it up cold without reloading session state.

---

## Now — do next (1 session each)

> **Status as of 2026-04-29 (late session)**: Counterfeit ensemble is now **3/7 wired** end-to-end (FFT rosette + color profile + embedding-anomaly). Manufacturer reference flywheel is live for both MTG (Scryfall) and Pokemon (PokemonTCG.io). All four ML-head trainer skeletons (corners, surface, identification/DinoV2, detection/YOLO) are committed; real training stays data-blocked on PSA corpus growth + variant-linkage work. The next high-leverage step is the public cert page — the first user-clickable surface that ties Grade + AuthenticityResult together.

- [ ] **Public cert page** — `/cert/[id]` route in apps/web
  - Phase 1 MVP requirement (every grading produces a shareable URL)
  - Frontend-only, no UX input required (data display, no user-input forms)
  - Fetches Grade + AuthenticityResult for a submission, renders all three detectors' scores + per-detector verdicts + combined verdict
  - The first piece of the system someone can actually click through to see end-to-end output. Highest demoability per session.

- [ ] **First real corners trainer run** (when corpus crosses 200 samples)
  - At current ingest rate (~30-50/day), expected ~3-5 days from 2026-04-29
  - `cd ml && uv run python -m training.trainers.corners train.smoke_only=true train.epochs=2`
  - Goal: prove the data pipeline plumbs through to a torch step. NOT "model converges" (corpus too small).
  - Expected: `dataset.min_samples=200` gate passes; one mini-batch flows through the EfficientNet-V2-S backbone; checkpoint written to `outputs/corners/best.pt`
  - First useful val signal needs ~1k samples (~3-4 weeks at free-tier rate, OR email `webcert@collectors.com` for paid tier).

- [ ] **Recalibrate counterfeit thresholds** — three detectors now, all on synthetic-fixture defaults
  - All thresholds in `ml/pipelines/counterfeit/ensemble.py` are calibrated against synthetic generators (the `synth_*` fixtures).
  - Embedding-anomaly added 2026-04-29 with placeholder thresholds (0.65 / 0.35 / 0.4) mirroring the rosette + color shape — needs real-data anchors before they're trustworthy.
  - Once recalibrated, lock the thresholds and version them in `model_versions` metadata. Benchmark harness (`python -m evaluation.counterfeit_benchmark`) is the regression gate.
  - Needs: a small labeled set with both real authentics (from PSA ingest, identified to known variants so embedding-anomaly fires) + known counterfeits.

---

## Soon — parallel-agent friendly (file-disjoint, no user input needed)

- [ ] **Bulk-data Scryfall ingest path** — corpus-scale alternative to the paginated client
  - `data/ingestion/scryfall.py` currently uses `/cards/search` which has a ~1750-result soft cap. Fine for set-by-set work, but to ingest the full ~25k-printing MTG catalog efficiently, use the bulk JSON endpoint (`/bulk-data` returns metadata, then GET the `default_cards` ~538 MB tarball).
  - New module `data/ingestion/scryfall_bulk.py` that streams the bulk JSON, reuses the same `LocalReferenceStore`. Public surface mirrors `ingest_query`.
  - Image download stays per-card against the existing CDN — bulk-data only saves on metadata round-trips.

- [ ] **Per-variant exemplar accumulation from PSA ingest** (broaden embedding-anomaly coverage)
  - Embedding-anomaly currently only has signal for cards whose variant is in the manufacturer reference store. PSA-graded submissions identified to the same variant ARE additional authentic exemplars — accumulate them into the same npz under the same key (the storage layer already handles `(n_refs, d)` arrays).
  - Builder script: walk PSA `scraped.jsonl`, for each record run identification, when identification picks a variant append the PSA submission's embedding to the variant's reference array.
  - Bumps `n_references` per variant which directly increases embedding-anomaly confidence.

- [ ] **Web capture flow** — functional-only capture pages
  - UX decisions: camera permissions flow, error states, retake handling, multi-shot wizard order
  - Pages to build: `/grade` (capture), `/grade/[id]` (status + in-progress)
  - The user said "no design polish yet" — but the capture flow still has UX choices
  - Pairs naturally with the public cert page (in "Now") since both touch apps/web

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
- [ ] **MTG catalog ingestion (DB-side)** — populate `card_variants` from Scryfall. Distinct from the manufacturer reference scrape: this is the SQL catalog the API queries at submission time, vs. the on-disk reference image library used by counterfeit detectors. Daily refresh job.
- [ ] **Pokemon catalog ingestion (DB-side)** — same pattern, PokemonTCG.io as source.
- [ ] **Real production deploy** — Aurora Postgres, ElastiCache, ECS Fargate API, EKS GPU workers, Triton inference, MLflow registry. Build out via Terraform after dev env exists.
- [ ] **CI** — GitHub Actions: `uv sync && pytest` on each PR for `apps/api` and `ml`. Type-check + lint gates. Coverage tracking.

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
- [ ] **Manufacturer reference re-embed cycle** — after each new Scryfall / PokemonTCG ingest run, re-run `python -m scripts.embed_references` so newly-ingested variants land in `reference_embeddings.npz` and become eligible for embedding-anomaly. Idempotent by `(manufacturer, variant_id)` key; the pass is fast for incremental adds.
- [ ] Public accuracy benchmark — predictions vs subsequent PSA grades, published monthly (trust-strategy item from plan).
- [ ] Capture-guidance acceptance test — real users on mid-range phones; rejection rate and retakes/submission. Target <2 retakes median.
- [ ] Latency SLO monitoring — p95 ≤ 20s, p99 ≤ 35s (Datadog dashboards + PagerDuty alerts).

---

## Recently shipped (2026-04-29 session)

- **Manufacturer reference scrapers** — `data/ingestion/scryfall.py` (faab948) + `data/ingestion/pokemontcg.py` (d24691a). CLI: `python -m scripts.manufacturer_refs_ingest --source {mtg,pokemon} --query "..."`.
- **Reference embedding pass** — `data/ingestion/reference_embeddings.py` (6caa8a1). `embed_references` walks the references JSONL, embeds with the identification embedder, writes a single npz keyed by `"<manufacturer>/<variant_id>"`. CLI: `python -m scripts.embed_references --data-dir ...`. Lookup helpers: `load_embeddings`, `lookup_references`.
- **Surface trainer skeleton** — `training/trainers/surface.py` (31b7adb). EfficientNet-V2-S encoder + SegFormer-style decoder, 7-class CrossEntropyLoss. Default `mask_loader` returns all-background until real per-pixel labels exist.
- **DinoV2 identification trainer skeleton** — `training/trainers/identification.py` (41e68f2). ViT-B/14 + triplet loss. Augmentation-based positives until variant_id linkage exists. Image size must be a multiple of 14 (hard ViT constraint).
- **YOLO detection manifest builder** — `data/catalogs/build_detection_manifest.py` (09fe2a4). Synthesizes labeled scenes from `card_in_scene` fixtures with analytically-derived bboxes. Manifest at `ml/data/catalogs/detection_dataset.yaml`.
- **Embedding-anomaly counterfeit detector wired** — `services/counterfeit.py` + `workers/pipeline_runner.py` (c5af923). Stage 3.5 now runs all three detectors. Graceful UNVERIFIED on no-references / unidentified paths so the ensemble degrades cleanly.
