# TODO

Action-oriented checklist of what's left. Full operational snapshot in [docs/roadmap.md](docs/roadmap.md). Strategic plan (4-phase business roadmap, MVP scope, accuracy targets) lives at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md`.

Each item has enough context to pick it up cold without reloading session state.

---

## Now — do next (1 session each)

> **Status as of 2026-04-29**: The data flywheel is live (PSA Public API ingest running daily under launchd, ~30–50 cert triples/day). Counterfeit ensemble has 2/7 detectors wired (FFT + color); embedding-anomaly built but unwired pending per-variant reference data. Corners ML trainer skeleton in place pending corpus growth. Counterfeit benchmark harness running (`python -m evaluation.counterfeit_benchmark`).

- [ ] **Public cert page** — `/cert/[id]` route in apps/web
  - Phase 1 MVP requirement (every grading produces a shareable URL)
  - Frontend-only, no UX input required (data display, no user-input forms)
  - Fetches Grade + AuthenticityResult for a submission, renders both detectors' scores + per-detector verdicts + combined verdict
  - The first piece of the system someone can actually click through to see end-to-end output. Highest demoability per session.

- [ ] **First real corners trainer run** (when corpus crosses 200 samples)
  - At current ingest rate (~30-50/day), expected ~3-5 days from now
  - `cd ml && uv run python -m training.trainers.corners train.smoke_only=true train.epochs=2`
  - Goal: prove the data pipeline plumbs through to a torch step. NOT "model converges" (corpus too small).
  - Expected: `dataset.min_samples=200` gate passes; one mini-batch flows through the EfficientNet-V2-S backbone; checkpoint written to `outputs/corners/best.pt`
  - First useful val signal needs ~1k samples (~3-4 weeks at free-tier rate, OR email `webcert@collectors.com` for paid tier).

- [ ] **Wire embedding-anomaly into pipeline_runner** (when per-variant reference exemplars exist)
  - Pure function is committed at `ml/pipelines/counterfeit/embedding_anomaly/measure.py` — ready to use.
  - Blocked on: per-variant authentic exemplars store. Two unblock paths:
    1. **Manufacturer reference scraper output** (item below) — gives 1+ canonical refs per variant
    2. **PSA ingest aggregation by SpecID** — once a variant accumulates ≥3 PSA-graded samples in our corpus, those are exemplars
  - Once unblocked: c174171-shape commit. Add `analyze_embedding_anomaly` service wrapper, extend Stage 3.5 ensemble to 3 detectors.

---

## Soon — parallel-agent friendly (file-disjoint, no user input needed)

- [ ] **Manufacturer reference image scraper** — try when off corporate network
  - Sources: Scryfall bulk JSON (MTG, ~25k cards), PokemonTCG.io API (~16k cards). Both are TLS-reset by user's corporate network (verified 2026-04-29 — `Connection reset by peer` to api.scryfall.com + api.pokemontcg.io while github.com + api.psacard.com work fine; corporate SNI filtering of certain Cloudflare zones).
  - Storage layout (planned): `~/manufacturer_refs/references.jsonl` + `images/<manufacturer>/<variant_id>/front.<ext>`. Disjoint from the PSA store.
  - Output unblocks: embedding-anomaly wiring, siamese reference detector, substrate paper detector.

- [ ] **Surface ML model trainer skeleton**
  - Mirror the corners trainer's pattern (commit 8ea54b4) for the surface-defect head
  - PSASurfaceDataset reads scraped.jsonl, loads canonical, passes through a SegFormer-B3-style backbone (multi-class semantic segmentation)
  - Same Hydra+MLflow shape as corners.py; same `min_samples` gate
  - Real training is data-blocked (needs flash-shot data + per-defect-class labels); skeleton is autonomous

- [ ] **DinoV2 fine-tune trainer skeleton for card identification**
  - Currently using `facebook/dinov2-base` baseline embedder. Fine-tune scaffold goes in `ml/training/trainers/identification.py`
  - Pattern: same Hydra+MLflow as detection.py + corners.py
  - Triplet-loss objective on (anchor, positive, negative) sampled from the catalog

- [ ] **YOLOv11-seg fine-tune for card detection**
  - Config + trainer already exist (commit 45fd695). Just needs a labeled dataset manifest at `ml/data/catalogs/detection_dataset.yaml`
  - Manifest is built from labeled fixtures + (eventually) PSA images with bounding boxes

---

## Soon — needs your input (sit-with-me sessions)

- [ ] **Web capture flow** — functional-only capture pages
  - UX decisions: camera permissions flow, error states, retake handling, multi-shot wizard order
  - Pages to build: `/grade` (capture), `/grade/[id]` (status + in-progress)
  - The user said "no design polish yet" — but the capture flow still has UX choices
  - Pairs naturally with the public cert page (in "Now") since both touch apps/web

- [ ] **Terraform dev environment**
  - File: `infra/terraform/envs/dev/main.tf`
  - Needs your AWS account ID, region, naming convention, IAM strategy
  - Resources: S3 bucket, Aurora Postgres + pgvector, ElastiCache Redis, ECR for Docker images, IAM roles
  - Don't deploy real infrastructure on autopilot

- [ ] **Decide policy on JWT email-claim updates**
  - Current behavior: `User.email` is NOT overwritten on re-login if Clerk's JWT carries a different email
  - Has GDPR/consent angle — worth a deliberate decision
  - One-line fix in `_get_or_create_user` in [services/auth.py](apps/api/grader/services/auth.py) once decided

- [ ] **Counterfeit confidence calibration thresholds** — both detectors
  - All thresholds (rosette + color, in `ml/pipelines/counterfeit/ensemble.py`) are calibrated against *synthetic* fixtures
  - Real cards may need re-calibration; needs a small labeled set (synthetic + real authentics from PSA ingest + known counterfeits)
  - Once recalibrated, lock the thresholds and version them in `model_versions` metadata. Benchmark harness (`python -m evaluation.counterfeit_benchmark`) is the regression gate.

---

## Blocked on data accumulation (PSA daily ingest is running)

- [ ] **Corners ML model — real training** — Skeleton committed (8ea54b4). Refuses to train below 200 samples; expected to cross threshold ~3-5 days from 2026-04-29. First useful val signal needs ~1k samples.
- [ ] **Surface ML model — real training** — Hardest model in the system. Needs flash-shot data (currently unavailable) + per-defect-class labels (currently unlabeled).
- [ ] **DinoV2 / YOLOv11-seg fine-tunes** — Identification + detection both blocked on labeled-at-scale dataset manifests.

---

## Blocked on other work

- [ ] **Typography counterfeit detector** (ensemble #5) — needs PaddleOCR or similar (new dep). OCR + glyph-shape comparison to canonical font templates.
- [ ] **Substrate paper detector** (ensemble #6) — needs the manufacturer reference image library to train per-manufacturer paper-fluorescence classifiers.
- [ ] **Holographic parallax detector** (ensemble #3) — needs the tilt shot in the capture flow. Two-angle optical flow on holo region.
- [ ] **Siamese reference detector** (ensemble #1) — needs reference image library. ConvNeXt-B siamese with triplet loss.

---

## Later — Phase 1 MVP polish

- [ ] **Stripe payments** — free 5/mo, $10/mo unlimited. Webhook handler for subscription lifecycle.
- [ ] **TCGplayer pricing comps** — pull current market price on identified-variant. Show predicted-grade-adjusted EV (`Σ P(grade) × median_price(grade)`).
- [ ] **MTG catalog ingestion** — Scryfall bulk JSON, daily refresh job
- [ ] **Pokemon catalog ingestion** — PokemonTCG.io API + TCGplayer fill-ins
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

- [x] **Counterfeit benchmark — v1 corpus + harness** (committed 6823e48). Run: `python -m evaluation.counterfeit_benchmark`. Current: 50-sample synthetic corpus, rosette 98% accuracy / color 100% / ensemble 98%. **Still TODO**: swap in real images as the corpus grows, then add the hard CI gate (recall ≥ 90% / FPR ≤ 2%).
- [ ] Public accuracy benchmark — predictions vs subsequent PSA grades, published monthly (trust-strategy item from plan).
- [ ] Capture-guidance acceptance test — real users on mid-range phones; rejection rate and retakes/submission. Target <2 retakes median.
- [ ] Latency SLO monitoring — p95 ≤ 20s, p99 ≤ 35s (Datadog dashboards + PagerDuty alerts).
