# TODO

Action-oriented checklist of what's left. Full operational snapshot in [docs/roadmap.md](docs/roadmap.md). Strategic plan (4-phase business roadmap, MVP scope, accuracy targets) lives at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md`.

Each item has enough context to pick it up cold without reloading session state.

---

## Now — do next (1 session each)

- [ ] **Wire FFT counterfeit detector into `pipeline_runner`**
  - Add a counterfeit branch after detection+dewarp: call `services.counterfeit.analyze_rosette(front_canonical_s3_key)`
  - New function `persist_authenticity_result(submission_id, result, db)` in [apps/api/grader/services/counterfeit.py](apps/api/grader/services/counterfeit.py) — writes the `authenticity_results` row using the existing schema
  - New audit log entries: `pipeline.counterfeit.started`, `pipeline.counterfeit.completed`
  - E2E test in [test_pipeline_runner.py](apps/api/tests/test_pipeline_runner.py): a submission produces *both* a Grade row AND an AuthenticityResult row
  - Why first: this is the smallest change with the biggest demo value — the system finally outputs both differentiators (grade + authenticity) end-to-end

- [ ] **Validate PSA HTML assumptions against a real cert page**
  - Fetch one cert page by hand (`https://www.psacard.com/cert/<some_known_cert_id>`)
  - Compare structure to the parser assumptions documented in [ml/data/ingestion/psa_pop_scraper.py](ml/data/ingestion/psa_pop_scraper.py) docstring
  - Adjust regex tuples at the top of the file if structure differs
  - Why first: doing this *before* the calendar-time scrape kicks off saves wasted scraping if the parser is wrong

---

## Soon — parallel-agent friendly (file-disjoint, no user input needed)

- [ ] **Color profile counterfeit detector** (ensemble detector #4)
  - New module: `ml/pipelines/counterfeit/color/measure.py`
  - CIELAB histogram of card vs expected authentic distribution; white-border-calibrated to neutralize lighting cast
  - Mirror module shape exactly from [ml/pipelines/counterfeit/rosette/measure.py](ml/pipelines/counterfeit/rosette/measure.py)
  - Service wrapper in [apps/api/grader/services/counterfeit.py](apps/api/grader/services/counterfeit.py): `analyze_color_profile(canonical_s3_key) -> ColorMeasurement`

- [ ] **Embedding anomaly counterfeit detector** (ensemble detector #7)
  - New module: `ml/pipelines/counterfeit/embedding_anomaly/measure.py`
  - Compare submitted card's identification embedding to the distribution among known authentics for that variant
  - Flag cards in the 99th percentile of embedding distance from the variant's authentic centroid
  - Depends on a populated catalog with multiple authentic examples per variant — degrades gracefully when only 1 example exists

- [ ] **Run PSA scraper against real cert range** (calendar-time)
  - Build is done; just needs to actually run
  - Recommend small slice first: pick 10,000 cert IDs in some narrow range (e.g., a known set's range), validate output, then expand
  - Output: S3 bucket prefix or local disk
  - Default rate limit (1 req/sec) + jitter — let it run for days
  - Watch the stats: `successful`, `not_found`, `parse_errors`, `rate_limited`. If `parse_errors > 5%` of `successful`, the parser needs fixing before going wider

- [ ] **Manufacturer reference image scraper** (second data track)
  - Per-manufacturer authentic-card photos. Sources: Pokemon Company press kits, Topps press releases, Wizards (Scryfall already has high-res images)
  - Reference data feeds the siamese, substrate, holographic detectors
  - New module: `ml/data/ingestion/manufacturer_references.py`
  - Can run in parallel with the PSA scrape; entirely disjoint storage

- [ ] **Refactor `_load_canonical_bgr` into shared helper** (~6 lines duplicated)
  - Currently in both `services/grading.py` and `services/counterfeit.py`
  - Move to `services/_canonical.py` or just `services/storage.py`
  - Update both callers
  - Trivial; do when a 3rd detector arrives

---

## Soon — needs your input (sit-with-me sessions)

- [ ] **Web capture flow** — functional-only landing page
  - UX decisions: camera permissions flow, error states, retake handling, multi-shot wizard order
  - Pages to build: `/grade` (capture), `/grade/[id]` (status + result), `/cert/[id]` (public cert page stub)
  - The user said "no design polish yet" — but functional capture flow still has UX choices

- [ ] **Terraform dev environment**
  - File: `infra/terraform/envs/dev/main.tf`
  - Needs your AWS account ID, region, naming convention, IAM strategy
  - Resources: S3 bucket, Aurora Postgres + pgvector, ElastiCache Redis, ECR for Docker images, IAM roles
  - Don't deploy real infrastructure on autopilot

- [ ] **Decide policy on JWT email-claim updates**
  - Current behavior: `User.email` is NOT overwritten on re-login if Clerk's JWT carries a different email
  - Has GDPR/consent angle — worth a deliberate decision
  - One-line fix in `_get_or_create_user` in [services/auth.py](apps/api/grader/services/auth.py) once decided

- [ ] **Counterfeit confidence calibration thresholds**
  - Current FFT thresholds calibrated against *synthetic* halftone fixtures
  - Real cards may need re-calibration; needs a small labeled set (synthetic + a handful of real authentics + known counterfeits)
  - Once recalibrated, lock the thresholds and version them in the model_versions metadata

---

## Blocked on data (PSA scraper needs to run for weeks)

- [ ] **Corners ML model** — EfficientNet-V2-S backbone, 4 corner crops, ordinal regression head per corner. Final corner subgrade = min(4) per PSA. Needs labeled corner crops at scale.
- [ ] **Surface ML model** — SegFormer-B3 multi-modal (no-flash + flash + optional tilt). Output mask per defect class {scratch, print_line, indentation, stain, paper_loss, foil_scratch}. Hardest model in the system.
- [ ] **DinoV2 fine-tune for card identification** — replaces the public `facebook/dinov2-base` baseline with card-specific weights. Trainer scaffold goes in `ml/training/trainers/identification.py`.
- [ ] **YOLOv11-seg fine-tune for card detection** — config already at [ml/training/configs/detection.yaml](ml/training/configs/detection.yaml); trainer at [ml/training/trainers/detection.py](ml/training/trainers/detection.py). Just needs a labeled dataset manifest.

---

## Blocked on other work

- [ ] **Typography counterfeit detector** (ensemble #5) — needs PaddleOCR or similar (new dep). OCR + glyph-shape comparison to canonical font templates.
- [ ] **Substrate paper detector** (ensemble #6) — needs the manufacturer reference image library to train per-manufacturer paper-fluorescence classifiers.
- [ ] **Holographic parallax detector** (ensemble #3) — needs the tilt shot in the capture flow. Two-angle optical flow on holo region.
- [ ] **Siamese reference detector** (ensemble #1) — needs reference image library. ConvNeXt-B siamese with triplet loss.

---

## Later — Phase 1 MVP polish

- [ ] **Public cert page** — every grading produces a shareable URL (`/cert/<submission_id>`); ISR-cacheable, SEO-indexable. Required for MVP per the plan.
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

- [ ] Public accuracy benchmark — published monthly, predictions vs subsequent PSA grades. Per the trust strategy in the plan.
- [ ] Counterfeit benchmark — curated test set (50 known authentics + 50 known counterfeits), refreshed quarterly. CI gate on recall ≥ 90% / FPR ≤ 2%.
- [ ] Capture-guidance acceptance test — record real users with mid-range phones; measure rejection rate and retakes/submission. Target <2 retakes median.
- [ ] Latency SLO monitoring — p95 ≤ 20s, p99 ≤ 35s (Datadog dashboards + PagerDuty alerts)
