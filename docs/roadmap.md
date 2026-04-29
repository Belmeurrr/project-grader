# Project Grader — Roadmap & Status

**Updated:** 2026-04-29

This doc tracks near-term work and current state. The architectural plan (4-phase roadmap, MVP scope, accuracy targets, business strategy) lives separately at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md` — that's the strategic baseline and shouldn't drift; this is the operational snapshot.

---

## Where we are

- **17 of 26 near-term roadmap items shipped** across 12 work sessions. The data flywheel is live (PSA Public API ingest running daily under launchd); the counterfeit ensemble has 2/7 detectors wired into the pipeline (FFT + color), 1 more built but not wired (embedding anomaly), and 4 still blocked on data prerequisites. The corners ML trainer skeleton is in place pending corpus growth.
- ~16,500 LOC across Python + TypeScript + YAML.
- **27 test files**; benchmark harness at `python -m evaluation.counterfeit_benchmark` runs in ~3s.
- **8 feature commits** on `main`:
  - `45fd695` — foundation through edges grading
  - `6f54de7` — parallel work: FFT counterfeit + Clerk JWT + PSA scraper
  - `c174171` — FFT wired into pipeline (Grade + AuthenticityResult rows)
  - `0c4a414` — PSA Public API client + GitHub seed-corpus ingestor
  - `ab43c6a` — calendar-time daily PSA ingest under launchd
  - `8ea54b4` — corners ML trainer skeleton
  - `ed29e35` + `94b83ca` — color-profile counterfeit detector built + wired
  - `65bbcdb` — embedding-anomaly counterfeit detector (pure fn; service wrapper deferred)
  - `00b056a` — refactor: dedupe canonical loader + lazy-load httpx imports
  - `6823e48` — counterfeit-detector benchmark + shared ensemble module
- **Calendar-time tracks running:**
  - PSA daily ingest (launchd `com.projectgrader.psa-ingest`, 09:00 local) — corpus at ~35 certs after first manual fire; expect ~30–50 successful triples/day on free tier.
  - Corpus needs ~165 more samples to cross the corners-trainer's `min_samples=200` threshold (~3-5 days at current rate).

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
                                    [Counterfeit ensemble]          ✓ rosette + color wired (Stage 3.5)
                                       └─ FFT rosette               ✓ built + wired
                                       └─ Color profile             ✓ built + wired (CIELAB chroma)
                                       └─ Embedding anomaly         ◐ pure fn built; wiring blocked on per-variant ref data
                                       └─ Typography                ⊘ TODO (needs OCR dep)
                                       └─ Substrate (paper)         ⊘ TODO (needs reference library)
                                       └─ Holographic parallax      ⊘ needs tilt shot
                                       └─ Siamese reference         ⊘ needs reference set
                                              │
                                              ▼
                                    [Submission COMPLETED]
                                       Grade row:           centering ✓ edges ✓ corners ⊘ surface ⊘ final ⊘
                                       AuthenticityResult:  rosette ✓  color ✓  others ⊘
```

`final` stays NULL until corners + surface land. By design — we never fabricate a number.

---

## Done (17 items)

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

---

## Near-term roadmap (refreshed)

In dependency / value order. Items marked **calendar-time** can run in the background while other work proceeds.

| # | Task | Type | Blocked on | Notes |
|---|---|---|---|---|
| 1 | **Public cert page** | Frontend | — | `/cert/[id]` route in apps/web. Displays Grade + AuthenticityResult for a submission. Phase 1 MVP. Doesn't need user UX input (data display only); the most demoable next step. |
| 2 | **First real corners trainer end-to-end run** | Run | corpus ≥ 200 (~3-5 days) | Skeleton committed; once corpus crosses `dataset.min_samples=200`, run `python -m training.trainers.corners train.smoke_only=true` to prove the torch step end-to-end. Loss is meaningless on tiny corpora; goal is "wiring works", not "model converges". |
| 3 | **Wire embedding-anomaly into pipeline_runner** | API | per-variant authentic exemplars | The pure function is committed. Wiring needs either (a) the manufacturer reference scraper output, or (b) a new `authentic_exemplars` table aggregating PSA ingest output by `SpecID`. Once exemplars exist for ≥1 variant the wiring is a c174171-shape commit. |
| 4 | **Manufacturer reference image scraper** | Calendar-time | non-corporate network access | Sources: Scryfall bulk (MTG), PokemonTCG.io (Pokemon), Topps press releases. Both Scryfall + PokemonTCG.io are reset by user's corporate network (TLS connection drops to certain Cloudflare zones); deferred until off-network or a different source works. Reference data feeds the siamese, substrate, holographic, embedding-anomaly detectors. |
| 5 | **Web capture flow** | Frontend | User UX input | Functional-only landing page. Capture flows have UX choices (camera permissions, error states, retake). Should pair, not autonomous agent. |
| 6 | **Terraform dev env** | DevOps | User AWS choices | Account ID, region, naming, IAM strategy — needs your input. |
| 7 | **Surface ML model trainer skeleton** | ML | — | Mirrors corners skeleton's pattern: PSACornerDataset → PSASurfaceDataset, regression head → SegFormer-B3 multi-modal. Skeleton is autonomous; full training is data-blocked. |
| 8 | **Typography counterfeit detector** | ML | OCR availability | Detector #5. OCR + glyph-shape comparison. Needs PaddleOCR or similar — a real new dep. |
| 9 | **Substrate paper detector** | ML | Reference library | Detector #6. Trained classifier on flash-shot paper region per manufacturer. |
| 10 | **Holographic parallax detector** | ML | Tilt shot capture flow | Detector #3. Two-angle optical flow on holo region. Needs the tilt shot in the capture flow. |
| 11 | **Siamese reference detector** | ML | Reference library | Detector #1. ConvNeXt-B siamese network with triplet loss. Most accurate on cards-with-references. |
| 12 | **Mobile app (RN/Expo)** | Frontend | Web capture flow | Phase 1 MVP scope explicitly excludes mobile. v0.2. |
| 13 | **Stripe payments** | Integration | Auth + capture flow | Free tier 5/mo, $10/mo unlimited per the plan. |
| 14 | **TCGplayer pricing comps** | Integration | Card identification | Pricing API integration on identified-variant. |
| 15 | **eBay sold-listings comps** | Integration | Pricing strategy | More involved; 130point licensing decision required. |
| 16 | **Buy/sell marketplace** | Frontend + backend | Phase 2 | Out of MVP scope. |

---

## Recommended next moves

Three good next sessions, by priority:

### Option A — Public cert page (1 session, autonomous)
`/cert/[id]` route in apps/web — displays Grade + AuthenticityResult for a submission. Phase 1 MVP item, frontend-only, doesn't need UX input (no user-input forms; data display only). The first piece of the system someone can actually click through to see the output. After this lands, all the backend work has a visible artifact.

### Option B — Wait for corpus, then real corners trainer run
The corners trainer is committed; the launchd PSA ingest is dripping data daily. In ~3-5 days the corpus crosses 200 samples, at which point we run `python -m training.trainers.corners train.smoke_only=true` end-to-end as a "proof the data pipeline plumbs through to a torch step" milestone. Real training (loss converging, useful val metrics) requires far more data; this run just proves the wiring.

### Option C — sit-with-me UX session
**Web capture flow.** Camera permissions, error states, retake handling, multi-shot wizard order. Needs your input on UX decisions; can't be done autonomously. Unblocks Stripe + the full demo flow. Pairs naturally with Option A.

### Option D — manufacturer reference scraper (off-corporate-network)
Scryfall + PokemonTCG.io are TLS-reset by user's corporate network (verified 2026-04-29). When off-network (home, mobile hotspot), the scraper builds in ~1 session. Unblocks the embedding-anomaly wiring + the substrate / siamese detectors that follow.

---

## Constraints and notes

### Things requiring your input
- AWS account / region / naming for Terraform
- Web capture UX decisions (permissions flow, error states)
- ~~PSA Public API token~~ — done 2026-04-29; token in user's shell as `PSA_PUBLIC_API_TOKEN`. Move to a `.env` once daily-budget runs start.
- Decision on paid PSA API tier (free is 100 calls/day; paid scales but pricing not public — email `webcert@collectors.com` once daily ceiling becomes a bottleneck)
- Counterfeit confidence calibration thresholds (currently set against synthetic fixtures; will retune with real data)

### Things NOT requiring your input (safe for autonomous agents)
- Additional counterfeit detectors that mirror FFT pattern (color, typography, embedding anomaly)
- Corners and surface ML training scripts (skeleton-only until data lands)
- Card catalog ingestion for Pokemon (PokemonTCG.io API) and MTG (Scryfall bulk)
- Geometric grading refinements (e.g., centering with non-uniform borders)

### Architectural decisions paying off so far
- **Protocol-based interfaces** (`CardDetector`, `ImageEmbedder`, `CatalogIndex`) — production swap is one env var
- **Lazy-loaded ML models** — workers boot fast
- **Reject-rather-than-guess** — quality gate, dewarp irregularity check, identification confidence threshold all use this; lets advertised accuracy honestly beat competitors
- **Per-criterion grading models, not a single regressor** — centering at ~99%, edges at ~95% (synthetic), unblocks corners/surface to add later without architectural change
- **Hand-written migration with AST parity test** — migration won't drift silently from models even when modified by hand
- **Two's-complement bigint round-trip for pHash storage** — Postgres bigint signedness gotcha solved with helpers + boundary test
- **File-disjoint scoping for parallel agents** — three agents wrote 4.4k LOC simultaneously with zero merge conflicts
