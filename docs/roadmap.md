# Project Grader — Roadmap & Status

**Updated:** 2026-04-28

This doc tracks near-term work and current state. The architectural plan (4-phase roadmap, MVP scope, accuracy targets, business strategy) lives separately at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md` — that's the strategic baseline and shouldn't drift; this is the operational snapshot.

---

## Where we are

- **9 of 14 near-term roadmap items shipped** across 8 work sessions (the most recent was a 3-agent parallel burst).
- ~13,800 LOC across Python + TypeScript + YAML.
- **24 test files** (162+ tests), **4 inline runners** verifying **116 tests** passing without `uv sync` (prior to the Clerk JWT and FFT integration tests, which need fastapi+pytest+moto+pyjwt installed via `uv sync`).
- **2 commits** on `main` branch:
  - `45fd695` — foundation through edges grading
  - `6f54de7` — parallel work: FFT counterfeit + Clerk JWT + PSA scraper

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
                                    [Counterfeit ensemble]          ⚠ FFT built, NOT wired in yet
                                       └─ FFT rosette               ✓ built
                                       └─ Color profile             ⊘ TODO
                                       └─ Typography                ⊘ TODO
                                       └─ Embedding anomaly         ⊘ TODO
                                       └─ Substrate (paper)         ⊘ TODO
                                       └─ Holographic parallax      ⊘ needs tilt shot
                                       └─ Siamese reference         ⊘ needs reference set
                                              │
                                              ▼
                                    [Submission COMPLETED]
                                       Grade row: centering ✓ edges ✓ corners ⊘ surface ⊘ final ⊘
```

`final` stays NULL until corners + surface land. By design — we never fabricate a number.

---

## Done (9 items)

| # | Item | Commit | Notes |
|---|---|---|---|
| 1 | Alembic migration + parity test | 45fd695 | 8 tables, 5 enums, HNSW index, `pgvector`+`uuid-ossp` extensions. Migration parity-tested via AST. |
| 2 | S3/MinIO presigned upload + quality gating | 45fd695 | Per-shot-kind thresholds (flash tolerates more glare; corner zooms relaxed fill). Reject-rather-than-guess principle. |
| 3 | Card detection + dewarp (Stages 1+2) | 45fd695 | YOLOv11-seg lazy loader + Canny+contour heuristic fallback. 4-point homography to canonical 750×1050. Quad-irregularity gate. |
| 4 | Card identification (Stage 3) | 45fd695 | pHash + dHash, `ImageEmbedder` Protocol with `SimpleEmbedder` fallback and lazy `DinoV2Embedder` production path. `pgvector` catalog repo with `bit_count` Hamming + `<=>` cosine NN. Two's-complement bigint round-trip. |
| 5 | Celery worker pipeline | 45fd695 | Orchestrates detection → identification → grading. Per-stage audit log. Partial-grade persistence. |
| 6 | Centering grading | 45fd695 | Geometric measurement, ~99% accurate by construction. Front/back support; back-tolerance follows PSA. |
| 7 | Edges grading v1 | 45fd695 | Geometric perimeter-strip MAD anomaly detection. Per-side localization. Worse-face dominates. |
| 8 | **PSA pop-report scraper** | 6f54de7 | Resumable, rate-limited, 429-aware, S3+local stores. **Built but not yet running against real PSA.** 18 inline tests passing. |
| 9 | **FFT print-rosette counterfeit detector** | 6f54de7 | First detector in the 7-detector authenticity ensemble. Halftone Fourier signature → manufacturer-level counterfeit fingerprint. **Built but not yet wired into the pipeline runner.** 21 inline tests passing. |
| 10 | **Real Clerk JWT auth** | 6f54de7 | `dev_auth_enabled` flag with env-derived default. JWKSCache with TTL. RS256-only verification. Alg-confusion attack guarded. 512 lines of tests with mocked JWKS + generated RSA keys. |

---

## Near-term roadmap (refreshed)

In dependency / value order. Items marked **calendar-time** can run in the background while other work proceeds.

| # | Task | Type | Blocked on | Notes |
|---|---|---|---|---|
| 1 | **Wire FFT into pipeline_runner** | Small fix | — | ~20-line follow-up to #9. Adds counterfeit branch parallel to grading. Persist `authenticity_results` row. |
| 2 | **Run PSA scraper against real cert range** | Calendar-time | — | Build is done; just needs to actually run. Recommend small slice first (cert IDs in some narrow range) to validate HTML assumptions before going wide. |
| 3 | **Color profile counterfeit detector** | ML | — | Detector #4 in the ensemble. CIELAB histogram vs authentic distribution, white-border-calibrated. Same module shape as FFT. |
| 4 | **Typography counterfeit detector** | ML | OCR availability | Detector #5. OCR + glyph-shape comparison. Needs PaddleOCR or similar — a real new dep. |
| 5 | **Embedding anomaly counterfeit detector** | ML | Catalog data | Detector #7. Identification embedding distance vs distribution among known authentics. Needs a populated catalog with multiple authentic examples per variant. |
| 6 | **Persist authenticity results service** | API | — | Writes the `authenticity_results` row from the counterfeit ensemble output. Trivial after #1 + #3. |
| 7 | **Web capture flow** | Frontend | User UX input | Functional-only landing page. The user said "no design polish yet" but capture flows have UX choices (camera permissions, error states, retake). Should pair with you, not autonomous agent. |
| 8 | **Terraform dev env** | DevOps | User AWS choices | Account ID, region, naming, IAM strategy — needs your input. |
| 9 | **Corners ML model** | ML | PSA data + labeling | EfficientNet-V2-S backbone, 4 corner crops, ordinal regression. Needs labeled corner crops at scale. |
| 10 | **Surface ML model** | ML | PSA data + flash shots in fixtures | SegFormer-B3 multi-modal (no-flash + flash + tilt). The hardest model; fully blocked until data flywheel turns. |
| 11 | **Manufacturer reference image library** | Calendar-time | — | Per-manufacturer authentic-card photos for the reference-based counterfeit detectors (siamese, holographic). Could parallel the PSA scrape. |
| 12 | **Substrate paper detector** | ML | Reference library | Detector #6. Trained classifier on flash-shot paper region per manufacturer. |
| 13 | **Holographic parallax detector** | ML | Tilt shot capture flow | Detector #3. Two-angle optical flow on holo region. Needs the tilt shot in the capture flow. |
| 14 | **Siamese reference detector** | ML | Reference library | Detector #1. ConvNeXt-B siamese network with triplet loss. Most accurate on cards-with-references. |
| 15 | **Mobile app (RN/Expo)** | Frontend | Web capture flow | Phase 1 MVP scope explicitly excludes mobile. v0.2. |
| 16 | **Stripe payments** | Integration | Auth + capture flow | Free tier 5/mo, $10/mo unlimited per the plan. |
| 17 | **TCGplayer pricing comps** | Integration | Card identification | Pricing API integration on identified-variant. |
| 18 | **eBay sold-listings comps** | Integration | Pricing strategy | More involved; 130point licensing decision required. |
| 19 | **Public cert page** | Frontend | Grade output | Every grading produces a shareable URL — Phase 1 MVP feature. |
| 20 | **Buy/sell marketplace** | Frontend + backend | Phase 2 | Out of MVP scope. |

---

## Recommended next moves

Three good next sessions, by priority:

### Option A — close the FFT loop (1 session)
**Wire FFT into the pipeline runner** + **persist authenticity_results**. This is the smallest change with the most demo value: a submission ends with both a Grade row AND an AuthenticityResult row. Now the system actually outputs *both* of the differentiators (grade + authenticity) end-to-end.

### Option B — keep parallel agent momentum (3 sessions)
Spawn 2–3 more parallel agents while we have momentum:
- Color profile counterfeit detector (mirrors FFT structure)
- Authenticity persistence service
- Manufacturer reference scraper (kick off second calendar-time data track)

This is what justifies the worktree-scoped dispatch pattern.

### Option C — start producing data (1 session + calendar time)
**Run the PSA scraper against a small cert range** (say 10K cert IDs as the validation slice). Verify the HTML assumptions hold. Then expand to a larger range and let it run for weeks. This is the unblock-everything move for corners/surface ML.

---

## Constraints and notes

### Things requiring your input
- AWS account / region / naming for Terraform
- Web capture UX decisions (permissions flow, error states)
- HTML structure verification on a real PSA cert page (validates scraper assumptions before kicking off real scrape)
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
