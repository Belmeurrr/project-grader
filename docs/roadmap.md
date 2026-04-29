# Project Grader — Roadmap & Status

**Updated:** 2026-04-28

This doc tracks near-term work and current state. The architectural plan (4-phase roadmap, MVP scope, accuracy targets, business strategy) lives separately at `C:\Users\Belme\.claude\plans\i-want-to-build-sequential-crane.md` — that's the strategic baseline and shouldn't drift; this is the operational snapshot.

---

## Where we are

- **11 of 20 near-term roadmap items shipped** across 10 work sessions (most recent: PSA Public API client + GitHub seed-corpus ingestor scaffolded and parser-verified against 4 real cert classes — pending dependency declaration, first end-to-end run, and commit).
- ~14,500 LOC across Python + TypeScript + YAML (+~700 LOC this session).
- **24 test files** (162+ tests), **5 inline runners** verifying **152 tests** passing without `uv sync`.
- **3 feature commits** on `main` branch:
  - `45fd695` — foundation through edges grading
  - `6f54de7` — parallel work: FFT counterfeit + Clerk JWT + PSA scraper
  - `c174171` — wire FFT counterfeit detector into pipeline_runner (Grade + AuthenticityResult rows produced end-to-end)
- **Uncommitted (this session):** [ml/data/ingestion/psa_public_api.py](ml/data/ingestion/psa_public_api.py), [ml/data/ingestion/github_seed.py](ml/data/ingestion/github_seed.py), [ml/scripts/psa_one_cert_smoke.py](ml/scripts/psa_one_cert_smoke.py); roadmap+TODO doc updates; project memory entries documenting verified PSA API shapes.

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
                                    [Counterfeit ensemble]          ✓ FFT wired (rosette only)
                                       └─ FFT rosette               ✓ built + wired (Stage 3.5)
                                       └─ Color profile             ⊘ TODO
                                       └─ Typography                ⊘ TODO
                                       └─ Embedding anomaly         ⊘ TODO
                                       └─ Substrate (paper)         ⊘ TODO
                                       └─ Holographic parallax      ⊘ needs tilt shot
                                       └─ Siamese reference         ⊘ needs reference set
                                              │
                                              ▼
                                    [Submission COMPLETED]
                                       Grade row:           centering ✓ edges ✓ corners ⊘ surface ⊘ final ⊘
                                       AuthenticityResult:  rosette ✓  others ⊘
```

`final` stays NULL until corners + surface land. By design — we never fabricate a number.

---

## Done (11 items)

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

---

## Near-term roadmap (refreshed)

In dependency / value order. Items marked **calendar-time** can run in the background while other work proceeds.

| # | Task | Type | Blocked on | Notes |
|---|---|---|---|---|
| 1 | **First end-to-end PSA Public API ingest** (5–10 certs) | Run | httpx pyproject decl | Module is built and parser-verified against 4 real cert classes; remaining unknowns are the storage path + image-bytes download. One run against ~10 modern cert IDs exercises everything: budget consumption, dual-cert skip, image hot-link, ScrapedRecord write to JSONL on disk. ~10 of 100 daily API calls. Once it passes, the calendar-time daily-budget run can start. |
| 2 | **First GitHub seed manifest write** | Run | — | Module built. `git clone samsilverman/PSA-Baseball-Grades` then call `write_manifest`. Confirms 11,500-line manifest with 1,150 per grade. |
| 3 | **Color profile counterfeit detector** | ML | — | Detector #4 in the ensemble. CIELAB histogram vs authentic distribution, white-border-calibrated. Same module shape as FFT. |
| 4 | **Typography counterfeit detector** | ML | OCR availability | Detector #5. OCR + glyph-shape comparison. Needs PaddleOCR or similar — a real new dep. |
| 5 | **Embedding anomaly counterfeit detector** | ML | Catalog data | Detector #7. Identification embedding distance vs distribution among known authentics. Needs a populated catalog with multiple authentic examples per variant. |
| 6 | **Web capture flow** | Frontend | User UX input | Functional-only landing page. The user said "no design polish yet" but capture flows have UX choices (camera permissions, error states, retake). Should pair with you, not autonomous agent. |
| 7 | **Terraform dev env** | DevOps | User AWS choices | Account ID, region, naming, IAM strategy — needs your input. |
| 8 | **Corners ML model** | ML | PSA data + labeling | EfficientNet-V2-S backbone, 4 corner crops, ordinal regression. Needs labeled corner crops at scale. |
| 9 | **Surface ML model** | ML | PSA data + flash shots in fixtures | SegFormer-B3 multi-modal (no-flash + flash + tilt). The hardest model; fully blocked until data flywheel turns. |
| 10 | **Manufacturer reference image library** | Calendar-time | — | Per-manufacturer authentic-card photos for the reference-based counterfeit detectors (siamese, holographic). Could parallel the PSA scrape. |
| 11 | **Substrate paper detector** | ML | Reference library | Detector #6. Trained classifier on flash-shot paper region per manufacturer. |
| 12 | **Holographic parallax detector** | ML | Tilt shot capture flow | Detector #3. Two-angle optical flow on holo region. Needs the tilt shot in the capture flow. |
| 13 | **Siamese reference detector** | ML | Reference library | Detector #1. ConvNeXt-B siamese network with triplet loss. Most accurate on cards-with-references. |
| 14 | **Mobile app (RN/Expo)** | Frontend | Web capture flow | Phase 1 MVP scope explicitly excludes mobile. v0.2. |
| 15 | **Stripe payments** | Integration | Auth + capture flow | Free tier 5/mo, $10/mo unlimited per the plan. |
| 16 | **TCGplayer pricing comps** | Integration | Card identification | Pricing API integration on identified-variant. |
| 17 | **eBay sold-listings comps** | Integration | Pricing strategy | More involved; 130point licensing decision required. |
| 18 | **Public cert page** | Frontend | Grade output | Every grading produces a shareable URL — Phase 1 MVP feature. |
| 19 | **Buy/sell marketplace** | Frontend + backend | Phase 2 | Out of MVP scope. |

---

## Recommended next moves

Three good next sessions, by priority:

### Option A — close the data-flywheel loop (1 session + calendar time)
Add `httpx` to `ml/pyproject.toml`, run `ingest_range` against ~10 modern certs end-to-end (item #1), run `write_manifest` against the cloned GitHub seed repo (item #2), then commit the new ingestion modules. After this session the calendar-time PSA drip can start producing ~50 authoritative cert triples/day. **Background**: the API client and seed ingestor were both scaffolded and parser-verified against 4 real cert classes (Pokemon Mint, half-grade, mid-grade baseball, PSA/DNA autograph) — what's left is the actual first run, which exercises the storage + image-download paths the smoke test couldn't cover.

### Option B — keep parallel agent momentum (2-3 sessions)
Spawn 2–3 more parallel agents while we have momentum:
- Color profile counterfeit detector (mirrors FFT structure)
- Embedding anomaly counterfeit detector
- Manufacturer reference scraper (kick off second calendar-time data track)

This is what justifies the worktree-scoped dispatch pattern. Disjoint files, no merge conflicts.

### Option C — sit-with-me UX session
**Web capture flow.** Camera permissions, error states, retake handling, multi-shot wizard order. Needs your input on UX decisions; can't be done autonomously. Unblocks Stripe + public cert page.

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
