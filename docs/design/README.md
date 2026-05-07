# Design source

Original design mockups for the Workbench / Capture / Cert / Pipeline / Notebook / Card pages, ported into the live `apps/web` Next.js app.

These files are reference material — standalone HTML pages that mount React via Babel-standalone. Not built or served. The `apps/web` implementation lives next to the rest of the frontend code.

| File | Maps to | Status |
|---|---|---|
| `workbench.html` + `app.jsx` | `apps/web/app/page.tsx` | **ported (2026-05-07)** |
| `capture.html` + `capture.jsx` | `apps/web/app/grade/[id]/page.tsx` | **ported (2026-05-07)** — wizard internals split to `apps/web/components/grade/Wizard.tsx` |
| `cert.html` + `cert.jsx` | `apps/web/app/cert/[id]/page.tsx` | **ported (2026-05-07)** |
| `pipeline.html` + `pipeline.jsx` | `apps/web/app/pipeline/[id]/page.tsx` | **ported (2026-05-07)** — design's mock playback clock replaced with status-driven stage states; per-stage telemetry from the API is a follow-up |
| `notebook.html` + `notebook.jsx` | (new) `apps/web/app/notebook/[id]/page.tsx` | **skipped** — design is an aggregate operator-metrics dashboard whose data sources don't exist (no regrade tracking, no agreement labels, no calibration history); 60% of the page would be empty state for personal use |
| `card.html` + `card.jsx` | (new) `apps/web/app/card/[variant_id]/page.tsx` | not yet ported |
| `shared.css` | `apps/web/app/globals.css` (CSS vars, nav classes) | merged |
| `pg-nav.js` | `apps/web/components/Nav.tsx` | ported |
| `components.jsx` | `apps/web/components/design/` | partially ported |
| `data.jsx` | (mock data, replaced by API calls) | superseded |
| `tweaks-panel.jsx` | (design demo only) | not ported, not needed |

To preview an original mockup, open the `.html` file directly in a browser — they pull React + Babel from CDN. The implementation deliberately does **not** match the mockups byte-for-byte; it preserves the aesthetic and information architecture while wiring everything to real API endpoints.
