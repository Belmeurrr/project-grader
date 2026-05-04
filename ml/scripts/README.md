# ml/scripts

Operational scripts. Not imported by application code; meant to be invoked from the command line or under launchd / cron.

## `daily_cycle.sh` — daily data-flywheel chain (launchd entry point)

The launchd plist (`com.projectgrader.psa-ingest.plist`) runs this wrapper, which in turn invokes three Python scripts in sequence:

1. **`psa_daily_ingest.py`** — drains the PSA Public API daily budget into `~/psa_data/`.
2. **`embed_references.py`** — embeds any newly-added manufacturer references (Scryfall / PokemonTCG.io) into `~/manufacturer_refs/reference_embeddings.npz`. Idempotent on `(manufacturer, variant_id)`; a no-op on days with no new ingest.
3. **`accumulate_psa_exemplars.py`** — walks today's PSA records, identifies each against the catalog, and appends matched-submission embeddings to the same npz under their variant key. Idempotent by `cert_id`. Bumps the embedding-anomaly detector's `n_references` per variant the next day.

Each step runs independently — a non-zero exit from step 1 (e.g. PSA budget exhausted, auth error) does **not** prevent steps 2 and 3 from running, because they have value on their own. The wrapper aggregates exit codes: it returns the first non-zero rc it saw, or 0 if all three succeeded. Look at the `launchd.stdout.log` per-step "rc=N" lines to triage which step failed.

### Configuration knobs (env, with defaults the wrapper supplies)

| Var | Default | Purpose |
|---|---|---|
| `PROJECT_ROOT` | `/Users/danielbelmeur/Github/project-grader` | repo root the wrapper cd's into |
| `PYTHON` | `/usr/bin/python3` | Python interpreter for all three steps |
| `PSA_INGEST_DATA_DIR` | `$HOME/psa_data` | where PSA scraped.jsonl + state.json + images live |
| `MANUFACTURER_REFS_DATA_DIR` | `$HOME/manufacturer_refs` | where references.jsonl + reference_embeddings.npz live |

The wrapper sources `$PROJECT_ROOT/.env` if present so secrets like `PSA_PUBLIC_API_TOKEN` propagate to all three child processes (see one-time setup below).

### Smoke-test (bypass launchd)

```sh
cd /Users/danielbelmeur/Github/project-grader
PROJECT_ROOT="$PWD" ml/scripts/daily_cycle.sh
```

This runs the full chain immediately and prints per-step timing + rc to stdout.

## `psa_daily_ingest.py` — calendar-time PSA Public API ingest

Drives one day's worth of PSA cert ingestion against the free-tier 100-call/day ceiling. Each run consumes the daily budget, advances a state-file cursor, and appends a stats row to `<data_dir>/ingest_log.jsonl`. The expected steady-state output is ~30–50 successful (cert metadata + front/back jpgs) per day on the free tier, with not_found / non_gradeable / images_missing accumulating in the same JSONL store.

### One-time setup

1. Create a project-root `.env` (already gitignored — see [.gitignore](../../.gitignore)):

   ```sh
   cat > /Users/danielbelmeur/Github/project-grader/.env <<EOF
   PSA_PUBLIC_API_TOKEN=your-token-here
   PSA_INGEST_DATA_DIR=$HOME/psa_data
   PSA_INGEST_START_CERT_ID=80000010
   EOF
   chmod 600 /Users/danielbelmeur/Github/project-grader/.env
   ```

   Token is obtained at [psacard.com/publicapi](https://www.psacard.com/publicapi) (self-serve, free tier = 100 calls/day).

2. Make sure the system `/usr/bin/python3` has the deps each step needs:

   ```sh
   # Step 1 (psa_daily_ingest):
   /usr/bin/python3 -m pip install --user 'httpx>=0.27.0'
   # Steps 2 + 3 (embed_references, accumulate_psa_exemplars) — they
   # load images and call the identification embedder:
   /usr/bin/python3 -m pip install --user 'numpy>=2.1' 'opencv-python-headless>=4.10' 'pillow>=11.0'
   ```

   (The launchd job uses `/usr/bin/python3` rather than a venv. If you'd prefer a venv, point `daily_cycle.sh`'s `PYTHON` env var at the venv's interpreter — no plist edit required.)

### Install the launchd job

```sh
cp ml/scripts/com.projectgrader.psa-ingest.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.projectgrader.psa-ingest.plist
```

The plist is configured to fire daily at **09:00 local**. `RunAtLoad` is `false` — the first run should be deliberate so you can watch it:

```sh
launchctl start com.projectgrader.psa-ingest
tail -f ~/psa_data/launchd.stdout.log
```

### Monitoring

```sh
# What did the daily cycle do?  Per-step rc lines come from daily_cycle.sh.
tail -80 ~/psa_data/launchd.stdout.log
tail -80 ~/psa_data/launchd.stderr.log

# Step 1 — what got ingested today?
tail -1 ~/psa_data/ingest_log.jsonl | python3 -m json.tool
# Where will tomorrow's run pick up?
cat ~/psa_data/state.json
# How big is the PSA corpus?
wc -l ~/psa_data/scraped.jsonl
find ~/psa_data/images -name '*.jpg' | wc -l

# Step 2 / 3 — reference embeddings store.
ls -lh ~/manufacturer_refs/reference_embeddings.npz
# How many cert exemplars has step 3 accumulated so far?
wc -l ~/manufacturer_refs/psa_exemplars_log.jsonl
```

### Configuration knobs (env vars)

| Var | Default | Purpose |
|---|---|---|
| `PSA_PUBLIC_API_TOKEN` | required | bearer token from psacard.com/publicapi |
| `PSA_INGEST_DATA_DIR` | `~/psa_data` | where state.json, ingest_log.jsonl, scraped.jsonl, and images/ live |
| `PSA_INGEST_START_CERT_ID` | `80000010` | cert ID to start from on the FIRST run only — ignored once `state.json` exists |
| `PSA_INGEST_MAX_PER_RUN` | `200` | upper bound on certs attempted per run; budget exits sooner on the free tier |
| `PSA_INGEST_DAILY_CAP` | `100` | client-side daily call ceiling (raise on paid tier) |

### Exit codes

| Code | Meaning | Action |
|---|---|---|
| 0 | normal end-of-run (budget exhausted or max_per_run reached) | none — next run will pick up |
| 1 | unhandled exception | inspect stderr log |
| 2 | missing `PSA_PUBLIC_API_TOKEN` | check .env / launchd env passthrough |
| 3 | auth error from PSA (401 / 403) | regenerate token; state was NOT advanced |

### Pause / resume / reset

```sh
# Pause without uninstalling
launchctl unload ~/Library/LaunchAgents/com.projectgrader.psa-ingest.plist

# Resume
launchctl load ~/Library/LaunchAgents/com.projectgrader.psa-ingest.plist

# Restart from a different cert ID (rare)
rm ~/psa_data/state.json
PSA_INGEST_START_CERT_ID=90000000 launchctl start com.projectgrader.psa-ingest

# Nuke and start over (DESTROYS COLLECTED DATA)
launchctl unload ~/Library/LaunchAgents/com.projectgrader.psa-ingest.plist
rm -rf ~/psa_data
launchctl load ~/Library/LaunchAgents/com.projectgrader.psa-ingest.plist
```

## `psa_one_cert_smoke.py` — stdlib-only one-cert validation

Validates a single cert against the API without needing httpx. Useful for sanity checks. See the script's docstring for usage.
