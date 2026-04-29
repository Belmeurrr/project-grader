# ml/scripts

Operational scripts. Not imported by application code; meant to be invoked from the command line or under launchd / cron.

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

2. Make sure `httpx` is installed for the system `/usr/bin/python3`:

   ```sh
   /usr/bin/python3 -m pip install --user 'httpx>=0.27.0'
   ```

   (The launchd job uses `/usr/bin/python3` rather than a venv. If you'd prefer a venv, edit the plist's `ProgramArguments` to point at it.)

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
# What got ingested today?
tail -1 ~/psa_data/ingest_log.jsonl | python3 -m json.tool

# Where will tomorrow's run pick up?
cat ~/psa_data/state.json

# How big is the corpus?
wc -l ~/psa_data/scraped.jsonl
find ~/psa_data/images -name '*.jpg' | wc -l

# Did the last launchd fire succeed?
tail -50 ~/psa_data/launchd.stdout.log
tail -50 ~/psa_data/launchd.stderr.log
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
