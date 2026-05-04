#!/usr/bin/env bash
# Daily data-flywheel cycle for project-grader.
#
# Sequence (run by launchd via com.projectgrader.psa-ingest.plist):
#
#   1. psa_daily_ingest         — drains the PSA Public API daily budget
#                                 into ~/psa_data/scraped.jsonl + images/
#   2. embed_references         — embeds any newly-added manufacturer
#                                 references (Scryfall/PokemonTCG.io) into
#                                 ~/manufacturer_refs/reference_embeddings.npz.
#                                 Idempotent by (manufacturer, variant_id);
#                                 a no-op on days with no new ingest.
#   3. accumulate_psa_exemplars — walks today's PSA records, identifies each
#                                 against the catalog, and appends matched
#                                 submissions' embeddings to the same npz
#                                 under their (manufacturer, variant_id) key.
#                                 Idempotent by cert_id (sidecar log).
#                                 Bumps n_references per variant → raises
#                                 the embedding-anomaly detector's confidence
#                                 ramp the next day.
#   4. sync_catalog_db          — upserts the JSONL + npz into the SQL
#                                 card_sets / card_variants tables that the
#                                 production identification path queries.
#                                 Until this lands, the catalog is empty
#                                 outside test fixtures and identification
#                                 returns no matches. Idempotent via
#                                 INSERT ... ON CONFLICT DO UPDATE.
#                                 Skipped when DATABASE_URL is unset (e.g.
#                                 dev boxes that don't have the API up).
#
# Each step runs independently; we do NOT bail on the first non-zero exit
# because steps 2-4 still have value when step 1 ran out of budget early.
# Exit codes are aggregated: the wrapper returns the first non-zero rc it
# saw, or 0 if all four succeeded.
#
# Configuration:
#   PROJECT_ROOT             default: /Users/danielbelmeur/Github/project-grader
#                            (override if you forked / relocated the repo)
#   PYTHON                   default: /usr/bin/python3
#   PSA_INGEST_DATA_DIR      default: $HOME/psa_data
#   MANUFACTURER_REFS_DATA_DIR  default: $HOME/manufacturer_refs
#
# Smoke-test invocation (bypasses launchd):
#   PROJECT_ROOT="$PWD" ml/scripts/daily_cycle.sh

set -u  # NOT -e: we explicitly want to keep going across step failures.

PROJECT_ROOT="${PROJECT_ROOT:-/Users/danielbelmeur/Github/project-grader}"
PYTHON="${PYTHON:-/usr/bin/python3}"
PSA_DATA_DIR="${PSA_INGEST_DATA_DIR:-$HOME/psa_data}"
REFS_DATA_DIR="${MANUFACTURER_REFS_DATA_DIR:-$HOME/manufacturer_refs}"

cd "$PROJECT_ROOT" || {
  echo "[daily_cycle] cannot cd to PROJECT_ROOT=$PROJECT_ROOT" >&2
  exit 1
}

# Source .env so PSA_PUBLIC_API_TOKEN, GRADER_EMBEDDER, etc. are exported
# to the child Python processes the same way the previous plist one-liner
# did. Missing .env is tolerated for environments that pass env vars in
# directly (CI, local testing).
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
overall=0
record_rc() {
  local rc="$1"
  if [ "$rc" -ne 0 ] && [ "$overall" -eq 0 ]; then
    overall="$rc"
  fi
}

echo "[$(ts)] daily_cycle start (project_root=$PROJECT_ROOT)"

echo "[$(ts)] step 1/4: psa_daily_ingest"
"$PYTHON" ml/scripts/psa_daily_ingest.py
rc=$?
echo "[$(ts)] step 1/4 done (rc=$rc)"
record_rc "$rc"

echo "[$(ts)] step 2/4: embed_references --data-dir $REFS_DATA_DIR"
"$PYTHON" ml/scripts/embed_references.py --data-dir "$REFS_DATA_DIR"
rc=$?
echo "[$(ts)] step 2/4 done (rc=$rc)"
record_rc "$rc"

echo "[$(ts)] step 3/4: accumulate_psa_exemplars"
"$PYTHON" ml/scripts/accumulate_psa_exemplars.py \
  --psa-data-dir "$PSA_DATA_DIR" \
  --refs-data-dir "$REFS_DATA_DIR"
rc=$?
echo "[$(ts)] step 3/4 done (rc=$rc)"
record_rc "$rc"

# Step 4 — sync the on-disk catalog into Postgres so the production
# identification path stops querying an empty table. Requires
# DATABASE_URL in the env (sourced from .env above for launchd, or
# exported directly in CI). When unset we skip rather than fail —
# dev boxes routinely run the flywheel without the API stack up.
echo "[$(ts)] step 4/4: sync_catalog_db --refs-data-dir $REFS_DATA_DIR"
if [ -n "${DATABASE_URL:-}" ]; then
  "$PYTHON" ml/scripts/sync_catalog_db.py \
    --refs-data-dir "$REFS_DATA_DIR" \
    --database-url "$DATABASE_URL"
  rc=$?
else
  echo "[$(ts)] DATABASE_URL not set — skipping (rc=0)"
  rc=0
fi
echo "[$(ts)] step 4/4 done (rc=$rc)"
record_rc "$rc"

echo "[$(ts)] daily_cycle complete (overall_rc=$overall)"
exit "$overall"
