"""Calendar-time daily PSA Public API ingest driver.

Runs once per day under launchd (see com.projectgrader.psa-ingest.plist).
Each run:
  1. Loads the next-cert-id from `<data_dir>/state.json` (created on first
     run from $PSA_INGEST_START_CERT_ID, default 80000010).
  2. Iterates `ingest_cert` on consecutive cert IDs until the daily API
     budget is exhausted or `MAX_CERTS_PER_RUN` is hit.
  3. Persists the new next-cert-id and a stats row to
     `<data_dir>/ingest_log.jsonl`.

The daily 100-call free-tier ceiling means each run will attempt
~50–100 certs before exhausting budget (each gradeable cert costs 2
calls; non-gradeable certs cost 1 due to the early-skip). One day of
ingestion produces, in the average case, ~30–50 successful triples plus
some not_found / non_gradeable / images_missing accumulating in the same
JSONL store as the existing scraper format.

Resume protocol: the state file IS the resume cursor. Crash-safe — if
the script is killed mid-run, the state file's next-cert-id reflects
the last cert we successfully past. The same store-level idempotency
(`has_record`) means a re-run within the same day is a no-op for the
already-ingested certs.

Failure modes:
  - Missing PSA_PUBLIC_API_TOKEN              → exit 2 (config)
  - IngestAborted (auth error from the API)   → exit 3, state unchanged
  - Any other unhandled exception             → exit 1, state unchanged
  - BudgetExhausted (normal end-of-run)       → exit 0, state advanced

Configuration via env (or .env loaded by the wrapper):
  PSA_PUBLIC_API_TOKEN     required
  PSA_INGEST_DATA_DIR      default: ~/psa_data
  PSA_INGEST_START_CERT_ID default: 80000010 (used on first run only)
  PSA_INGEST_MAX_PER_RUN   default: 200      (upper bound; budget exits sooner)
  PSA_INGEST_DAILY_CAP     default: 100      (free tier; raise on paid tier)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Path tweak: this script is expected to run with the project's `ml/`
# dir on sys.path (set by the launchd wrapper via cd + PYTHONPATH).
# Defensive: if not, add it ourselves.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from data.ingestion import (  # noqa: E402
    BudgetExhausted,
    DailyBudget,
    IngestAborted,
    IngestStats,
    LocalScrapedRecordStore,
    ingest_cert,
)


_logger = logging.getLogger("psa_daily_ingest")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_state(state_path: Path, default_start: int) -> dict:
    if not state_path.exists():
        return {"next_cert_id": default_start, "created_at": _now_iso(), "runs": 0}
    with state_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, state_path)


def _append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        _logger.warning("env %s=%r not an int; using default %d", name, raw, default)
        return default


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    token = os.environ.get("PSA_PUBLIC_API_TOKEN")
    if not token:
        _logger.error("PSA_PUBLIC_API_TOKEN not set; aborting")
        return 2

    data_dir = Path(os.environ.get("PSA_INGEST_DATA_DIR") or (Path.home() / "psa_data"))
    start_cert_id = _env_int("PSA_INGEST_START_CERT_ID", 80_000_010)
    max_per_run = _env_int("PSA_INGEST_MAX_PER_RUN", 200)
    daily_cap = _env_int("PSA_INGEST_DAILY_CAP", 100)

    state_path = data_dir / "state.json"
    log_path = data_dir / "ingest_log.jsonl"

    state = _read_state(state_path, default_start=start_cert_id)
    starting_cert_id = int(state["next_cert_id"])
    _logger.info(
        "starting run data_dir=%s next_cert_id=%d daily_cap=%d max_per_run=%d",
        data_dir,
        starting_cert_id,
        daily_cap,
        max_per_run,
    )

    store = LocalScrapedRecordStore(data_dir)
    budget = DailyBudget(daily_cap=daily_cap)
    stats = IngestStats()

    cert_id = starting_cert_id
    end_cert_id = starting_cert_id + max_per_run
    stop_reason = "max_per_run_reached"

    try:
        with httpx.Client() as client:
            while cert_id < end_cert_id:
                try:
                    ingest_cert(
                        cert_id,
                        token=token,
                        client=client,
                        store=store,
                        budget=budget,
                        stats=stats,
                    )
                except BudgetExhausted:
                    stop_reason = "budget_exhausted"
                    break
                cert_id += 1
            else:
                # range completed without budget exhaustion — unusual on the
                # free tier but possible if max_per_run is small or many
                # certs hit not_found.
                stop_reason = "max_per_run_reached"
    except IngestAborted as e:
        _logger.error("aborting run on auth error: %s", e)
        # Don't advance state — auth needs human attention
        _append_log(
            log_path,
            {
                "timestamp": _now_iso(),
                "starting_cert_id": starting_cert_id,
                "ending_cert_id": cert_id,
                "stop_reason": "ingest_aborted",
                "error": str(e),
                "stats": asdict(stats),
            },
        )
        return 3

    # cert_id is the next un-attempted cert. Persist that as the new cursor.
    state["next_cert_id"] = cert_id
    state["last_run_at"] = _now_iso()
    state["runs"] = int(state.get("runs", 0)) + 1
    _write_state(state_path, state)

    _append_log(
        log_path,
        {
            "timestamp": _now_iso(),
            "starting_cert_id": starting_cert_id,
            "ending_cert_id": cert_id,
            "stop_reason": stop_reason,
            "stats": asdict(stats),
        },
    )

    _logger.info(
        "run done stop_reason=%s starting=%d ending=%d stats=%s",
        stop_reason,
        starting_cert_id,
        cert_id,
        asdict(stats),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
