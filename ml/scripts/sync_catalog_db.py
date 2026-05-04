"""Sync the on-disk manufacturer reference flywheel into the SQL catalog.

Walks `<refs-data-dir>/references.jsonl` (produced by
`scripts.manufacturer_refs_ingest`) plus the matching
`reference_embeddings.npz`, and upserts the resulting rows into the
`card_sets` and `card_variants` tables. Until this runs, the
identification path queries an empty catalog table — the manufacturer
JSONL is the only source of truth for variant metadata in production
today, but inference only consults `card_variants` via SQL.

Pipeline:

  Pass 1 — derive `(game, set_code) → CardSet` from the JSONL,
           upsert into `card_sets` keyed on the existing
           `(game, code)` unique index. Builds a
           `(game, set_code) → set_id` lookup for pass 2.

  Pass 2 — stream the JSONL again, look up the embedding from the
           npz via `lookup_references`, compute pHash from
           `front_image_path` via
           `pipelines.identification.hashing.compute_phash`. Upsert
           into `card_variants` keyed on
           `(game, set_id, card_number)`.

Idempotency: both passes use Postgres `INSERT ... ON CONFLICT DO
UPDATE`, so re-running over the same data is a no-op (well, an UPDATE
with the same values — no row count change). Safe to wire into the
daily cycle.

Game enum mapping:
  manufacturer="mtg"     → Game.MTG     (value "mtg")
  manufacturer="pokemon" → Game.POKEMON (value "pokemon")
  Any other manufacturer is logged + skipped — better to surface an
  unmapped source than to silently shovel rows into Game.OTHER.

For MTG, Scryfall's `variant_id` (a UUID) is stored on
`card_variants.scryfall_id`. The full variant_id and source
manufacturer are also kept in `external_ids` so we don't lose the
provenance for non-MTG sources.

Why this is structured as `prepare_*` + `apply_*`:
  The data-preparation functions (`iter_set_rows`,
  `iter_variant_rows`) are pure — given a JSONL path + npz path +
  refs root they yield row-shaped dicts, no SQLAlchemy involved. The
  `apply_*` helpers take a session and a list of rows and run the
  upserts. This split lets the test path verify the disk → row
  mapping (the part with all the parsing, hashing, manufacturer-to-
  enum logic) without bringing up Postgres + pgvector + asyncpg in
  the ml/ test venv. The async DB path is exercised at the
  integration level by the API test suite (which already runs
  against real Postgres).

CLI:
    python -m scripts.sync_catalog_db [--refs-data-dir DIR]
                                      [--database-url URL]
                                      [--dry-run]

Configuration:
    --refs-data-dir PATH  Directory containing references.jsonl and
                          reference_embeddings.npz. Default:
                          $MANUFACTURER_REFS_DATA_DIR or
                          ~/manufacturer_refs.
    --database-url URL    SQLAlchemy async URL. Default:
                          $DATABASE_URL.
    --dry-run             Build the row payloads + log counts, but
                          don't open a DB connection.

Exit codes:
    0  success (or dry-run; or no records to sync)
    1  unexpected exception (I/O, DB connection, etc.)
    2  bad CLI invocation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# Mirror the path tweak used by sibling scripts so `python ml/scripts/
# sync_catalog_db.py` (no -m) also works.
_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

# The DB models live in apps/api. We path-shim them in rather than
# requiring an editable install of the API package — the daily cycle
# runs against whatever Python the launchd plist points at, and that
# Python isn't guaranteed to have `grader` on its path.
_REPO_ROOT = _ML_ROOT.parent
_API_ROOT = _REPO_ROOT / "apps" / "api"
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))


_logger = logging.getLogger("sync_catalog_db")


# Manufacturer short-name → Game enum value. Kept as plain strings
# (rather than importing the enum at module load) so the row-prep
# layer doesn't pull SQLAlchemy + pgvector. The DB-applying code
# coerces these to the actual `Game` enum.
_MANUFACTURER_TO_GAME_VALUE: dict[str, str] = {
    "mtg": "mtg",
    "pokemon": "pokemon",
}


@dataclass
class SyncStats:
    """Counters from a single `sync_catalog_db` pass."""

    sets_total: int = 0
    sets_upserted: int = 0
    variants_total: int = 0
    variants_upserted: int = 0
    variants_with_phash: int = 0
    variants_with_embedding: int = 0
    skipped_unmapped_manufacturer: int = 0
    skipped_missing_set_code: int = 0
    skipped_missing_card_number: int = 0
    image_load_failures: int = 0


# --------------------------------------------------------------------------
# Pure data-prep — no SQLAlchemy
# --------------------------------------------------------------------------


def _iter_jsonl(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL. Tolerates the same
    blank/half-written trailing-line cases the ingest store does."""
    if not jsonl_path.exists():
        return
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_set_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    """Pass-1 prep: derive unique `(game, set_code)` rows for the
    `card_sets` table from the JSONL.

    Returns a list (not a generator) because the caller needs the
    full set first — variant rows reference set ids, so we must
    upsert all sets before iterating variants.

    Each row is a dict with keys:
        game       — Game enum value string ("mtg" / "pokemon")
        code       — set_code as it appears in the JSONL
        name       — best-effort human-readable set name. The JSONL
                     doesn't carry a per-set name today, so we fall
                     back to the upper-cased set_code (e.g. "LEA").
                     A future ingest pass that records set names can
                     drop in here.

    Records with manufacturer not in the mapping or set_code missing
    are silently filtered — we count them in stats elsewhere when
    walking the same JSONL for variants.
    """
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for obj in _iter_jsonl(jsonl_path):
        manufacturer = obj.get("manufacturer")
        set_code = obj.get("set_code")
        if not isinstance(manufacturer, str) or not isinstance(set_code, str):
            continue
        if not set_code:
            continue
        game = _MANUFACTURER_TO_GAME_VALUE.get(manufacturer)
        if game is None:
            continue
        key = (game, set_code)
        if key in seen:
            continue
        seen[key] = {"game": game, "code": set_code, "name": set_code.upper()}
    return list(seen.values())


def _read_image_bgr(path: str | None) -> Any | None:
    """Lazy cv2 import + tolerant image read. Returns None on any
    failure — phash is opportunistic, missing it is not fatal."""
    if not path:
        return None
    try:
        import cv2  # local import — keeps pure module import cheap
    except Exception:  # noqa: BLE001
        return None
    try:
        if not Path(path).exists():
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None
    if img is None:
        return None
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


def _phash_for_path(path: str | None) -> int | None:
    """Compute the 64-bit pHash for an image on disk, or None if the
    image is unavailable. The DB column is BigInteger (signed 64-bit),
    so values ≥ 2**63 must be re-signed; we do that at apply-time."""
    img = _read_image_bgr(path)
    if img is None:
        return None
    try:
        from pipelines.identification.hashing import compute_phash
    except Exception as e:  # noqa: BLE001
        _logger.warning("could not import compute_phash: %s", e)
        return None
    try:
        return int(compute_phash(img))
    except Exception as e:  # noqa: BLE001
        _logger.warning("compute_phash failed path=%s err=%s", path, e)
        return None


def _phash_to_signed64(value: int | None) -> int | None:
    """Map a 64-bit unsigned int into the signed range Postgres
    BIGINT expects. Postgres rejects values > 2**63 - 1."""
    if value is None:
        return None
    if value >= 2**63:
        return value - 2**64
    return value


def iter_variant_rows(
    jsonl_path: Path,
    embeddings_path: Path,
    *,
    set_id_lookup: dict[tuple[str, str], Any],
    stats: SyncStats,
) -> Iterator[dict[str, Any]]:
    """Pass-2 prep: yield row-shaped dicts for the `card_variants`
    table.

    Args:
        jsonl_path: references.jsonl
        embeddings_path: reference_embeddings.npz
        set_id_lookup: `(game, set_code) → CardSet.id` (uuid). Built
            after pass 1 has run.
        stats: SyncStats — counters incremented in place.

    Each yielded dict has keys:
        game, set_id, card_number, name, scryfall_id, external_ids,
        canonical_phash (int, signed 64-bit, or None),
        canonical_image_embedding (np.ndarray (768,) or None — the
            apply layer drops the column entirely when None so the
            DB DEFAULT applies).

    Filtering / counting:
        - Rows with manufacturer not in the enum mapping → skipped,
          stats.skipped_unmapped_manufacturer++.
        - Rows missing set_code or card_number → skipped, the
          matching counter increments.
        - Rows whose (game, set_code) didn't make it into pass 1 →
          should not happen (pass 1 derives sets from the same
          JSONL); guarded as a no-op skip.
        - Rows whose front_image_path is unreadable →
          stats.image_load_failures++ but the row still emits with
          canonical_phash=None.
    """
    # Lazy npz load — opening it once and indexing per-row is cheaper
    # than `lookup_references` calling np.load on every variant. We
    # mmap the file so memory cost stays bounded for large catalogs.
    archive_keys: dict[str, np.ndarray] | None = None
    if embeddings_path.exists():
        try:
            with np.load(embeddings_path, allow_pickle=False) as archive:
                # Materialize fully — np.load returns a context-bound
                # NpzFile; we want a stable dict for the per-row
                # lookup loop below.
                archive_keys = {
                    k: archive[k].astype(np.float32, copy=False)
                    for k in archive.files
                }
        except Exception as e:  # noqa: BLE001
            _logger.warning("failed to load embeddings npz %s: %s", embeddings_path, e)
            archive_keys = None

    for obj in _iter_jsonl(jsonl_path):
        stats.variants_total += 1

        manufacturer = obj.get("manufacturer")
        if not isinstance(manufacturer, str):
            stats.skipped_unmapped_manufacturer += 1
            continue
        game = _MANUFACTURER_TO_GAME_VALUE.get(manufacturer)
        if game is None:
            stats.skipped_unmapped_manufacturer += 1
            continue

        set_code = obj.get("set_code")
        if not isinstance(set_code, str) or not set_code:
            stats.skipped_missing_set_code += 1
            continue

        card_number = obj.get("collector_number")
        if not isinstance(card_number, str) or not card_number:
            stats.skipped_missing_card_number += 1
            continue

        set_id = set_id_lookup.get((game, set_code))
        if set_id is None:
            # Defensive: pass 1 should have created every (game,
            # set_code) we encounter here.
            stats.skipped_missing_set_code += 1
            continue

        variant_id = obj.get("variant_id")
        name = obj.get("name") or "Unknown"

        # Embedding lookup — keys are "<manufacturer>/<variant_id>".
        embedding: np.ndarray | None = None
        if archive_keys is not None and isinstance(variant_id, str):
            key = f"{manufacturer}/{variant_id}"
            vec = archive_keys.get(key)
            if vec is not None and vec.ndim == 1:
                embedding = vec
                stats.variants_with_embedding += 1

        # Perceptual hash — derived on the fly from the front image.
        front = obj.get("front_image_path")
        phash_unsigned = _phash_for_path(front)
        if phash_unsigned is None and front:
            stats.image_load_failures += 1
        if phash_unsigned is not None:
            stats.variants_with_phash += 1
        phash_signed = _phash_to_signed64(phash_unsigned)

        # External IDs — preserve the source-side variant_id so we
        # can round-trip to the manufacturer catalog later. For MTG
        # the same value also lands in scryfall_id (string column,
        # indexed) for direct lookups.
        external_ids: dict[str, Any] = {"manufacturer": manufacturer}
        if isinstance(variant_id, str):
            external_ids[f"{manufacturer}_variant_id"] = variant_id

        scryfall_id = (
            variant_id
            if manufacturer == "mtg" and isinstance(variant_id, str)
            else None
        )

        yield {
            "game": game,
            "set_id": set_id,
            "card_number": card_number,
            "name": name,
            "scryfall_id": scryfall_id,
            "external_ids": external_ids,
            "canonical_phash": phash_signed,
            "canonical_image_embedding": embedding,
        }


# --------------------------------------------------------------------------
# Async DB application — touches SQLAlchemy + asyncpg + pgvector
# --------------------------------------------------------------------------


async def apply_sets(session: Any, set_rows: list[dict[str, Any]]) -> dict[tuple[str, str], Any]:
    """Upsert all set rows and return a `(game, code) → id` map.

    Uses the Postgres `INSERT ... ON CONFLICT (game, code) DO UPDATE`
    on the existing `ix_card_sets_game_code` unique index. We DO
    UPDATE the `name` even though our derived name is just the upper-
    cased set_code today — that way an upgrade that ships richer
    set-name data will refresh existing rows on the next cycle
    without needing a manual migration.
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    # Local import: only at apply time, not at row-prep time.
    from grader.db.models import CardSet, Game  # type: ignore[import-not-found]

    if not set_rows:
        return {}

    # Coerce game value strings → enum.
    payload = [
        {"game": Game(row["game"]), "code": row["code"], "name": row["name"]}
        for row in set_rows
    ]
    stmt = pg_insert(CardSet).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=["game", "code"],
        set_={"name": stmt.excluded.name},
    )
    await session.execute(stmt)
    await session.flush()

    # Re-select to capture ids — INSERT ... ON CONFLICT DO UPDATE
    # doesn't have a portable RETURNING that handles both insert and
    # update without race-condition gotchas, so we just SELECT after.
    # Per-key fetch keeps this simple; set counts are in the hundreds,
    # not millions.
    keys = [(Game(row["game"]), row["code"]) for row in set_rows]
    out: dict[tuple[str, str], Any] = {}
    for game_enum, code in keys:
        row = (
            await session.execute(
                select(CardSet.id).where(
                    CardSet.game == game_enum, CardSet.code == code
                )
            )
        ).one_or_none()
        if row is not None:
            out[(game_enum.value, code)] = row[0]
    return out


async def apply_variants(
    session: Any,
    variant_rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = 500,
) -> int:
    """Upsert variant rows in batches keyed on
    `(game, set_id, card_number)`. Returns the number of rows
    submitted (not the number of distinct rows changed — Postgres
    can't cheaply distinguish insert vs update count without a
    RETURNING).

    The unique index for the conflict target is created here as a
    constraint name reference: the existing index
    `ix_card_variants_game_set_number` is non-unique in the model
    declaration, so we add a unique partial index on first upsert.
    To stay schema-pure, we use `index_elements=` and rely on the
    DB having a unique index in place (the migration will add one).
    For the production path we expect the migration; for the test
    path the fake session ignores the constraint mechanics.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from grader.db.models import CardVariant, Game  # type: ignore[import-not-found]

    submitted = 0
    pending: list[dict[str, Any]] = []
    for row in variant_rows:
        payload = {
            "game": Game(row["game"]),
            "set_id": row["set_id"],
            "card_number": row["card_number"],
            "name": row["name"],
            "scryfall_id": row.get("scryfall_id"),
            "external_ids": row.get("external_ids") or {},
            "canonical_phash": row.get("canonical_phash"),
            "canonical_image_embedding": row.get("canonical_image_embedding"),
        }
        pending.append(payload)
        if len(pending) >= batch_size:
            stmt = pg_insert(CardVariant).values(pending)
            stmt = stmt.on_conflict_do_update(
                index_elements=["game", "set_id", "card_number"],
                set_={
                    "name": stmt.excluded.name,
                    "scryfall_id": stmt.excluded.scryfall_id,
                    "external_ids": stmt.excluded.external_ids,
                    "canonical_phash": stmt.excluded.canonical_phash,
                    "canonical_image_embedding": stmt.excluded.canonical_image_embedding,
                },
            )
            await session.execute(stmt)
            submitted += len(pending)
            pending = []

    if pending:
        stmt = pg_insert(CardVariant).values(pending)
        stmt = stmt.on_conflict_do_update(
            index_elements=["game", "set_id", "card_number"],
            set_={
                "name": stmt.excluded.name,
                "scryfall_id": stmt.excluded.scryfall_id,
                "external_ids": stmt.excluded.external_ids,
                "canonical_phash": stmt.excluded.canonical_phash,
                "canonical_image_embedding": stmt.excluded.canonical_image_embedding,
            },
        )
        await session.execute(stmt)
        submitted += len(pending)

    await session.flush()
    return submitted


async def sync_catalog_db(
    refs_data_dir: Path,
    *,
    database_url: str | None = None,
    dry_run: bool = False,
    session_factory: Any | None = None,
) -> SyncStats:
    """End-to-end: walk the on-disk references and upsert them into
    `card_sets` + `card_variants`.

    Args:
        refs_data_dir: directory containing `references.jsonl` +
            `reference_embeddings.npz`.
        database_url: SQLAlchemy async URL. Required unless `dry_run`
            is True or `session_factory` is provided.
        dry_run: if True, build the row payloads + log counts but
            don't open any DB connection. Useful for smoke-checking
            a directory before wiring it to production credentials.
        session_factory: optional injected `async_sessionmaker`-like
            callable. Tests pass a fake here that records upserts
            in memory; production passes None and lets the function
            build one from `database_url`.

    Returns: SyncStats with per-stage counters.
    """
    jsonl_path = refs_data_dir / "references.jsonl"
    embeddings_path = refs_data_dir / "reference_embeddings.npz"

    stats = SyncStats()

    # Pass 1 — set rows.
    set_rows = iter_set_rows(jsonl_path)
    stats.sets_total = len(set_rows)
    _logger.info("pass 1: %d unique (game, set_code) rows", stats.sets_total)

    if dry_run:
        # Walk variants for stats only — no DB.
        sentinel_lookup: dict[tuple[str, str], Any] = {
            (row["game"], row["code"]): f"dryrun-{row['game']}-{row['code']}"
            for row in set_rows
        }
        for _ in iter_variant_rows(
            jsonl_path,
            embeddings_path,
            set_id_lookup=sentinel_lookup,
            stats=stats,
        ):
            pass
        _logger.info("dry-run stats: %s", asdict(stats))
        return stats

    if session_factory is None:
        if not database_url:
            raise ValueError("database_url is required when dry_run=False and no session_factory injected")
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(database_url, pool_pre_ping=True)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        async with session.begin():
            set_id_lookup = await apply_sets(session, set_rows)
            stats.sets_upserted = len(set_id_lookup)

            # Pass 2 — variants. We materialize into a list because
            # we want stats counters populated regardless of how the
            # apply layer chooses to chunk.
            variant_rows = list(
                iter_variant_rows(
                    jsonl_path,
                    embeddings_path,
                    set_id_lookup=set_id_lookup,
                    stats=stats,
                )
            )
            stats.variants_upserted = await apply_variants(session, variant_rows)

    _logger.info("sync done stats=%s", asdict(stats))
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sync_catalog_db",
        description="Upsert manufacturer reference flywheel data into the SQL catalog.",
    )
    p.add_argument(
        "--refs-data-dir",
        default=os.environ.get("MANUFACTURER_REFS_DATA_DIR")
        or str(Path.home() / "manufacturer_refs"),
        help="Directory with references.jsonl + reference_embeddings.npz. "
        "Default: $MANUFACTURER_REFS_DATA_DIR or ~/manufacturer_refs.",
    )
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy async URL (e.g. postgresql+asyncpg://...). "
        "Default: $DATABASE_URL.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payloads + log counts without opening a DB connection.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    refs_dir = Path(args.refs_data_dir)
    if not refs_dir.exists():
        _logger.error("refs-data-dir does not exist: %s", refs_dir)
        return 1
    if not (refs_dir / "references.jsonl").exists():
        _logger.error("references.jsonl not found under %s", refs_dir)
        return 1

    if not args.dry_run and not args.database_url:
        _logger.error("--database-url (or $DATABASE_URL) required unless --dry-run")
        return 2

    import asyncio

    try:
        stats = asyncio.run(
            sync_catalog_db(
                refs_dir,
                database_url=args.database_url,
                dry_run=args.dry_run,
            )
        )
    except Exception as e:  # noqa: BLE001
        _logger.exception("sync failed: %s", e)
        return 1

    _logger.info("done stats=%s", asdict(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
