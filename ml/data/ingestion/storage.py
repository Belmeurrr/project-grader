"""Storage backends for scraped pop-report records.

Why a Protocol-based abstraction:
    The scraper itself doesn't care whether records land on local disk (dev
    laptop, CI fixtures) or in S3 (long-running scrape against the real
    cert space). Both backends share the same shape — append a JSONL row,
    write image bytes under a deterministic key — but the implementations
    are mechanically different. Splitting via Protocol means the scraper
    has zero `if local:` branches, and a future backend (GCS, MinIO over
    a different endpoint, an Iceberg table) plugs in without touching
    scraper code.

Why JSONL for records:
    Append-only, line-oriented, crash-safe (a partial line at the end of a
    file is the only failure mode and it's trivially detectable when re-
    reading). Pandas / DuckDB / DVC can all consume it. Resumption is just
    "scan the file once, build a set of cert_ids already done."

Why we don't write a manifest of consumed cert_ids:
    The JSONL itself IS the manifest. Single source of truth. A separate
    manifest invites drift between "we wrote a record" and "we wrote a
    manifest entry."

Image key layout (mirrors apps/api/grader/services/storage.py style):
    <prefix>/images/<cert_id>/<kind>.<ext>
    where kind in {"front", "back"} and ext is derived from content-type.

Resume protocol:
    `iter_existing_cert_ids()` returns the set of cert_ids already
    persisted. The scraper unions this with the requested range and skips
    those certs. `has_record(cert_id)` is the cheap per-cert check used
    inside the scrape loop.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from data.ingestion.psa_pop_scraper import ScrapedRecord


_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


def _ext_from_content_type(content_type: str) -> str:
    """Map an HTTP Content-Type to a file extension.

    Falls back to "bin" rather than raising — an unknown image type is not
    a reason to abort an entire scrape run."""
    if not content_type:
        return "bin"
    primary = content_type.split(";", 1)[0].strip().lower()
    return _CONTENT_TYPE_TO_EXT.get(primary, "bin")


def image_relative_key(cert_id: int, kind: str, content_type: str) -> str:
    """Return the relative storage path for an image of a given cert.

    Pure helper so callers can format URIs without instantiating a store."""
    ext = _ext_from_content_type(content_type)
    return f"images/{cert_id}/{kind}.{ext}"


@runtime_checkable
class ScrapedRecordStore(Protocol):
    """Storage backend Protocol for scraped pop-report records.

    Implementations MUST be safe for single-process appends. Multi-process
    coordination is out of scope at v1 — run one scraper instance per
    backend. (If we later need parallelism, the obvious split is by cert_id
    range, with each shard writing its own JSONL file.)"""

    def write_record(self, record: "ScrapedRecord") -> None:
        """Append `record` durably. Must be atomic per-record."""

    def write_image(
        self,
        cert_id: int,
        kind: str,
        content_type: str,
        data: bytes,
    ) -> str:
        """Persist image bytes; return the storage URI/key written."""

    def has_record(self, cert_id: int) -> bool:
        """Cheap idempotency check used by the scrape loop."""

    def iter_existing_cert_ids(self) -> set[int]:
        """One-shot scan of all already-persisted cert_ids. Used for resume."""


class LocalScrapedRecordStore:
    """Filesystem-backed store. Default for dev and CI fixtures.

    Layout under `root`:
        scraped.jsonl                  one ScrapedRecord per line
        images/<cert_id>/<kind>.<ext>  image bytes

    `has_record` is implemented by maintaining an in-memory set populated
    lazily on first call from the JSONL. We refresh the set whenever
    `write_record` runs so subsequent `has_record` calls reflect the
    current process's writes.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self._records_path = self.root / "scraped.jsonl"
        self._lock = threading.Lock()
        self._cert_ids_cache: set[int] | None = None

    @property
    def records_path(self) -> Path:
        return self._records_path

    def _ensure_cache(self) -> set[int]:
        if self._cert_ids_cache is None:
            self._cert_ids_cache = self._scan_cert_ids()
        return self._cert_ids_cache

    def _scan_cert_ids(self) -> set[int]:
        ids: set[int] = set()
        if not self._records_path.exists():
            return ids
        with self._records_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # tolerate a half-written trailing line from a crash
                    continue
                cert_id = obj.get("cert_id")
                if isinstance(cert_id, int):
                    ids.add(cert_id)
        return ids

    def write_record(self, record: "ScrapedRecord") -> None:
        payload = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with self._records_path.open("a", encoding="utf-8") as fh:
                fh.write(payload + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            cache = self._ensure_cache()
            cache.add(record.cert_id)

    def write_image(
        self,
        cert_id: int,
        kind: str,
        content_type: str,
        data: bytes,
    ) -> str:
        rel = image_relative_key(cert_id, kind, content_type)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename for atomicity vs partial writes during crash
        tmp = target.with_suffix(target.suffix + ".part")
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return str(target)

    def has_record(self, cert_id: int) -> bool:
        return cert_id in self._ensure_cache()

    def iter_existing_cert_ids(self) -> set[int]:
        # Force a fresh scan; resume is a once-per-run operation so the
        # cost is acceptable, and this guarantees we see records written
        # by a previous process.
        self._cert_ids_cache = self._scan_cert_ids()
        return set(self._cert_ids_cache)


class S3ScrapedRecordStore:
    """S3 / MinIO-backed store with the same shape as the local store.

    Records are appended via a read-modify-write of a single
    `<prefix>/scraped.jsonl` object. That is NOT a high-throughput pattern
    — but pop-report scraping is rate-limited to ~1 req/sec by design, so
    the bottleneck is upstream, not S3. If we ever shard across multiple
    workers, switch to one JSONL per shard and merge offline.

    `boto3` is imported lazily so tests / dev environments without it can
    still import this module. Tests inject a mock client directly via the
    `s3_client` arg.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "psa_pop",
        s3_client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = s3_client
        self._lock = threading.Lock()
        self._cert_ids_cache: set[int] | None = None

    @property
    def records_key(self) -> str:
        return f"{self.prefix}/scraped.jsonl"

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # lazy: the dep is optional unless S3 is used

            self._client = boto3.client("s3")
        return self._client

    def _read_existing_records_blob(self) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self.records_key)
        except Exception as e:  # botocore.ClientError, but we don't import it
            # Treat any "not found" as empty; surface other errors to caller.
            msg = str(e)
            if "NoSuchKey" in msg or "404" in msg or "Not Found" in msg:
                return b""
            err = getattr(e, "response", None)
            if isinstance(err, dict):
                code = err.get("Error", {}).get("Code", "")
                if code in {"NoSuchKey", "404", "NotFound"}:
                    return b""
            raise
        body = obj["Body"]
        return body.read()

    def _scan_cert_ids(self) -> set[int]:
        blob = self._read_existing_records_blob()
        if not blob:
            return set()
        ids: set[int] = set()
        for line in blob.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cert_id = obj.get("cert_id")
            if isinstance(cert_id, int):
                ids.add(cert_id)
        return ids

    def _ensure_cache(self) -> set[int]:
        if self._cert_ids_cache is None:
            self._cert_ids_cache = self._scan_cert_ids()
        return self._cert_ids_cache

    def write_record(self, record: "ScrapedRecord") -> None:
        payload = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        with self._lock:
            existing = self._read_existing_records_blob()
            new_blob = existing + payload.encode("utf-8")
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.records_key,
                Body=new_blob,
                ContentType="application/x-ndjson",
            )
            cache = self._ensure_cache()
            cache.add(record.cert_id)

    def write_image(
        self,
        cert_id: int,
        kind: str,
        content_type: str,
        data: bytes,
    ) -> str:
        rel = image_relative_key(cert_id, kind, content_type)
        key = f"{self.prefix}/{rel}"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type or "application/octet-stream",
        )
        return f"s3://{self.bucket}/{key}"

    def has_record(self, cert_id: int) -> bool:
        return cert_id in self._ensure_cache()

    def iter_existing_cert_ids(self) -> set[int]:
        # Re-scan the JSONL; cheaper and authoritative compared to listing
        # image prefixes (an image without a record is not a "done" cert).
        self._cert_ids_cache = self._scan_cert_ids()
        return set(self._cert_ids_cache)


__all__ = [
    "LocalScrapedRecordStore",
    "S3ScrapedRecordStore",
    "ScrapedRecordStore",
    "image_relative_key",
]
