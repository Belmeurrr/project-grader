"""One-shot validation runner for the PSA Public API.

Hits the two endpoints we care about (`GetByCertNumber` for metadata,
`GetImagesByCertNumber` for image URLs) for a single cert. Dumps raw
JSON responses to `/tmp/psa_cert_<n>.{metadata,images}.json` and prints
what we extracted using the same logic as the production parsers in
`data.ingestion.psa_public_api`.

Why stdlib-only: the `ml/` side has no venv yet and `httpx` isn't
installed. This runner uses `urllib.request` so it works on any
Python ≥ 3.12 with zero deps.

Usage:
    export PSA_PUBLIC_API_TOKEN=<your-token>
    python3 ml/scripts/psa_one_cert_smoke.py 28493978

A successful run will (a) report HTTP 200 from both endpoints, (b)
write the two JSON files to /tmp, and (c) print the parsed
ScrapedRecord-shaped output. Burns 2 of your 100 daily API calls.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

BASE = "https://api.psacard.com/publicapi"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def fetch(url: str, token: str) -> tuple[int, dict | None, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"bearer {token}",
            "Accept": "application/json",
            "User-Agent": "ProjectGrader-Smoke/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            try:
                payload = json.loads(body)
            except ValueError:
                print(f"  WARN: non-JSON 200 from {url}; body[:200]={body[:200]!r}")
                payload = None
            return resp.status, payload, dict(resp.headers)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, None, {"_error_body": body[:500]}


def is_image_url(value) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return False
    path = (parsed.path or "").lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTS)


def extract_image_urls(payload):
    found: list[tuple[str, str]] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}".lower())
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif is_image_url(node):
            found.append((path, node))

    walk(payload, "")
    front = back = None
    for path, url in found:
        if "front" in path and front is None:
            front = url
        elif "back" in path and back is None:
            back = url
    if front is None and back is None and found:
        front = found[0][1]
        if len(found) >= 2:
            back = found[1][1]
    return front, back, found


def parse_grade(metadata) -> float | None:
    psa = (metadata or {}).get("PSACert") or {}
    raw = psa.get("CardGrade")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    desc = psa.get("GradeDescription") or ""
    m = re.search(r"\b(\d{1,2}(?:\.5)?)\b", str(desc))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: psa_one_cert_smoke.py <cert_number>", file=sys.stderr)
        return 2
    cert_id = sys.argv[1].strip()
    if not cert_id.isdigit():
        print(f"cert_number must be numeric, got {cert_id!r}", file=sys.stderr)
        return 2

    token = os.environ.get("PSA_PUBLIC_API_TOKEN")
    if not token:
        print("PSA_PUBLIC_API_TOKEN not set", file=sys.stderr)
        return 2

    print(f"--- Cert {cert_id} ---")
    print()

    print("[1/2] GetByCertNumber")
    meta_url = f"{BASE}/cert/GetByCertNumber/{cert_id}"
    meta_code, meta_payload, _ = fetch(meta_url, token)
    print(f"  status: {meta_code}")
    if meta_code == 200 and meta_payload:
        path = f"/tmp/psa_cert_{cert_id}.metadata.json"
        with open(path, "w") as fh:
            json.dump(meta_payload, fh, indent=2, ensure_ascii=False)
        print(f"  saved: {path}")
        psa = (meta_payload.get("PSACert") or {})
        grade = parse_grade(meta_payload)
        print(f"  parsed:")
        print(f"    grade        = {grade}")
        print(f"    Year         = {psa.get('Year')!r}")
        print(f"    Brand        = {psa.get('Brand')!r}")
        print(f"    Subject      = {psa.get('Subject')!r}")
        print(f"    CardNumber   = {psa.get('CardNumber')!r}")
        print(f"    GradeDescr.  = {psa.get('GradeDescription')!r}")
    else:
        print(f"  metadata fetch failed; skipping image fetch")
        return 1

    print()
    print("[2/2] GetImagesByCertNumber")
    img_url = f"{BASE}/cert/GetImagesByCertNumber/{cert_id}"
    img_code, img_payload, _ = fetch(img_url, token)
    print(f"  status: {img_code}")
    if img_code == 200 and img_payload is not None:
        path = f"/tmp/psa_cert_{cert_id}.images.json"
        with open(path, "w") as fh:
            json.dump(img_payload, fh, indent=2, ensure_ascii=False)
        print(f"  saved: {path}")
        front, back, all_found = extract_image_urls(img_payload)
        print(f"  parsed:")
        print(f"    front = {front!r}")
        print(f"    back  = {back!r}")
        print(f"    all image-shaped URLs found ({len(all_found)}):")
        for p, u in all_found[:10]:
            print(f"      at {p!r}: {u}")
        if isinstance(img_payload, dict):
            print(f"    top-level keys: {sorted(img_payload.keys())}")
    else:
        print(f"  image fetch failed")

    print()
    print("Done. 2 of 100 daily API calls used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
