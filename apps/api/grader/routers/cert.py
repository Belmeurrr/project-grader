"""Public certificate endpoint.

Serves the read-only, cacheable, no-auth view of a COMPLETED
submission for the `/cert/[id]` page in apps/web. Distinct from the
owner-side `GET /submissions/{id}` (which requires auth + ownership);
this endpoint is the artifact a user shares with friends or attaches
to a marketplace listing.

What's exposed:
  - cert_id (= submission_id)
  - completed_at
  - identified card (name, set, year — if identification succeeded)
  - per-criterion grades (centering, edges, corners, surface, final)
  - authenticity verdict + per-detector breakdown (scores, verdicts,
    forensic metadata like peak_strength / p95_chroma)

What's NOT exposed:
  - user_id (privacy)
  - raw S3 keys (the `images` block exposes only time-limited
    presigned-GET URLs scoped to a single object, never the bucket
    path)
  - audit log entries (internal pipeline detail)
  - shot-level metadata (blur scores, perspective angles)
  - in-progress / failed submissions (404 unless status==COMPLETED;
    don't surface partial results that may still change)

What IS exposed (image-side):
  - 1-hour presigned-GET URLs for the canonical front, optional
    flash, and optional tilt. These power the Card Vision opacity
    slider on the cert page (regular front <-> flash crossfade).

Caching:
  Sets `Cache-Control: public, max-age=2400` (40 min, no SWR) on
  successful responses. The original 5-min max-age + 1-hour SWR is
  shortened here because the response body now embeds presigned URLs
  with a 1-hour TTL — SWR would happily serve a stale URL after
  expiry. 40 min keeps a 20-min margin of presign validity at the
  edge of the cache window without sacrificing too much hit rate.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from grader.db import get_db
from grader.db.models import CardVariant, Grade, ShotKind, Submission, SubmissionStatus
from grader.schemas.submissions import (
    CertAuthenticityPublic,
    CertImagePublic,
    CertificatePublic,
    DetectorScorePublic,
    GradeOut,
    IdentifiedCard,
    PopulationStat,
    RegionKind,
    RegionScore,
    RegionSeverity,
    _severity_from_score,
)
from grader.services import storage
from grader.services.rate_limit import limiter

router = APIRouter(prefix="/cert", tags=["cert"])


# Presigned-GET URLs we hand out for the canonical card images expire
# after _IMAGE_PRESIGN_TTL_SECONDS. The cache header below is shortened
# (vs. the original 5-min max-age + 1-hour SWR) so that a CDN-cached
# cert payload always has at least 20 minutes of presign validity left
# when served. Trade-off: we lose some hit rate at the CDN, but the
# alternative is serving a stale URL that 403s in the user's browser.
_IMAGE_PRESIGN_TTL_SECONDS = 3600  # 1 hour
_PUBLIC_CACHE_HEADER = "public, max-age=2400"  # 40 min, no SWR


@router.get("/{submission_id}", response_model=CertificatePublic)
@limiter.limit("60/minute")  # default key_func: per-IP. Public endpoint, no user context.
async def get_certificate(
    request: Request,
    submission_id: uuid.UUID,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> CertificatePublic:
    """Public, cacheable read of a completed submission's certificate.

    Returns 404 for any of:
      - submission_id doesn't exist
      - submission exists but status != COMPLETED (in-progress or failed)

    Both 404s use the same opaque message so a probing client can't
    distinguish "doesn't exist" from "exists but isn't ready" — minor
    privacy benefit, no functional cost."""
    # Use `select(...).options(...)` rather than `db.get(...,
    # options=...)`: the SA-2.0.x release we ship silently drops the
    # `options=` argument from `get(...)`, which leaves the
    # relationships unloaded and crashes the cert builder with
    # MissingGreenlet on the first attribute access.
    result = await db.execute(
        select(Submission)
        .where(Submission.id == submission_id)
        .options(
            selectinload(Submission.grades),
            selectinload(Submission.authenticity),
            # Chain through CardVariant → CardSet so we can render the
            # set code on the cert without a second query.
            selectinload(Submission.identified_variant).selectinload(
                CardVariant.set
            ),
        )
    )
    submission = result.scalar_one_or_none()
    if submission is None or submission.status != SubmissionStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="cert not found",
        )

    grades = [GradeOut.model_validate(g) for g in submission.grades]
    auth = (
        _build_authenticity_public(submission.authenticity)
        if submission.authenticity is not None
        else None
    )
    # Damage-heatmap regions. PSA is the canonical scheme on the cert
    # page; build the regions list off the PSA grade if present, else
    # the first available scheme. Empty list when no grade row exists
    # at all (a defensive case — COMPLETED submissions should always
    # have a Grade row, but the cert endpoint already tolerates the
    # `grades=[]` edge in `GradesSection` of the web page).
    primary_grade = _primary_grade(submission)
    regions = _build_regions_for_grade(primary_grade) if primary_grade else []

    population = await _build_population_stat(db, submission)
    images = _build_images_public(submission.id)

    response.headers["Cache-Control"] = _PUBLIC_CACHE_HEADER
    return CertificatePublic(
        cert_id=submission.id,
        completed_at=submission.completed_at,  # required when status=COMPLETED
        identified_card=_identified_card_or_none(submission),
        grades=grades,
        authenticity=auth,
        regions=regions,
        population=population,
        images=images,
    )


async def _build_population_stat(
    db: AsyncSession, submission: Submission
) -> PopulationStat | None:
    """Compute the TAG-style pop/rank/chron counter for this submission.

    Returns None when:
      - the submission has no identified variant (no peer set)
      - the submission's PSA grade row isn't found in the window result
        (defensive — a COMPLETED submission with no PSA grade row would
        already have returned `regions=[]` upstream; we don't want to
        crash the cert page on that edge)

    Otherwise computes via a single window-function query over all
    COMPLETED submissions for the same `identified_variant_id`:

      - `total_graded`   = COUNT(*) over the window
      - `this_rank`      = RANK() over (ORDER BY g.final DESC NULLS LAST)
      - `max_grade`      = MAX(g.final) over the window
      - `chronological_index` = ROW_NUMBER() over (ORDER BY s.completed_at)

    Then filters in Python to find the row matching this submission's id.
    Pinned to `scheme = 'psa'` so rank/max are computed against the same
    scheme the cert page leads with — mixing PSA + BGS finals would
    double-count the same submission and produce nonsense ranks."""
    if submission.identified_variant_id is None:
        return None

    # Cast string literals to the native PG enums so this works regardless
    # of whether Alembic created the enum types with lowercase values
    # ('completed') or SQLAlchemy's default uppercase names ('COMPLETED').
    # Comparing s.status::text = 'completed' would work too, but the cast
    # back to the enum keeps the comparison index-friendly.
    sql = text(
        """
        SELECT
            s.id AS submission_id,
            COUNT(*) OVER () AS total_graded,
            RANK() OVER (ORDER BY g.final DESC NULLS LAST) AS rank_by_grade,
            MAX(g.final) OVER () AS max_grade,
            ROW_NUMBER() OVER (ORDER BY s.completed_at) AS chronological_index
        FROM submissions s
        JOIN grades g ON g.submission_id = s.id
        WHERE s.identified_variant_id = :variant_id
          AND s.status::text = :completed_status
          AND g.scheme::text = :psa_scheme
        """
    )
    rows = (
        await db.execute(
            sql,
            {
                "variant_id": submission.identified_variant_id,
                # SQLAlchemy `Enum(...)` on a plain enum.Enum stores the
                # member NAME (uppercase), not the lowercase string value.
                # Cast to text on the column side and compare uppercase.
                "completed_status": SubmissionStatus.COMPLETED.name,
                "psa_scheme": "PSA",
            },
        )
    ).mappings().all()

    for row in rows:
        if row["submission_id"] == submission.id:
            return PopulationStat(
                total_graded=int(row["total_graded"]),
                this_rank=int(row["rank_by_grade"]),
                max_grade=(
                    float(row["max_grade"])
                    if row["max_grade"] is not None
                    else None
                ),
                chronological_index=int(row["chronological_index"]),
            )
    return None


def _primary_grade(submission: Submission) -> Grade | None:
    """Pick the canonical Grade row for the cert page.

    The web cert page leads with PSA when present (``primary = ...
    find scheme === "psa"``), so the heatmap should mirror that to keep
    the numbers and the overlay self-consistent."""
    if not submission.grades:
        return None
    for g in submission.grades:
        if g.scheme.value == "psa":
            return g
    return submission.grades[0]


# Canonical-image S3 keys are derived deterministically from the
# submission id and shot kind by ``detection._canonical_s3_key`` —
# layout: ``submissions/<id>/canonical/<kind>.png``. We don't persist
# the keys on the Submission row; instead the cert endpoint HEADs each
# of the three public-facing kinds and presigns the ones that exist.
# Front_full is always present for COMPLETED submissions; flash + tilt
# are optional captures that may legitimately be missing.
_PUBLIC_CANONICAL_KINDS: tuple[ShotKind, ...] = (
    ShotKind.FRONT_FULL,
    ShotKind.FRONT_FULL_FLASH,
    ShotKind.TILT_30,
)


def _canonical_key_for(submission_id: uuid.UUID, kind: ShotKind) -> str:
    """Mirror of ``detection._canonical_s3_key`` — kept inline here to
    avoid a router → services/detection import cycle and because the
    layout is part of the public API contract (the cert page renders
    these keys via presigned URLs)."""
    return f"submissions/{submission_id}/canonical/{kind.value}.png"


def _build_images_public(submission_id: uuid.UUID) -> CertImagePublic | None:
    """Build the public images block by HEADing each canonical kind and
    presigning the ones that exist.

    Returns None when none of the canonical kinds resolve — keeps the
    cert payload honest for submissions whose detection stage soft-
    failed (no canonicals produced) and for any future submissions
    where S3 itself is unhealthy."""
    urls: dict[str, str | None] = {
        "front_canonical_url": None,
        "front_flash_url": None,
        "tilt_url": None,
    }
    field_for_kind = {
        ShotKind.FRONT_FULL: "front_canonical_url",
        ShotKind.FRONT_FULL_FLASH: "front_flash_url",
        ShotKind.TILT_30: "tilt_url",
    }
    found_any = False
    for kind in _PUBLIC_CANONICAL_KINDS:
        key = _canonical_key_for(submission_id, kind)
        try:
            head = storage.head_shot(key)
        except Exception:
            head = None
        if head is None:
            continue
        url = storage.presigned_get_for_canonical(
            key, expires_in_seconds=_IMAGE_PRESIGN_TTL_SECONDS
        )
        if url is None:
            continue
        urls[field_for_kind[kind]] = url
        found_any = True

    if not found_any:
        return None
    return CertImagePublic(
        front_canonical_url=urls["front_canonical_url"],
        front_flash_url=urls["front_flash_url"],
        tilt_url=urls["tilt_url"],
        expires_at=datetime.now(timezone.utc)
        + timedelta(seconds=_IMAGE_PRESIGN_TTL_SECONDS),
    )


# --------------------------------------------------------------------------
# DINGS-style itemized defect rationale.
#
# TAG Grading's perceived-transparency edge is the textual reason list
# next to the heatmap ("Top-left corner: minor whitening", "Bottom edge:
# notch at 3% from corner"). We don't have measured per-defect outputs
# yet — the corners/surface trainers are skeletons, the edges grader
# emits per-side measurements but nothing is persisted at the bb /
# pixel-coordinate level. So the strings below are a HEURISTIC stand-in
# keyed off (kind, severity): future trainer work will replace these by
# populating ``RegionScore.reasons`` directly with measured rationale
# (e.g. "notch at 3% from top-left corner, 1.2mm").
#
# Empty string lists are intentional for severity=ok ("nothing to flag")
# so the cert page can show an honest "No defects flagged" empty state.
# severity=unknown surfaces "Analysis pending" only for kinds whose
# trainers haven't shipped (corners, surface today) — for centering /
# edges, an unknown score means the row literally lacks the column,
# which is an unusual partial-grade state already disclosed by the
# preliminary-grade banner above.
# --------------------------------------------------------------------------
_REASON_TABLE: dict[tuple[RegionKind, RegionSeverity], list[str]] = {
    ("centering", "minor"): ["Minor off-center"],
    ("centering", "major"): ["Significant off-center crop"],
    ("corner", "minor"): ["Minor whitening or wear"],
    ("corner", "major"): ["Visible corner damage"],
    ("edge", "minor"): ["Minor edge wear"],
    ("edge", "major"): ["Edge chipping or notching"],
    ("surface", "minor"): ["Light surface marks"],
    ("surface", "major"): ["Significant surface damage"],
}

# Kinds whose trainers are still skeletons in Phase 1; surface "Analysis
# pending" for unknown severity rather than nothing, so the user sees
# why a defect callout might be absent. Centering + edges have shipped
# scoring and shouldn't show a placeholder.
_PENDING_KINDS: frozenset[RegionKind] = frozenset({"corner", "surface"})


def _reasons_for(kind: RegionKind, severity: RegionSeverity) -> list[str]:
    """Heuristic stand-in mapping (kind, severity) → rationale strings.

    TODO: once the trainers ship richer per-defect outputs (measured
    pixel locations, per-defect bounding boxes, textual classifications)
    populate ``RegionScore.reasons`` from those outputs upstream and
    drop this helper. The table is intentionally tiny and string-keyed
    so that's a clean swap-in."""
    if severity == "ok":
        return []
    if severity == "unknown":
        return ["Analysis pending"] if kind in _PENDING_KINDS else []
    return list(_REASON_TABLE.get((kind, severity), []))


def _build_regions_for_grade(grade: Grade) -> list[RegionScore]:
    """Translate a Grade row into the public ``regions`` list.

    Phase-1 MVP shape (10 entries):
      - 1 centering, position=whole_card, score = grade.centering / 10
      - 4 edges (top/right/bottom/left). The edges grader internally
        emits per-side measurements (see
        ``ml/pipelines/grading/edges/measure.py::SideMeasurement``) but
        the persistence layer collapses to a single aggregate column
        (``Grade.edges``). Until per-side scores are persisted to
        ``Grade.distributions``, all four side entries share that
        aggregate score. Documented gap; see TODO.md.
      - 4 corners (top_left/top_right/bottom_left/bottom_right). The
        corners trainer is a Phase-1 skeleton, so each entry surfaces
        ``score=None`` / ``severity="unknown"``.
      - 1 surface, position=whole_card, score=None / unknown today.

    Scores are normalized from PSA's 0-10 grade scale to [0, 1] for the
    severity mapping. None scores stay None; ``_severity_from_score``
    handles the unknown bucket."""
    regions: list[RegionScore] = []

    centering_norm = grade.centering / 10.0 if grade.centering is not None else None
    centering_severity = _severity_from_score(centering_norm)
    regions.append(
        RegionScore(
            kind="centering",
            position="whole_card",
            score=centering_norm,
            severity=centering_severity,
            reasons=_reasons_for("centering", centering_severity),
        )
    )

    edges_norm = grade.edges / 10.0 if grade.edges is not None else None
    edges_severity = _severity_from_score(edges_norm)
    edges_reasons = _reasons_for("edge", edges_severity)
    for edge_pos in ("top", "right", "bottom", "left"):
        regions.append(
            RegionScore(
                kind="edge",
                position=edge_pos,  # type: ignore[arg-type]
                score=edges_norm,
                severity=edges_severity,
                reasons=list(edges_reasons),
            )
        )

    corners_norm = grade.corners / 10.0 if grade.corners is not None else None
    corners_severity = _severity_from_score(corners_norm)
    corners_reasons = _reasons_for("corner", corners_severity)
    for corner_pos in ("top_left", "top_right", "bottom_left", "bottom_right"):
        regions.append(
            RegionScore(
                kind="corner",
                position=corner_pos,  # type: ignore[arg-type]
                score=corners_norm,
                severity=corners_severity,
                reasons=list(corners_reasons),
            )
        )

    surface_norm = grade.surface / 10.0 if grade.surface is not None else None
    surface_severity = _severity_from_score(surface_norm)
    regions.append(
        RegionScore(
            kind="surface",
            position="whole_card",
            score=surface_norm,
            severity=surface_severity,
            reasons=_reasons_for("surface", surface_severity),
        )
    )

    return regions


def _identified_card_or_none(submission: Submission) -> IdentifiedCard | None:
    """Build the public IdentifiedCard payload from a submission's
    eager-loaded `identified_variant`, or return None if the variant
    didn't load / wasn't identified.

    Confidence comes from the submission row (the identification
    service writes it alongside `identified_variant_id`). If the row
    has a variant but a NULL confidence — which shouldn't happen in
    practice but isn't enforced at the schema level — we coerce to 0.0
    rather than 404 the cert."""
    variant = submission.identified_variant
    if variant is None:
        return None
    return IdentifiedCard(
        variant_id=variant.id,
        name=variant.name,
        set_code=variant.set.code,
        card_number=variant.card_number,
        confidence=float(submission.identification_confidence or 0.0),
    )


def _build_authenticity_public(row) -> CertAuthenticityPublic:
    """Translate the persisted AuthenticityResult into the public shape.

    `detector_scores` on the row is a nested dict:
        {"rosette": {"score": float, "verdict": str, "confidence": float, ...},
         "color":   {"score": float, "verdict": str, "confidence": float, ...}}
    We unpack each detector entry into a DetectorScorePublic and
    surface the rest under metadata."""
    detectors: list[DetectorScorePublic] = []
    raw = row.detector_scores or {}
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            # Defensive — old single-detector rows stored a flat float.
            # Skip rather than crash; the row is from before the
            # ensemble refactor and doesn't fit the public schema.
            continue
        detectors.append(
            DetectorScorePublic(
                detector=name,
                score=float(payload.get("score", 0.0)),
                verdict=payload.get("verdict", "unverified"),
                confidence=float(payload.get("confidence", 0.0)),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in {"score", "verdict", "confidence"}
                },
            )
        )

    return CertAuthenticityPublic(
        verdict=row.verdict,
        confidence=float(row.confidence),
        reasons=list(row.reasons or []),
        model_versions=dict(row.model_versions or {}),
        detectors=detectors,
    )
