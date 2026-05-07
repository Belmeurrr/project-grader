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
  - S3 keys / image URLs (would require signed-URL flow we don't
    have for public consumers; v2 swap-in)
  - audit log entries (internal pipeline detail)
  - shot-level metadata (blur scores, perspective angles)
  - in-progress / failed submissions (404 unless status==COMPLETED;
    don't surface partial results that may still change)

Caching:
  Sets `Cache-Control: public, max-age=300, stale-while-revalidate=3600`
  on successful responses. Next.js ISR uses the same window. Once a
  submission completes, its cert payload is immutable in practice
  (the unique-on-submission_id row gets re-upserted only if the
  pipeline re-runs, which is rare); 5-minute freshness with 1-hour
  SWR is a generous fit.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from grader.db import get_db
from grader.db.models import CardVariant, Grade, Submission, SubmissionStatus
from grader.schemas.submissions import (
    CertAuthenticityPublic,
    CertificatePublic,
    DetectorScorePublic,
    GradeOut,
    IdentifiedCard,
    RegionKind,
    RegionScore,
    RegionSeverity,
    _severity_from_score,
)
from grader.services.rate_limit import limiter

router = APIRouter(prefix="/cert", tags=["cert"])


_PUBLIC_CACHE_HEADER = "public, max-age=300, stale-while-revalidate=3600"


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

    response.headers["Cache-Control"] = _PUBLIC_CACHE_HEADER
    return CertificatePublic(
        cert_id=submission.id,
        completed_at=submission.completed_at,  # required when status=COMPLETED
        identified_card=_identified_card_or_none(submission),
        grades=grades,
        authenticity=auth,
        regions=regions,
    )


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
