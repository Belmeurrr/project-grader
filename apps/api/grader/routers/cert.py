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
from grader.db.models import CardVariant, Submission, SubmissionStatus
from grader.schemas.submissions import (
    CertAuthenticityPublic,
    CertificatePublic,
    DetectorScorePublic,
    GradeOut,
    IdentifiedCard,
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

    response.headers["Cache-Control"] = _PUBLIC_CACHE_HEADER
    return CertificatePublic(
        cert_id=submission.id,
        completed_at=submission.completed_at,  # required when status=COMPLETED
        identified_card=_identified_card_or_none(submission),
        grades=grades,
        authenticity=auth,
    )


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
