from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from grader.db.models import (
    AuthenticityVerdict,
    Game,
    GradingScheme,
    ShotKind,
    SubmissionStatus,
)

# --------------------------------------------------------------------------
# Damage-heatmap region thresholds (Phase 1 MVP).
#
# Score → severity mapping for the public cert page's damage heatmap. The
# midpoints below are PLACEHOLDERS — they were chosen by intuition pending
# the empirical Grade-distribution work (the same exercise that produced
# the counterfeit recalibration thresholds for rosette/color). Do not tune
# them yet. The intent is purely to give the heatmap three visually
# distinct buckets (green/amber/red) so users see a "Precision Defect
# Mapping"–style overlay; the underlying numbers continue to be sourced
# from the existing per-criterion grades for back-compat.
# --------------------------------------------------------------------------
REGION_SEVERITY_OK_THRESHOLD: float = 0.95
REGION_SEVERITY_MINOR_THRESHOLD: float = 0.85


RegionKind = Literal["centering", "corner", "edge", "surface"]
RegionPosition = Literal[
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "top",
    "right",
    "bottom",
    "left",
    "whole_card",
]
RegionSeverity = Literal["ok", "minor", "major", "unknown"]


def _severity_from_score(score: float | None) -> RegionSeverity:
    """Map a normalized [0, 1] region score to a severity bucket.

    Inputs above ``REGION_SEVERITY_OK_THRESHOLD`` (0.95) → "ok"; between
    that and ``REGION_SEVERITY_MINOR_THRESHOLD`` (0.85) → "minor"; below
    → "major"; ``None`` → "unknown" (e.g. corners/surface, whose trainers
    are still skeletons in Phase 1)."""
    if score is None:
        return "unknown"
    if score > REGION_SEVERITY_OK_THRESHOLD:
        return "ok"
    if score >= REGION_SEVERITY_MINOR_THRESHOLD:
        return "minor"
    return "major"


class RegionScore(BaseModel):
    """One cell of the public cert page's damage-heatmap overlay.

    Phase-1 MVP scope: surface the per-region grades we already produce
    (centering whole-card, an aggregate edges entry until the per-side
    breakdown gets persisted, corner placeholders, and a surface
    placeholder). The web overlay positions cells by ``position`` and
    color-codes them by ``severity``.

    ``reasons`` is a free-text list of itemized rationale strings —
    "DINGS-style" defect descriptions surfaced under the heatmap on the
    cert page (e.g. "Minor whitening or wear", "Edge chipping or
    notching"). Today these are derived heuristically from severity +
    kind in ``grader.routers.cert._build_regions_for_grade``; once the
    trainers ship richer outputs (measured pixel locations, per-defect
    bounding boxes, textual classifications) those should populate this
    field directly so the cert page renders verbatim. Intentionally a
    plain ``list[str]`` rather than a structured shape so future trainer
    work can swap in without a schema bump."""

    kind: RegionKind
    position: RegionPosition
    score: float | None = None
    severity: RegionSeverity
    reasons: list[str] = Field(default_factory=list)


class SubmissionCreate(BaseModel):
    game_hint: Game | None = Field(default=None, description="Optional game hint to skip detection.")


class ShotUploadUrlRequest(BaseModel):
    kind: ShotKind
    content_type: str = Field(pattern=r"^image/(jpeg|png|webp)$")


class ShotUploadUrlResponse(BaseModel):
    """Presigned POST form for direct client → S3 upload.

    The shape mirrors the AWS ``generate_presigned_post`` response: the
    client constructs a ``multipart/form-data`` POST against ``url``
    with every entry in ``fields`` set verbatim, then appends the file
    blob under field name ``file`` (last). The signed policy embedded
    in ``fields["policy"]`` includes a ``content-length-range``
    condition that S3 enforces server-side, so an oversized upload is
    rejected with 400 before any bytes hit our budget."""

    shot_id: uuid.UUID
    url: str
    fields: dict[str, str]
    s3_key: str
    expires_at: datetime


class ShotRegisterRequest(BaseModel):
    shot_id: uuid.UUID
    s3_key: str
    # ``kind`` is required and validated to match the kind that was
    # presigned for this s3_key. Inferring it from the key path alone
    # let a client mix shot_ids/keys across two presigns and end up
    # with a row whose ``kind`` column doesn't reflect the upload — see
    # the kind-spoofing test in test_shots.py.
    kind: ShotKind


class ShotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: ShotKind
    s3_key: str
    width_px: int
    height_px: int
    quality_passed: bool
    quality_metadata: dict[str, object] = Field(default_factory=dict)


class SubmitResponse(BaseModel):
    """Returned from POST /submissions/{id}/submit. The pipeline runs
    asynchronously; clients poll GET /submissions/{id} until status moves
    out of `processing`."""

    submission_id: uuid.UUID
    status: SubmissionStatus
    task_id: str | None = None


class GradeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scheme: GradingScheme
    # All subgrades + final are nullable: corners/edges/surface trainers
    # are still skeletons in Phase 1, and `compute_psa_final` returns
    # None when any input is missing. The DB columns mirror this. The
    # cert UI renders missing values as "—" and shows a banner explaining
    # the partial state.
    centering: float | None
    corners: float | None
    edges: float | None
    surface: float | None
    final: float | None
    confidence: float
    front_centering_lr: float | None = None
    front_centering_tb: float | None = None
    back_centering_lr: float | None = None
    back_centering_tb: float | None = None


class AuthenticityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    verdict: AuthenticityVerdict
    confidence: float
    reasons: list[str]
    # The persisted shape under `AuthenticityResult.detector_scores` is
    # `{<detector>: {<metric>: <value>, ...}}` — see
    # `services/counterfeit.py::persist_authenticity_result`. The owner-
    # side response surfaces it verbatim; the public cert page uses the
    # richer `CertAuthenticityPublic` instead.
    detector_scores: dict[str, dict[str, Any]]


class IdentifiedCard(BaseModel):
    variant_id: uuid.UUID
    name: str
    set_code: str
    card_number: str
    confidence: float


class SubmissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: SubmissionStatus
    created_at: datetime
    completed_at: datetime | None
    rejection_reason: str | None
    identified_card: IdentifiedCard | None = None
    grades: list[GradeOut] = Field(default_factory=list)
    authenticity: AuthenticityOut | None = None


class DetectorScorePublic(BaseModel):
    """Per-detector entry in the public cert page's authenticity block.

    Mirrors the shape persisted under
    `AuthenticityResult.detector_scores[<detector>]` after commit
    94b83ca: scalar score + per-detector verdict + per-detector
    confidence, with a typed-but-open `metadata` slot for detector-
    specific extras (peak_strength for rosette, p95_chroma for color,
    etc.) so the cert page can render forensic detail without each
    new detector requiring a schema bump."""

    detector: str
    score: float
    verdict: AuthenticityVerdict
    confidence: float
    metadata: dict[str, object] = Field(default_factory=dict)


class CertAuthenticityPublic(BaseModel):
    """Public-cert view of the AuthenticityResult row.

    Distinct from `AuthenticityOut` (which surfaces the raw nested
    `detector_scores` blob verbatim). This shape exposes the per-
    detector breakdown the cert page renders."""

    verdict: AuthenticityVerdict
    confidence: float
    reasons: list[str]
    model_versions: dict[str, object] = Field(default_factory=dict)
    detectors: list[DetectorScorePublic] = Field(default_factory=list)


class PopulationStat(BaseModel):
    """Population / chronology counter for a cert.

    TAG-inspired scarcity stats: how many of THIS variant we've graded,
    where this submission ranks by final grade among them, the highest
    grade we've ever issued for the variant, and where this submission
    sits in the chronological grading history. Computed on the fly from
    the COMPLETED submissions for the same `identified_variant_id`; the
    cert payload omits the field entirely (`population=None`) when the
    submission has no identified variant — surfacing "1 of 1 graded" is
    noise we'd rather skip."""

    total_graded: int
    this_rank: int
    max_grade: float | None
    chronological_index: int


class CertImagePublic(BaseModel):
    """Time-limited presigned-GET URLs for the canonical card images.

    Expire 1 hour after the cert payload is generated. Suitable for
    direct rendering in the public cert page — the URLs are bounded
    blast-radius (single S3 object, single cert id, narrow TTL) so the
    privacy posture stays the same as the rest of the public cert
    payload: no raw S3 keys leaked, no ambient bucket access.

    Any of the URL fields may be None: ``front_canonical_url`` is the
    only one guaranteed when canonicals exist (it's the always-required
    front_full shot), while ``front_flash_url`` and ``tilt_url`` depend
    on the user having captured those optional shots."""

    front_canonical_url: str | None = None
    front_flash_url: str | None = None
    tilt_url: str | None = None
    expires_at: datetime


class CertificatePublic(BaseModel):
    """Public-cert view of a COMPLETED submission.

    Returned by `GET /cert/{submission_id}` without authentication.
    Deliberately a strict SUBSET of the owner-side `SubmissionOut` —
    no user_id, no S3 keys, no shot-level metadata, no audit log.
    The `cert_id` field is the submission UUID; the owner sees it as
    `id`, the public sees it as `cert_id` to match the user-facing
    label."""

    cert_id: uuid.UUID
    completed_at: datetime
    identified_card: IdentifiedCard | None = None
    grades: list[GradeOut] = Field(default_factory=list)
    authenticity: CertAuthenticityPublic | None = None
    # Damage-heatmap regions. Purely additive; the existing
    # per-criterion `grades` array stays the canonical source of truth
    # for the numeric subgrades. Built fresh on each request from the
    # primary Grade row — see `_build_regions_for_grade` in
    # `grader.routers.cert`.
    regions: list[RegionScore] = Field(default_factory=list)
    # Population / chronology counter. Null when the submission has no
    # identified variant (no peer set to compute against). See
    # `PopulationStat` and `_build_population_stat` in `grader.routers.cert`.
    population: PopulationStat | None = None
    # Presigned-GET URLs for the canonical card images. None when no
    # canonicals exist (e.g. detection soft-failed) or when presigning
    # itself failed (S3 unhealthy, bad creds — best-effort degrade so
    # the cert page still renders the rest of the payload).
    images: CertImagePublic | None = None
