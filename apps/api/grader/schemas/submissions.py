from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from grader.db.models import (
    AuthenticityVerdict,
    Game,
    GradingScheme,
    ShotKind,
    SubmissionStatus,
)


class SubmissionCreate(BaseModel):
    game_hint: Game | None = Field(default=None, description="Optional game hint to skip detection.")


class ShotUploadUrlRequest(BaseModel):
    kind: ShotKind
    content_type: str = Field(pattern=r"^image/(jpeg|png|webp)$")


class ShotUploadUrlResponse(BaseModel):
    shot_id: uuid.UUID
    upload_url: str
    s3_key: str
    expires_at: datetime
    required_headers: dict[str, str]


class ShotRegisterRequest(BaseModel):
    shot_id: uuid.UUID
    s3_key: str


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
    detector_scores: dict[str, float]


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

    Distinct from `AuthenticityOut` (which carries the legacy flat
    `dict[str, float]` for detector_scores). This shape exposes the
    per-detector breakdown the cert page renders."""

    verdict: AuthenticityVerdict
    confidence: float
    reasons: list[str]
    model_versions: dict[str, object] = Field(default_factory=dict)
    detectors: list[DetectorScorePublic] = Field(default_factory=list)


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
