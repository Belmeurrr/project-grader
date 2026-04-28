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
    centering: float
    corners: float
    edges: float
    surface: float | None
    final: float
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
