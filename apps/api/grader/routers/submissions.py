from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from grader.db import get_db
from grader.db.models import (
    CardVariant,
    Submission,
    SubmissionShot,
    SubmissionStatus,
    User,
)
from grader.schemas.submissions import (
    AuthenticityOut,
    GradeOut,
    IdentifiedCard,
    ShotOut,
    ShotRegisterRequest,
    ShotUploadUrlRequest,
    ShotUploadUrlResponse,
    SubmissionCreate,
    SubmissionOut,
    SubmitResponse,
)
from grader.services import quality, storage
from grader.services.auth import get_current_user
from grader.workers.pipeline_runner import REQUIRED_SHOTS

router = APIRouter(prefix="/submissions", tags=["submissions"])


async def _load_owned_submission(
    submission_id: uuid.UUID,
    user: User,
    db: AsyncSession,
) -> Submission:
    sub = await db.get(Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="submission not found")
    return sub


@router.post("", response_model=SubmissionOut, status_code=status.HTTP_201_CREATED)
async def create_submission(
    payload: SubmissionCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SubmissionOut:
    submission = Submission(
        user_id=user.id,
        status=SubmissionStatus.PENDING,
        metadata_={"game_hint": payload.game_hint.value if payload.game_hint else None},
    )
    db.add(submission)
    await db.commit()
    # Re-fetch with the same selectinload options that `_to_out` requires,
    # using `populate_existing=True` so the options apply even though the
    # row is already in the identity map from the just-committed write.
    # Without this the relationship attributes (`.grades`, `.authenticity`,
    # `.identified_variant`) are unloaded and accessing them in `_to_out`
    # triggers a sync lazy-load under asyncpg → MissingGreenlet.
    submission = await db.get(
        Submission,
        submission.id,
        options=[
            selectinload(Submission.grades),
            selectinload(Submission.authenticity),
            selectinload(Submission.identified_variant).selectinload(
                CardVariant.set
            ),
        ],
        populate_existing=True,
    )
    assert submission is not None  # we just committed it
    return _to_out(submission)


@router.get("/{submission_id}", response_model=SubmissionOut)
async def get_submission(
    submission_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SubmissionOut:
    submission = await db.get(
        Submission,
        submission_id,
        options=[
            selectinload(Submission.grades),
            selectinload(Submission.authenticity),
            selectinload(Submission.identified_variant).selectinload(
                CardVariant.set
            ),
        ],
        # `populate_existing=True` makes the eager-load options apply even
        # when the row is already in the session's identity map (e.g. from
        # a recent write in the same async session). Without it, a cache
        # hit silently drops the selectinload options and `_to_out` then
        # triggers a sync lazy-load under asyncpg → MissingGreenlet.
        populate_existing=True,
    )
    if submission is None or submission.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="submission not found")
    return _to_out(submission)


@router.post(
    "/{submission_id}/shots/upload-url",
    response_model=ShotUploadUrlResponse,
    status_code=status.HTTP_201_CREATED,
)
async def request_shot_upload_url(
    submission_id: uuid.UUID,
    payload: ShotUploadUrlRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ShotUploadUrlResponse:
    """Issue a presigned PUT URL for the client to upload one shot directly to S3.

    The shot row is NOT created here — only after the client confirms the
    upload via POST /shots. We reserve the shot_id up-front so the client
    can include it in the eventual register request and we can correlate."""
    await _load_owned_submission(submission_id, user, db)
    shot_id = uuid.uuid4()
    presigned = storage.presigned_put_for_shot(
        submission_id=submission_id,
        shot_id=shot_id,
        kind=payload.kind.value,
        content_type=payload.content_type,
    )
    return ShotUploadUrlResponse(
        shot_id=shot_id,
        upload_url=presigned.upload_url,
        s3_key=presigned.s3_key,
        expires_at=presigned.expires_at,
        required_headers=presigned.required_headers,
    )


@router.post(
    "/{submission_id}/shots",
    response_model=ShotOut,
    status_code=status.HTTP_201_CREATED,
)
async def register_shot(
    submission_id: uuid.UUID,
    payload: ShotRegisterRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ShotOut:
    """Register a previously-uploaded shot. Triggers quality gating synchronously.

    The client calls this after the PUT to S3 succeeds. We confirm the object
    exists in S3, decode it, run quality checks, and persist the result. The
    response carries quality_passed + reasons so the UI can prompt for retake."""
    sub = await _load_owned_submission(submission_id, user, db)

    expected_prefix = f"submissions/{submission_id}/shots/{payload.shot_id}/"
    if not payload.s3_key.startswith(expected_prefix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="s3_key does not match submission/shot ids",
        )

    existing = await db.scalar(
        select(SubmissionShot).where(SubmissionShot.id == payload.shot_id)
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="shot already registered",
        )

    kind_str = payload.s3_key[len(expected_prefix) :].split(".", 1)[0]
    try:
        from grader.db.models import ShotKind

        kind = ShotKind(kind_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unrecognized shot kind in key: {kind_str}",
        )

    try:
        result = quality.ingest_shot(payload.s3_key, kind)
    except quality.ShotNotFoundInS3Error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload not found in storage; PUT may have failed",
        )
    except quality.ShotDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded bytes are not a decodable image",
        )

    shot = SubmissionShot(
        id=payload.shot_id,
        submission_id=sub.id,
        kind=kind,
        s3_key=payload.s3_key,
        width_px=result.width_px,
        height_px=result.height_px,
        blur_score=result.quality.blur,
        glare_score=result.quality.glare,
        perspective_deg=result.quality.perspective,
        quality_passed=result.quality.passed,
        quality_metadata=result.quality.to_dict(),
    )
    db.add(shot)
    if sub.status == SubmissionStatus.PENDING:
        sub.status = SubmissionStatus.CAPTURING
    await db.commit()
    await db.refresh(shot)
    return ShotOut.model_validate(shot)


@router.post(
    "/{submission_id}/submit",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_submission(
    submission_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SubmitResponse:
    """Enqueue the grading pipeline for this submission.

    Pre-flight check: every required shot must be uploaded with quality
    passed. Optional shots (back) are accepted with or without — front-only
    centering is a valid grading run.

    Idempotency: re-submitting a submission already in PROCESSING returns
    the current status without enqueuing again. Re-submitting a COMPLETED
    or FAILED submission returns the existing terminal state."""
    sub = await _load_owned_submission(submission_id, user, db)

    if sub.status in {SubmissionStatus.PROCESSING, SubmissionStatus.COMPLETED, SubmissionStatus.FAILED}:
        return SubmitResponse(submission_id=sub.id, status=sub.status, task_id=None)

    shots = await db.scalars(
        select(SubmissionShot).where(SubmissionShot.submission_id == sub.id)
    )
    have_passing = {s.kind for s in shots.all() if s.quality_passed}
    missing = [k for k in REQUIRED_SHOTS if k not in have_passing]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "reason": "missing_required_shots",
                "missing": [k.value for k in missing],
            },
        )

    sub.status = SubmissionStatus.PROCESSING
    await db.commit()

    task_id: str | None = None
    try:
        from grader.workers.grading_pipeline import process_submission

        async_result = process_submission.delay(str(sub.id))
        task_id = async_result.id
    except Exception:
        # Broker unreachable — revert the row to CAPTURING so the user can
        # retry once the broker recovers (the early-return guard above
        # would otherwise strand this submission in PROCESSING forever),
        # and surface a 503 so the client knows to retry.
        sub.status = SubmissionStatus.CAPTURING
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"reason": "broker_unavailable", "retry_after": 30},
        )

    return SubmitResponse(submission_id=sub.id, status=sub.status, task_id=task_id)


def _to_out(submission: Submission) -> SubmissionOut:
    grades = [GradeOut.model_validate(g) for g in submission.grades]
    auth = AuthenticityOut.model_validate(submission.authenticity) if submission.authenticity else None
    return SubmissionOut(
        id=submission.id,
        status=submission.status,
        created_at=submission.created_at,
        completed_at=submission.completed_at,
        rejection_reason=submission.rejection_reason,
        identified_card=_identified_card_or_none(submission),
        grades=grades,
        authenticity=auth,
    )


def _identified_card_or_none(submission: Submission) -> IdentifiedCard | None:
    """Build the IdentifiedCard payload from `identified_variant` if it
    eager-loaded; otherwise None.

    Mirrors the helper in `routers/cert.py` — kept inline here rather
    than shared because the two routers serve different audiences
    (owner vs. public) and we want each one's response shape to be
    its own concern. Five lines is below the share-vs-duplicate
    threshold."""
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
