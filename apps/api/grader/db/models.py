from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB}


class Game(str, enum.Enum):
    POKEMON = "pokemon"
    MTG = "mtg"
    YUGIOH = "yugioh"
    ONE_PIECE = "one_piece"
    LORCANA = "lorcana"
    SPORTS = "sports"
    OTHER = "other"


class SubmissionStatus(str, enum.Enum):
    PENDING = "pending"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED_QUALITY = "rejected_quality"


class ShotKind(str, enum.Enum):
    FRONT_FULL = "front_full"
    FRONT_FULL_FLASH = "front_full_flash"
    BACK_FULL = "back_full"
    CORNER_TL = "corner_tl"
    CORNER_TR = "corner_tr"
    CORNER_BL = "corner_bl"
    CORNER_BR = "corner_br"
    TILT_30 = "tilt_30"


class GradingScheme(str, enum.Enum):
    PSA = "psa"
    BGS = "bgs"
    TRUGRADE = "trugrade"


class AuthenticityVerdict(str, enum.Enum):
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    LIKELY_COUNTERFEIT = "likely_counterfeit"
    UNVERIFIED = "unverified"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clerk_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(120))
    training_consent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), unique=True)

    submissions: Mapped[list[Submission]] = relationship(back_populates="user")


class CardSet(Base, TimestampMixin):
    __tablename__ = "card_sets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game: Mapped[Game] = mapped_column(Enum(Game), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    release_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    variants: Mapped[list[CardVariant]] = relationship(back_populates="set")

    __table_args__ = (Index("ix_card_sets_game_code", "game", "code", unique=True),)


class CardVariant(Base, TimestampMixin):
    __tablename__ = "card_variants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game: Mapped[Game] = mapped_column(Enum(Game), nullable=False, index=True)
    set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("card_sets.id"), nullable=False)
    card_number: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_foil: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_holo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_promo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    parallel: Mapped[str | None] = mapped_column(String(64))

    canonical_image_url: Mapped[str | None] = mapped_column(Text)
    canonical_phash: Mapped[int | None] = mapped_column(BigInteger, index=True)
    canonical_image_embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)

    tcgplayer_id: Mapped[str | None] = mapped_column(String(64), index=True)
    scryfall_id: Mapped[str | None] = mapped_column(String(64), index=True)
    external_ids: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    set: Mapped[CardSet] = relationship(back_populates="variants")

    __table_args__ = (
        Index("ix_card_variants_game_set_number", "game", "set_id", "card_number"),
    )


class Submission(Base, TimestampMixin):
    __tablename__ = "submissions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus), nullable=False, default=SubmissionStatus.PENDING, index=True
    )
    identified_variant_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("card_variants.id"), index=True
    )
    identification_confidence: Mapped[float | None] = mapped_column(Float)

    rejection_reason: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    user: Mapped[User] = relationship(back_populates="submissions")
    shots: Mapped[list[SubmissionShot]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )
    grades: Mapped[list[Grade]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )
    authenticity: Mapped[AuthenticityResult | None] = relationship(
        back_populates="submission", cascade="all, delete-orphan", uselist=False
    )
    # Many-to-one to the catalog variant the identification stage matched.
    # No back_populates — we never navigate from CardVariant → submissions
    # (that's an inverted index we don't need). Lazy="select" by default,
    # so callers that want the variant in an async context MUST eager-load
    # via `selectinload(Submission.identified_variant)` to avoid a sync
    # lazy-load over the asyncpg driver.
    identified_variant: Mapped[CardVariant | None] = relationship()


class SubmissionShot(Base, TimestampMixin):
    __tablename__ = "submission_shots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[ShotKind] = mapped_column(Enum(ShotKind), nullable=False)
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    height_px: Mapped[int] = mapped_column(Integer, nullable=False)

    blur_score: Mapped[float | None] = mapped_column(Float)
    glare_score: Mapped[float | None] = mapped_column(Float)
    perspective_deg: Mapped[float | None] = mapped_column(Float)
    quality_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    quality_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    submission: Mapped[Submission] = relationship(back_populates="shots")

    __table_args__ = (Index("ix_submission_shots_submission_kind", "submission_id", "kind"),)


class Grade(Base, TimestampMixin):
    """A single per-scheme grade output (PSA-style, BGS-style, TruGrade)."""

    __tablename__ = "grades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scheme: Mapped[GradingScheme] = mapped_column(Enum(GradingScheme), nullable=False)

    # Centering is deterministic and always present once a canonical exists.
    # Other criteria are nullable so partial grading runs (e.g. centering-only
    # before the corners/edges/surface models land) can persist a Grade row.
    centering: Mapped[float] = mapped_column(Float, nullable=False)
    corners: Mapped[float | None] = mapped_column(Float)
    edges: Mapped[float | None] = mapped_column(Float)
    surface: Mapped[float | None] = mapped_column(Float)
    final: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    front_centering_lr: Mapped[float | None] = mapped_column(Float)
    front_centering_tb: Mapped[float | None] = mapped_column(Float)
    back_centering_lr: Mapped[float | None] = mapped_column(Float)
    back_centering_tb: Mapped[float | None] = mapped_column(Float)

    distributions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    explanations: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    submission: Mapped[Submission] = relationship(back_populates="grades")

    __table_args__ = (
        Index("ix_grades_submission_scheme", "submission_id", "scheme", unique=True),
    )


class AuthenticityResult(Base, TimestampMixin):
    __tablename__ = "authenticity_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    verdict: Mapped[AuthenticityVerdict] = mapped_column(Enum(AuthenticityVerdict), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detector_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    submission: Mapped[Submission] = relationship(back_populates="authenticity")


class AuditLog(Base, TimestampMixin):
    """Immutable audit trail. Every grade/authenticity decision must be reproducible."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("submissions.id"), index=True
    )
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    action: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
