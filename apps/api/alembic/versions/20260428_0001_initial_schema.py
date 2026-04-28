"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-28

Brings up the v1 data model: users, card sets/variants (with pgvector
embedding), submissions and shots, grades, authenticity results, audit log.

Notes:
- The pgvector extension must be installable on the target database. Aurora
  Postgres 15+ supports it; in dev we use the `pgvector/pgvector:pg16` image.
- Enums are created explicitly (not via SQLAlchemy's create_type) so we control
  type names and downgrade behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Enum names mirror grader.db.models — keep in sync.
GAME_VALUES = ("pokemon", "mtg", "yugioh", "one_piece", "lorcana", "sports", "other")
SUBMISSION_STATUS_VALUES = (
    "pending",
    "capturing",
    "processing",
    "completed",
    "failed",
    "rejected_quality",
)
SHOT_KIND_VALUES = (
    "front_full",
    "front_full_flash",
    "back_full",
    "corner_tl",
    "corner_tr",
    "corner_bl",
    "corner_br",
    "tilt_30",
)
GRADING_SCHEME_VALUES = ("psa", "bgs", "trugrade")
AUTHENTICITY_VERDICT_VALUES = (
    "authentic",
    "suspicious",
    "likely_counterfeit",
    "unverified",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    game_enum = postgresql.ENUM(*GAME_VALUES, name="game", create_type=True)
    submission_status_enum = postgresql.ENUM(
        *SUBMISSION_STATUS_VALUES, name="submissionstatus", create_type=True
    )
    shot_kind_enum = postgresql.ENUM(*SHOT_KIND_VALUES, name="shotkind", create_type=True)
    grading_scheme_enum = postgresql.ENUM(
        *GRADING_SCHEME_VALUES, name="gradingscheme", create_type=True
    )
    authenticity_verdict_enum = postgresql.ENUM(
        *AUTHENTICITY_VERDICT_VALUES, name="authenticityverdict", create_type=True
    )
    bind = op.get_bind()
    for enum in (
        game_enum,
        submission_status_enum,
        shot_kind_enum,
        grading_scheme_enum,
        authenticity_verdict_enum,
    ):
        enum.create(bind, checkfirst=True)

    # users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("clerk_id", sa.String(64), nullable=False, unique=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("display_name", sa.String(120), nullable=True),
        sa.Column(
            "training_consent",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("stripe_customer_id", sa.String(64), nullable=True, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_clerk_id", "users", ["clerk_id"])
    op.create_index("ix_users_email", "users", ["email"])

    # card_sets
    op.create_table(
        "card_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("game", game_enum, nullable=False),
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("release_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_card_sets_game", "card_sets", ["game"])
    op.create_index("ix_card_sets_code", "card_sets", ["code"])
    op.create_index(
        "ix_card_sets_game_code",
        "card_sets",
        ["game", "code"],
        unique=True,
    )

    # card_variants — primary catalog table; carries pgvector embedding
    op.create_table(
        "card_variants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("game", game_enum, nullable=False),
        sa.Column(
            "set_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("card_sets.id"),
            nullable=False,
        ),
        sa.Column("card_number", sa.String(32), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "is_foil",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "is_holo",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "is_promo",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("parallel", sa.String(64), nullable=True),
        sa.Column("canonical_image_url", sa.Text(), nullable=True),
        sa.Column("canonical_phash", sa.BigInteger(), nullable=True),
        sa.Column("canonical_image_embedding", Vector(768), nullable=True),
        sa.Column("tcgplayer_id", sa.String(64), nullable=True),
        sa.Column("scryfall_id", sa.String(64), nullable=True),
        sa.Column(
            "external_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_card_variants_game", "card_variants", ["game"])
    op.create_index("ix_card_variants_canonical_phash", "card_variants", ["canonical_phash"])
    op.create_index("ix_card_variants_tcgplayer_id", "card_variants", ["tcgplayer_id"])
    op.create_index("ix_card_variants_scryfall_id", "card_variants", ["scryfall_id"])
    op.create_index(
        "ix_card_variants_game_set_number",
        "card_variants",
        ["game", "set_id", "card_number"],
    )
    # HNSW index for embedding similarity. m=16, ef_construction=64 are pgvector
    # defaults that work well up to a few million vectors.
    op.execute(
        "CREATE INDEX ix_card_variants_embedding_hnsw "
        "ON card_variants USING hnsw (canonical_image_embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    # submissions
    op.create_table(
        "submissions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "status",
            submission_status_enum,
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "identified_variant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("card_variants.id"),
            nullable=True,
        ),
        sa.Column("identification_confidence", sa.Float(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_submissions_user_id", "submissions", ["user_id"])
    op.create_index("ix_submissions_status", "submissions", ["status"])
    op.create_index(
        "ix_submissions_identified_variant_id",
        "submissions",
        ["identified_variant_id"],
    )

    # submission_shots
    op.create_table(
        "submission_shots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "submission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("submissions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", shot_kind_enum, nullable=False),
        sa.Column("s3_key", sa.Text(), nullable=False),
        sa.Column("width_px", sa.Integer(), nullable=False),
        sa.Column("height_px", sa.Integer(), nullable=False),
        sa.Column("blur_score", sa.Float(), nullable=True),
        sa.Column("glare_score", sa.Float(), nullable=True),
        sa.Column("perspective_deg", sa.Float(), nullable=True),
        sa.Column(
            "quality_passed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "quality_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_submission_shots_submission_id",
        "submission_shots",
        ["submission_id"],
    )
    op.create_index(
        "ix_submission_shots_submission_kind",
        "submission_shots",
        ["submission_id", "kind"],
    )

    # grades
    op.create_table(
        "grades",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "submission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("submissions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scheme", grading_scheme_enum, nullable=False),
        sa.Column("centering", sa.Float(), nullable=False),
        sa.Column("corners", sa.Float(), nullable=False),
        sa.Column("edges", sa.Float(), nullable=False),
        sa.Column("surface", sa.Float(), nullable=True),
        sa.Column("final", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("front_centering_lr", sa.Float(), nullable=True),
        sa.Column("front_centering_tb", sa.Float(), nullable=True),
        sa.Column("back_centering_lr", sa.Float(), nullable=True),
        sa.Column("back_centering_tb", sa.Float(), nullable=True),
        sa.Column(
            "distributions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "explanations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "model_versions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_grades_submission_id", "grades", ["submission_id"])
    op.create_index(
        "ix_grades_submission_scheme",
        "grades",
        ["submission_id", "scheme"],
        unique=True,
    )

    # authenticity_results — exactly one per submission
    op.create_table(
        "authenticity_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "submission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("submissions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("verdict", authenticity_verdict_enum, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "detector_scores",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "reasons",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
        sa.Column(
            "model_versions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_authenticity_results_submission_id",
        "authenticity_results",
        ["submission_id"],
    )

    # audit_log
    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "submission_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("submissions.id"),
            nullable=True,
        ),
        sa.Column("actor", sa.String(120), nullable=False),
        sa.Column("action", sa.String(120), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_log_submission_id", "audit_log", ["submission_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_action", table_name="audit_log")
    op.drop_index("ix_audit_log_submission_id", table_name="audit_log")
    op.drop_table("audit_log")

    op.drop_index(
        "ix_authenticity_results_submission_id", table_name="authenticity_results"
    )
    op.drop_table("authenticity_results")

    op.drop_index("ix_grades_submission_scheme", table_name="grades")
    op.drop_index("ix_grades_submission_id", table_name="grades")
    op.drop_table("grades")

    op.drop_index(
        "ix_submission_shots_submission_kind", table_name="submission_shots"
    )
    op.drop_index(
        "ix_submission_shots_submission_id", table_name="submission_shots"
    )
    op.drop_table("submission_shots")

    op.drop_index(
        "ix_submissions_identified_variant_id", table_name="submissions"
    )
    op.drop_index("ix_submissions_status", table_name="submissions")
    op.drop_index("ix_submissions_user_id", table_name="submissions")
    op.drop_table("submissions")

    op.execute("DROP INDEX IF EXISTS ix_card_variants_embedding_hnsw")
    op.drop_index(
        "ix_card_variants_game_set_number", table_name="card_variants"
    )
    op.drop_index("ix_card_variants_scryfall_id", table_name="card_variants")
    op.drop_index("ix_card_variants_tcgplayer_id", table_name="card_variants")
    op.drop_index(
        "ix_card_variants_canonical_phash", table_name="card_variants"
    )
    op.drop_index("ix_card_variants_game", table_name="card_variants")
    op.drop_table("card_variants")

    op.drop_index("ix_card_sets_game_code", table_name="card_sets")
    op.drop_index("ix_card_sets_code", table_name="card_sets")
    op.drop_index("ix_card_sets_game", table_name="card_sets")
    op.drop_table("card_sets")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_clerk_id", table_name="users")
    op.drop_table("users")

    for name in (
        "authenticityverdict",
        "gradingscheme",
        "shotkind",
        "submissionstatus",
        "game",
    ):
        op.execute(f"DROP TYPE IF EXISTS {name}")

    # Leave pgvector and uuid-ossp extensions installed; other databases on the
    # cluster may depend on them.
