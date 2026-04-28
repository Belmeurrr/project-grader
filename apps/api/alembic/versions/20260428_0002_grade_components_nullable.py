"""make grade components nullable for partial grading runs

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-28

A Grade row represents what was graded in a single pipeline run. Until
all four criteria models exist (currently only centering is implemented;
corners/edges/surface come later), grading runs will produce *partial*
Grade rows where some criteria are NULL. The final composite score is
also NULL until every criterion required by the chosen scheme is present.

Centering remains NOT NULL: if we're persisting a Grade at all, at
minimum centering must be measured (it's deterministic and always
available once a canonical image exists)."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("grades", "corners", nullable=True)
    op.alter_column("grades", "edges", nullable=True)
    op.alter_column("grades", "final", nullable=True)
    # surface was already nullable in 0001; no-op here, kept for symmetry
    # in case a future revision tightens it.


def downgrade() -> None:
    op.alter_column("grades", "corners", nullable=False)
    op.alter_column("grades", "edges", nullable=False)
    op.alter_column("grades", "final", nullable=False)
