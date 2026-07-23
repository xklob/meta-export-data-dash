"""add a covering index for exact active NLP status counts

Revision ID: 7c2d4e8f1a90
Revises: 3b77f95c2a10
Create Date: 2026-07-21

"""

from collections.abc import Sequence

from alembic import op

revision: str = "7c2d4e8f1a90"
down_revision: str | Sequence[str] | None = "3b77f95c2a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_nlp_annotations_active_completed_task_job",
        "nlp_annotations",
        ["active", "status", "task", "job_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_nlp_annotations_active_completed_task_job",
        table_name="nlp_annotations",
    )
