"""stage3 feedback identity

Revision ID: d3b7f50a21c4
Revises: b91e6c40d2a7
Create Date: 2026-07-20

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d3b7f50a21c4"
down_revision: str | Sequence[str] | None = "b91e6c40d2a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Keep the latest judgment for each coherent identity before enforcing uniqueness.
    op.execute(
        """
        DELETE FROM relevance_feedback
        WHERE id IN (
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY query_sha256, search_mode, result_id
                        ORDER BY created_at DESC, id DESC
                    ) AS duplicate_rank
                FROM relevance_feedback
            ) ranked
            WHERE duplicate_rank > 1
        )
        """
    )
    op.create_index(
        "uq_relevance_feedback_query_mode_result",
        "relevance_feedback",
        ["query_sha256", "search_mode", "result_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_relevance_feedback_query_mode_result",
        table_name="relevance_feedback",
    )
