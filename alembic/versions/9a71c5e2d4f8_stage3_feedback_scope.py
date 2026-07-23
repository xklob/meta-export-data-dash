"""persist relevance-feedback search scope

Revision ID: 9a71c5e2d4f8
Revises: 6f3c9e82a1b4
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "9a71c5e2d4f8"
down_revision: str | Sequence[str] | None = "6f3c9e82a1b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMPTY_SCOPE_SHA256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def upgrade() -> None:
    with op.batch_alter_table("relevance_feedback") as batch_op:
        batch_op.add_column(
            sa.Column(
                "filter_scope_sha256",
                sa.String(length=64),
                server_default=EMPTY_SCOPE_SHA256,
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "filter_scope_json",
                sa.JSON(),
                server_default=sa.text("'{}'"),
                nullable=False,
            )
        )
        batch_op.drop_index("uq_relevance_feedback_query_mode_result")
        batch_op.create_index(
            "uq_relevance_feedback_query_mode_scope_result",
            ["query_sha256", "search_mode", "filter_scope_sha256", "result_id"],
            unique=True,
        )


def downgrade() -> None:
    # Scope-aware rows may collapse to the same legacy identity. Retain the newest judgment.
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
    with op.batch_alter_table("relevance_feedback") as batch_op:
        batch_op.drop_index("uq_relevance_feedback_query_mode_scope_result")
        batch_op.drop_column("filter_scope_json")
        batch_op.drop_column("filter_scope_sha256")
        batch_op.create_index(
            "uq_relevance_feedback_query_mode_result",
            ["query_sha256", "search_mode", "result_id"],
            unique=True,
        )
