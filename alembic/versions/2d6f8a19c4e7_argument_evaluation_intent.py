"""tag held-out retrieval evaluation intent

Revision ID: 2d6f8a19c4e7
Revises: 7c2d4e8f1a90
Create Date: 2026-07-21

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "2d6f8a19c4e7"
down_revision: str | Sequence[str] | None = "7c2d4e8f1a90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ARGUMENT_QUERY_SHA256 = "b74f83c9353b55cd27a6083e8a136486e5be81e8d2f9e83576c931d2400ceaa1"


def upgrade() -> None:
    with op.batch_alter_table("relevance_feedback") as batch_op:
        batch_op.add_column(sa.Column("evaluation_intent", sa.String(length=64)))
    # The literal is the public, user-specified flagship query. Match only its digest so the
    # migration neither reads nor embeds private query text in logs or migration output.
    op.execute(
        sa.text(
            """
            UPDATE relevance_feedback
            SET evaluation_intent = 'argument_direct_conflict'
            WHERE evaluation_split = 'test'
              AND search_mode IN ('semantic', 'hybrid')
              AND query_sha256 = :query_sha256
            """
        ).bindparams(query_sha256=ARGUMENT_QUERY_SHA256)
    )


def downgrade() -> None:
    with op.batch_alter_table("relevance_feedback") as batch_op:
        batch_op.drop_column("evaluation_intent")
