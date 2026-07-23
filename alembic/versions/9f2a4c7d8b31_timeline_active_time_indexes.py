"""add active timeline ordering indexes

Revision ID: 9f2a4c7d8b31
Revises: 6e24b9f0c1d7
Create Date: 2026-07-22

"""

from collections.abc import Sequence

from alembic import op

revision: str = "9f2a4c7d8b31"
down_revision: str | Sequence[str] | None = "6e24b9f0c1d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_messages_deleted_time_id",
        "messages",
        ["deleted_at", "timestamp_ms", "id"],
        unique=False,
    )
    op.create_index(
        "ix_activity_deleted_time_id",
        "activity_records",
        ["deleted_at", "timestamp_ms", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_activity_deleted_time_id", table_name="activity_records")
    op.drop_index("ix_messages_deleted_time_id", table_name="messages")
