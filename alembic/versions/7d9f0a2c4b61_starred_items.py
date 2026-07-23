"""add durable starred items

Revision ID: 7d9f0a2c4b61
Revises: 5a8c1e7f2b40
Create Date: 2026-07-23

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "7d9f0a2c4b61"
down_revision: str | Sequence[str] | None = "5a8c1e7f2b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "stars",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=False),
        sa.Column(
            "starred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "target_type IN ("
            "'conversation', 'message', 'activity', 'attachment', 'blob', "
            "'messenger_contact'"
            ")",
            name="ck_stars_target_type",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("target_type", "target_id", name="uq_stars_target"),
    )
    op.create_index(
        "ix_stars_starred_at_id",
        "stars",
        ["starred_at", "id"],
        unique=False,
    )
    op.create_index(
        "ix_stars_target_type_starred_at_id",
        "stars",
        ["target_type", "starred_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_stars_target_type_starred_at_id", table_name="stars")
    op.drop_index("ix_stars_starred_at_id", table_name="stars")
    op.drop_table("stars")
