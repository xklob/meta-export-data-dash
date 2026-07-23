"""media derivatives

Revision ID: 54c2d92a6e31
Revises: 8c4af76413b2
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "54c2d92a6e31"
down_revision: str | Sequence[str] | None = "8c4af76413b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "media_derivatives",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("blob_sha256", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(length=80), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("output_sha256", sa.String(length=64), nullable=True),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["blob_sha256"], ["blobs.sha256"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "blob_sha256",
            "kind",
            "pipeline_version",
            name="uq_media_derivative_blob_kind_version",
        ),
        sa.UniqueConstraint("relative_path"),
    )
    op.create_index(
        "ix_media_derivatives_blob_status",
        "media_derivatives",
        ["blob_sha256", "status"],
        unique=False,
    )
    op.create_index(
        "ix_media_derivatives_kind_status",
        "media_derivatives",
        ["kind", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_media_derivatives_kind_status", table_name="media_derivatives")
    op.drop_index("ix_media_derivatives_blob_status", table_name="media_derivatives")
    op.drop_table("media_derivatives")
