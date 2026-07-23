"""stage2 conversation identity corrections

Revision ID: b91e6c40d2a7
Revises: a742ed91c630
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b91e6c40d2a7"
down_revision: str | Sequence[str] | None = "a742ed91c630"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_corrections",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("before_json", sa.JSON(), nullable=False),
        sa.Column("after_json", sa.JSON(), nullable=False),
        sa.Column("catalog_revision", sa.String(length=64), nullable=False),
        sa.Column("result_revision", sa.String(length=64), nullable=False),
        sa.Column("revert_revision", sa.String(length=64), nullable=True),
        sa.Column("effect_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("reverted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_corrections_status_created",
        "conversation_corrections",
        ["status", "created_at"],
    )

    op.create_table(
        "conversation_correction_segments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("correction_id", sa.String(length=36), nullable=False),
        sa.Column("segment_id", sa.String(length=64), nullable=False),
        sa.Column("thread_path", sa.Text(), nullable=False),
        sa.Column("before_conversation_id", sa.String(length=64), nullable=False),
        sa.Column("after_conversation_id", sa.String(length=64), nullable=False),
        sa.Column("previous_correction_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(
            ["correction_id"], ["conversation_corrections.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "correction_id",
            "segment_id",
            name="uq_conversation_correction_segment",
        ),
    )
    op.create_index(
        "ix_conversation_correction_segments_segment",
        "conversation_correction_segments",
        ["segment_id", "correction_id"],
    )
    op.create_index(
        "ix_conversation_correction_segments_conversations",
        "conversation_correction_segments",
        ["before_conversation_id", "after_conversation_id"],
    )

    op.create_table(
        "conversation_correction_audits",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("correction_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("detail_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["correction_id"], ["conversation_corrections.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_correction_audits_correction_created",
        "conversation_correction_audits",
        ["correction_id", "created_at"],
    )

    op.create_table(
        "conversation_segment_overrides",
        sa.Column("segment_id", sa.String(length=64), nullable=False),
        sa.Column("base_conversation_id", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=False),
        sa.Column("correction_id", sa.String(length=36), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["segment_id"], ["conversation_segments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["correction_id"], ["conversation_corrections.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("segment_id"),
    )
    op.create_index(
        "ix_conversation_segment_overrides_conversation",
        "conversation_segment_overrides",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_segment_overrides_conversation",
        table_name="conversation_segment_overrides",
    )
    op.drop_table("conversation_segment_overrides")
    op.drop_index(
        "ix_conversation_correction_audits_correction_created",
        table_name="conversation_correction_audits",
    )
    op.drop_table("conversation_correction_audits")
    op.drop_index(
        "ix_conversation_correction_segments_conversations",
        table_name="conversation_correction_segments",
    )
    op.drop_index(
        "ix_conversation_correction_segments_segment",
        table_name="conversation_correction_segments",
    )
    op.drop_table("conversation_correction_segments")
    op.drop_index(
        "ix_conversation_corrections_status_created",
        table_name="conversation_corrections",
    )
    op.drop_table("conversation_corrections")
