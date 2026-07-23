"""stage5 deletion operations

Revision ID: e1a5b4c73d20
Revises: c318f9a72b6d
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e1a5b4c73d20"
down_revision: str | Sequence[str] | None = "c318f9a72b6d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "deletion_operations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("selector_json", sa.JSON(), nullable=False),
        sa.Column("impact_json", sa.JSON(), nullable=False),
        sa.Column("catalog_revision", sa.String(length=64), nullable=False),
        sa.Column("selection_sha256", sa.String(length=64), nullable=False),
        sa.Column("commit_token_hash", sa.String(length=64), nullable=True),
        sa.Column("purge_token_hash", sa.String(length=64), nullable=True),
        sa.Column("purge_revision", sa.String(length=64), nullable=True),
        sa.Column("purge_plan_json", sa.JSON(), nullable=False),
        sa.Column("manifest_relative_path", sa.Text(), nullable=True),
        sa.Column("item_count", sa.BigInteger(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("restored_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_deletion_operations_status_created",
        "deletion_operations",
        ["status", "created_at"],
    )

    op.create_table(
        "deletion_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("operation_id", sa.String(length=36), nullable=False),
        sa.Column("entity_type", sa.String(length=24), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.String(length=40), nullable=False),
        sa.Column("snapshot_sha256", sa.String(length=64), nullable=False),
        sa.Column("previous_deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["operation_id"], ["deletion_operations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "operation_id",
            "entity_type",
            "entity_id",
            name="uq_deletion_item_operation_entity",
        ),
    )
    op.create_index("ix_deletion_items_entity", "deletion_items", ["entity_type", "entity_id"])

    op.create_table(
        "deletion_audits",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("operation_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=True),
        sa.Column("outcome", sa.String(length=24), nullable=False),
        sa.Column("detail_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["operation_id"], ["deletion_operations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_deletion_audits_operation_created",
        "deletion_audits",
        ["operation_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_deletion_audits_operation_created", table_name="deletion_audits")
    op.drop_table("deletion_audits")
    op.drop_index("ix_deletion_items_entity", table_name="deletion_items")
    op.drop_table("deletion_items")
    op.drop_index("ix_deletion_operations_status_created", table_name="deletion_operations")
    op.drop_table("deletion_operations")
