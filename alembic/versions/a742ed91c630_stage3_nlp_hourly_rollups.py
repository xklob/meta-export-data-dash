"""stage3 nlp hourly rollups

Revision ID: a742ed91c630
Revises: e1a5b4c73d20
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a742ed91c630"
down_revision: str | Sequence[str] | None = "e1a5b4c73d20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("nlp_annotations") as batch_op:
        batch_op.drop_constraint("uq_nlp_annotation_versioned_target", type_="unique")
        batch_op.create_unique_constraint(
            "uq_nlp_annotation_versioned_target",
            [
                "target_type",
                "target_id",
                "task",
                "pipeline_version",
                "model_name",
                "model_revision",
                "input_sha256",
            ],
        )

    op.create_table(
        "nlp_hourly_rollups",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("hour_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.String(length=24), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=80), nullable=False),
        sa.Column("configuration_sha256", sa.String(length=64), nullable=False),
        sa.Column("configuration_json", sa.JSON(), nullable=False),
        sa.Column("predicted_label", sa.String(length=80), nullable=False),
        sa.Column("sample_count", sa.BigInteger(), nullable=False),
        sa.Column("confidence_sum", sa.Float(), nullable=False),
        sa.Column("probability_sums_json", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "task",
            "hour_start_ms",
            "direction",
            "pipeline_version",
            "model_name",
            "model_revision",
            "configuration_sha256",
            "predicted_label",
            name="uq_nlp_hourly_rollup_versioned_bucket",
        ),
    )
    op.create_index(
        "ix_nlp_hourly_rollups_active_hour",
        "nlp_hourly_rollups",
        ["active", "hour_start_ms"],
    )
    op.create_index(
        "ix_nlp_hourly_rollups_task_direction",
        "nlp_hourly_rollups",
        ["task", "direction"],
    )


def downgrade() -> None:
    op.drop_index("ix_nlp_hourly_rollups_task_direction", table_name="nlp_hourly_rollups")
    op.drop_index("ix_nlp_hourly_rollups_active_hour", table_name="nlp_hourly_rollups")
    op.drop_table("nlp_hourly_rollups")

    with op.batch_alter_table("nlp_annotations") as batch_op:
        batch_op.drop_constraint("uq_nlp_annotation_versioned_target", type_="unique")
        batch_op.create_unique_constraint(
            "uq_nlp_annotation_versioned_target",
            [
                "target_type",
                "target_id",
                "task",
                "pipeline_version",
                "model_revision",
                "input_sha256",
            ],
        )
