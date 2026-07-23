"""conflict aggregate product

Revision ID: 4e1a72c9bf30
Revises: d3b7f50a21c4
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "4e1a72c9bf30"
down_revision: str | Sequence[str] | None = "d3b7f50a21c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conflict_aggregate_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=80), nullable=False),
        sa.Column("calibration_id", sa.String(length=64), nullable=True),
        sa.Column("semantic_config_id", sa.String(length=64), nullable=True),
        sa.Column("corpus_revision", sa.String(length=64), nullable=False),
        sa.Column("source_revision", sa.String(length=64), nullable=False),
        sa.Column("configuration_sha256", sa.String(length=64), nullable=False),
        sa.Column("configuration_json", sa.JSON(), nullable=False),
        sa.Column("window_count", sa.BigInteger(), nullable=False),
        sa.Column("scored_count", sa.BigInteger(), nullable=False),
        sa.Column("direct_conflict_count", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["calibration_id"], ["conflict_calibrations.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["semantic_config_id"], ["semantic_index_configs.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("configuration_sha256", name="uq_conflict_aggregate_run_config"),
    )
    op.create_index(
        "ix_conflict_aggregate_runs_active_status",
        "conflict_aggregate_runs",
        ["active", "status"],
    )

    op.create_table(
        "conflict_window_scores",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("window_id", sa.String(length=64), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("first_timestamp_ms", sa.BigInteger(), nullable=True),
        sa.Column("predicted_label", sa.String(length=32), nullable=False),
        sa.Column("direct_conflict_score", sa.Float(), nullable=False),
        sa.Column("scores_json", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["conflict_aggregate_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["window_id"], ["search_windows.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "window_id", name="uq_conflict_score_run_window"),
    )
    op.create_index(
        "ix_conflict_window_scores_active_time",
        "conflict_window_scores",
        ["active", "first_timestamp_ms"],
    )
    op.create_index(
        "ix_conflict_window_scores_run_label",
        "conflict_window_scores",
        ["run_id", "predicted_label"],
    )

    op.create_table(
        "conflict_hourly_rollups",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("hour_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("sample_count", sa.BigInteger(), nullable=False),
        sa.Column("direct_conflict_count", sa.BigInteger(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["run_id"], ["conflict_aggregate_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "hour_start_ms", name="uq_conflict_rollup_run_hour"),
    )
    op.create_index(
        "ix_conflict_hourly_rollups_active_hour",
        "conflict_hourly_rollups",
        ["active", "hour_start_ms"],
    )


def downgrade() -> None:
    op.drop_index("ix_conflict_hourly_rollups_active_hour", table_name="conflict_hourly_rollups")
    op.drop_table("conflict_hourly_rollups")
    op.drop_index("ix_conflict_window_scores_run_label", table_name="conflict_window_scores")
    op.drop_index("ix_conflict_window_scores_active_time", table_name="conflict_window_scores")
    op.drop_table("conflict_window_scores")
    op.drop_index("ix_conflict_aggregate_runs_active_status", table_name="conflict_aggregate_runs")
    op.drop_table("conflict_aggregate_runs")
