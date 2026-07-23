"""stage3 local enrichment

Revision ID: c318f9a72b6d
Revises: 54c2d92a6e31
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c318f9a72b6d"
down_revision: str | Sequence[str] | None = "54c2d92a6e31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audio_transcriptions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("blob_sha256", sa.String(length=64), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=80), nullable=False),
        sa.Column("device", sa.String(length=32), nullable=False),
        sa.Column("compute_type", sa.String(length=32), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("language_probability", sa.Float(), nullable=True),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("text_display", sa.Text(), nullable=True),
        sa.Column("text_search", sa.Text(), nullable=True),
        sa.Column("segment_count", sa.Integer(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("index_eligible", sa.Boolean(), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "blob_sha256",
            "pipeline_version",
            "model_revision",
            name="uq_audio_transcription_blob_pipeline_model",
        ),
    )
    op.create_index("ix_audio_transcriptions_status", "audio_transcriptions", ["status"])
    op.create_index("ix_audio_transcriptions_blob", "audio_transcriptions", ["blob_sha256"])

    op.create_table(
        "audio_transcript_segments",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("transcription_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=False),
        sa.Column("end_ms", sa.BigInteger(), nullable=False),
        sa.Column("text_display", sa.Text(), nullable=False),
        sa.Column("text_search", sa.Text(), nullable=False),
        sa.Column("avg_log_probability", sa.Float(), nullable=True),
        sa.Column("no_speech_probability", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("words_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["transcription_id"], ["audio_transcriptions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transcription_id", "ordinal", name="uq_audio_segment_ordinal"),
    )
    op.create_index(
        "ix_audio_segments_transcription_time",
        "audio_transcript_segments",
        ["transcription_id", "start_ms"],
    )

    op.create_table(
        "attachment_text_extractions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("blob_sha256", sa.String(length=64), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("extractor_name", sa.String(length=80), nullable=False),
        sa.Column("extractor_revision", sa.String(length=80), nullable=False),
        sa.Column("filename", sa.Text(), nullable=True),
        sa.Column("media_type", sa.String(length=255), nullable=True),
        sa.Column("text_display", sa.Text(), nullable=True),
        sa.Column("text_search", sa.Text(), nullable=True),
        sa.Column("character_count", sa.BigInteger(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
        sa.Column("index_eligible", sa.Boolean(), nullable=False),
        sa.Column("sandboxed", sa.Boolean(), nullable=False),
        sa.Column("sandbox_status", sa.String(length=40), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("limits_json", sa.JSON(), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "blob_sha256", "pipeline_version", name="uq_attachment_text_blob_pipeline"
        ),
    )
    op.create_index("ix_attachment_text_status", "attachment_text_extractions", ["status"])
    op.create_index("ix_attachment_text_blob", "attachment_text_extractions", ["blob_sha256"])

    op.create_table(
        "nlp_annotations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=80), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("predicted_label", sa.String(length=80), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("probabilities_json", sa.JSON(), nullable=False),
        sa.Column("attribution_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "target_type",
            "target_id",
            "task",
            "pipeline_version",
            "model_revision",
            "input_sha256",
            name="uq_nlp_annotation_versioned_target",
        ),
    )
    op.create_index("ix_nlp_annotations_task_label", "nlp_annotations", ["task", "predicted_label"])
    op.create_index("ix_nlp_annotations_target", "nlp_annotations", ["target_type", "target_id"])
    op.create_index("ix_nlp_annotations_active", "nlp_annotations", ["active"])

    op.create_table(
        "conflict_labels",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("window_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("evaluation_split", sa.String(length=24), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["window_id"], ["search_windows.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conflict_labels_window_split", "conflict_labels", ["window_id", "evaluation_split"]
    )

    op.create_table(
        "conflict_calibrations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("direct_conflict_precision", sa.Float(), nullable=True),
        sa.Column("macro_f1", sa.Float(), nullable=True),
        sa.Column("aggregation_enabled", sa.Boolean(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("provenance_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )

    with op.batch_alter_table("search_windows") as batch_op:
        batch_op.add_column(sa.Column("audio_segment_id", sa.String(length=64), nullable=True))
        batch_op.add_column(
            sa.Column("attachment_extraction_id", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("attachment_occurrence_id", sa.String(length=64), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_search_windows_audio_segment",
            "audio_transcript_segments",
            ["audio_segment_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_search_windows_attachment_extraction",
            "attachment_text_extractions",
            ["attachment_extraction_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_foreign_key(
            "fk_search_windows_attachment_occurrence",
            "attachment_occurrences",
            ["attachment_occurrence_id"],
            ["id"],
            ondelete="CASCADE",
        )

    op.execute(
        """
        CREATE VIRTUAL TABLE derived_text_fts USING fts5(
            entity_kind UNINDEXED,
            source_id UNINDEXED,
            text_search,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS derived_text_fts")
    with op.batch_alter_table("search_windows") as batch_op:
        batch_op.drop_constraint("fk_search_windows_attachment_occurrence", type_="foreignkey")
        batch_op.drop_constraint("fk_search_windows_attachment_extraction", type_="foreignkey")
        batch_op.drop_constraint("fk_search_windows_audio_segment", type_="foreignkey")
        batch_op.drop_column("attachment_occurrence_id")
        batch_op.drop_column("attachment_extraction_id")
        batch_op.drop_column("audio_segment_id")
    op.drop_table("conflict_calibrations")
    op.drop_index("ix_conflict_labels_window_split", table_name="conflict_labels")
    op.drop_table("conflict_labels")
    op.drop_index("ix_nlp_annotations_active", table_name="nlp_annotations")
    op.drop_index("ix_nlp_annotations_target", table_name="nlp_annotations")
    op.drop_index("ix_nlp_annotations_task_label", table_name="nlp_annotations")
    op.drop_table("nlp_annotations")
    op.drop_index("ix_attachment_text_blob", table_name="attachment_text_extractions")
    op.drop_index("ix_attachment_text_status", table_name="attachment_text_extractions")
    op.drop_table("attachment_text_extractions")
    op.drop_index("ix_audio_segments_transcription_time", table_name="audio_transcript_segments")
    op.drop_table("audio_transcript_segments")
    op.drop_index("ix_audio_transcriptions_blob", table_name="audio_transcriptions")
    op.drop_index("ix_audio_transcriptions_status", table_name="audio_transcriptions")
    op.drop_table("audio_transcriptions")
