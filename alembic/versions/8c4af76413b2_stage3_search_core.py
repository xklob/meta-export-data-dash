"""stage3 search core

Revision ID: 8c4af76413b2
Revises: fbe9b916ee6b
Create Date: 2026-07-20

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "8c4af76413b2"
down_revision: str | Sequence[str] | None = "fbe9b916ee6b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rag_audits",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("query_sha256", sa.String(length=64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("search_mode", sa.String(length=24), nullable=False),
        sa.Column("retrieved_ids_json", sa.JSON(), nullable=False),
        sa.Column("citation_ids_json", sa.JSON(), nullable=False),
        sa.Column("excerpt_hashes_json", sa.JSON(), nullable=False),
        sa.Column("remote_response_id", sa.String(length=255), nullable=True),
        sa.Column("usage_json", sa.JSON(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rag_audits_created_at", "rag_audits", ["created_at"], unique=False)

    op.create_table(
        "relevance_feedback",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("query_sha256", sa.String(length=64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("search_mode", sa.String(length=24), nullable=False),
        sa.Column("result_id", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=24), nullable=False),
        sa.Column("evaluation_split", sa.String(length=24), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_relevance_feedback_query_result",
        "relevance_feedback",
        ["query_sha256", "result_id"],
        unique=False,
    )

    op.create_table(
        "saved_searches",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("query_json", sa.JSON(), nullable=False),
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
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "search_windows",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("window_version", sa.String(length=40), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=64), nullable=True),
        sa.Column("segment_id", sa.String(length=64), nullable=True),
        sa.Column("activity_id", sa.String(length=64), nullable=True),
        sa.Column("first_message_id", sa.String(length=64), nullable=True),
        sa.Column("last_message_id", sa.String(length=64), nullable=True),
        sa.Column("first_timestamp_ms", sa.BigInteger(), nullable=True),
        sa.Column("last_timestamp_ms", sa.BigInteger(), nullable=True),
        sa.Column("turn_count", sa.Integer(), nullable=False),
        sa.Column("approx_token_count", sa.Integer(), nullable=False),
        sa.Column("text_display", sa.Text(), nullable=False),
        sa.Column("text_search", sa.Text(), nullable=False),
        sa.Column("source_category", sa.String(length=80), nullable=True),
        sa.Column("participants_search", sa.Text(), nullable=False),
        sa.Column("source_ids_json", sa.JSON(), nullable=False),
        sa.Column("build_generation", sa.String(length=36), nullable=False),
        sa.Column("embedding_config_hash", sa.String(length=64), nullable=True),
        sa.Column("embedding_updated_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["activity_id"], ["activity_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["segment_id"], ["conversation_segments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_search_windows_entity_time",
        "search_windows",
        ["entity_type", "first_timestamp_ms"],
        unique=False,
    )
    op.create_index(
        "ix_search_windows_conversation_time",
        "search_windows",
        ["conversation_id", "first_timestamp_ms"],
        unique=False,
    )
    op.create_index(
        "ix_search_windows_embedding",
        "search_windows",
        ["embedding_config_hash"],
        unique=False,
    )
    op.create_index(
        "ix_search_windows_generation",
        "search_windows",
        ["build_generation"],
        unique=False,
    )

    op.create_table(
        "search_window_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("window_id", sa.String(length=64), nullable=False),
        sa.Column("message_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["window_id"], ["search_windows.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("window_id", "message_id", name="uq_search_window_message"),
        sa.UniqueConstraint("window_id", "ordinal", name="uq_search_window_message_ordinal"),
    )
    op.create_index(
        "ix_search_window_messages_message",
        "search_window_messages",
        ["message_id"],
        unique=False,
    )

    op.create_table(
        "semantic_index_configs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("backend", sa.String(length=40), nullable=False),
        sa.Column("backend_version", sa.String(length=40), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_revision", sa.String(length=255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("distance_metric", sa.String(length=24), nullable=False),
        sa.Column("document_prefix", sa.String(length=80), nullable=False),
        sa.Column("query_prefix", sa.String(length=80), nullable=False),
        sa.Column("vector_table", sa.String(length=80), nullable=False),
        sa.Column("window_version", sa.String(length=40), nullable=False),
        sa.Column("window_count", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("audit_json", sa.JSON(), nullable=False),
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
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("vector_table"),
    )
    op.create_index(
        "ix_semantic_index_configs_active",
        "semantic_index_configs",
        ["active"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_semantic_index_configs_active", table_name="semantic_index_configs")
    op.drop_table("semantic_index_configs")
    op.drop_index("ix_search_window_messages_message", table_name="search_window_messages")
    op.drop_table("search_window_messages")
    op.drop_index("ix_search_windows_generation", table_name="search_windows")
    op.drop_index("ix_search_windows_embedding", table_name="search_windows")
    op.drop_index("ix_search_windows_conversation_time", table_name="search_windows")
    op.drop_index("ix_search_windows_entity_time", table_name="search_windows")
    op.drop_table("search_windows")
    op.drop_table("saved_searches")
    op.drop_index("ix_relevance_feedback_query_result", table_name="relevance_feedback")
    op.drop_table("relevance_feedback")
    op.drop_index("ix_rag_audits_created_at", table_name="rag_audits")
    op.drop_table("rag_audits")
