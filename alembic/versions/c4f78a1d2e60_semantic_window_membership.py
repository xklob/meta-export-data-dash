"""add per-config semantic window membership

Revision ID: c4f78a1d2e60
Revises: 9a71c5e2d4f8
Create Date: 2026-07-20 22:45:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c4f78a1d2e60"
down_revision: str | Sequence[str] | None = "9a71c5e2d4f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "semantic_window_embeddings",
        sa.Column("config_id", sa.String(length=64), nullable=False),
        sa.Column("window_id", sa.String(length=64), nullable=False),
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["config_id"], ["semantic_index_configs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["window_id"], ["search_windows.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("config_id", "window_id"),
    )
    op.create_index(
        "ix_semantic_window_embeddings_window",
        "semantic_window_embeddings",
        ["window_id"],
        unique=False,
    )
    # Preserve the last active/checkpointed membership represented by the legacy single-value
    # column. Future builds write the relation directly and retain memberships for older tables.
    op.execute(
        """
        INSERT OR IGNORE INTO semantic_window_embeddings (config_id, window_id, embedded_at)
        SELECT search_windows.embedding_config_hash,
               search_windows.id,
               COALESCE(search_windows.embedding_updated_at, CURRENT_TIMESTAMP)
        FROM search_windows
        JOIN semantic_index_configs
          ON semantic_index_configs.id = search_windows.embedding_config_hash
        WHERE search_windows.embedding_config_hash IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_semantic_window_embeddings_window",
        table_name="semantic_window_embeddings",
    )
    op.drop_table("semantic_window_embeddings")
