"""index versioned NLP inference results for exact-input reuse

Revision ID: 6f3c9e82a1b4
Revises: 4e1a72c9bf30
Create Date: 2026-07-20

"""

from collections.abc import Sequence

from alembic import op

revision: str = "6f3c9e82a1b4"
down_revision: str | Sequence[str] | None = "4e1a72c9bf30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_nlp_annotations_inference_reuse",
        "nlp_annotations",
        [
            "task",
            "pipeline_version",
            "model_name",
            "model_revision",
            "input_sha256",
            "status",
        ],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_nlp_annotations_inference_reuse", table_name="nlp_annotations")
