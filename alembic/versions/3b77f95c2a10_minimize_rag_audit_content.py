"""minimize retained cloud RAG audit content

Revision ID: 3b77f95c2a10
Revises: c4f78a1d2e60
Create Date: 2026-07-20

"""

from collections.abc import Sequence

from alembic import op

revision: str = "3b77f95c2a10"
down_revision: str | Sequence[str] | None = "c4f78a1d2e60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The audit contract requires only the deterministic question hash and a safe outcome code.
    # Scrub legacy rows rather than leaving private questions/provider bodies indefinitely.
    op.execute("UPDATE rag_audits SET query_text = '' WHERE query_text <> ''")
    op.execute(
        "UPDATE rag_audits SET error_text = 'LegacyRagFailure' "
        "WHERE error_text IS NOT NULL AND error_text <> ''"
    )


def downgrade() -> None:
    # Privacy erasure is intentionally irreversible. The legacy non-null column remains present,
    # so older application code can still read the empty value after a schema downgrade.
    pass
