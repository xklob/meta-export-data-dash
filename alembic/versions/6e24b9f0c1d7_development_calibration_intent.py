"""tag direct-conflict development-calibration feedback

Revision ID: 6e24b9f0c1d7
Revises: 2d6f8a19c4e7
Create Date: 2026-07-22

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "6e24b9f0c1d7"
down_revision: str | Sequence[str] | None = "2d6f8a19c4e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ARGUMENT_QUERY_SHA256 = "b74f83c9353b55cd27a6083e8a136486e5be81e8d2f9e83576c931d2400ceaa1"
DEFAULT_SCOPE_SHA256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
DEVELOPMENT_INTENT = "argument_direct_conflict_development"
HIGH_INTENSITY_REVIEW_MARKER = "blind-high-intensity-direct-conflict-v1%"


def upgrade() -> None:
    # The first cohort is the literal user-specified query, identified only by its stable digest.
    # The second is the failed blind test that protocol moved wholesale into development data;
    # its existing internal note marker avoids embedding that private query in this migration.
    op.execute(
        sa.text(
            """
            UPDATE relevance_feedback
            SET evaluation_intent = :development_intent
            WHERE evaluation_split = 'validation'
              AND search_mode = 'semantic'
              AND filter_scope_sha256 = :default_scope_sha256
              AND filter_scope_json = '{}'
              AND evaluation_intent IS NULL
              AND (
                    query_sha256 = :argument_query_sha256
                    OR notes LIKE :high_intensity_review_marker
                  )
            """
        ).bindparams(
            development_intent=DEVELOPMENT_INTENT,
            default_scope_sha256=DEFAULT_SCOPE_SHA256,
            argument_query_sha256=ARGUMENT_QUERY_SHA256,
            high_intensity_review_marker=HIGH_INTENSITY_REVIEW_MARKER,
        )
    )


def downgrade() -> None:
    # Older application versions do not understand this intent value. Preserve the judgments and
    # split while returning their optional intent tag to the pre-migration state.
    op.execute(
        sa.text(
            """
            UPDATE relevance_feedback
            SET evaluation_intent = NULL
            WHERE evaluation_intent = :development_intent
            """
        ).bindparams(development_intent=DEVELOPMENT_INTENT)
    )
