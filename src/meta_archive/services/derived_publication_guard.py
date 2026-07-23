from __future__ import annotations

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from meta_archive.models import DeletionOperation

# A committed quarantine promises an exact restore of the derived products captured at commit.
# cleanup_pending is an irreversible purge whose filesystem and catalog finalization must remain
# isolated. A normalized-only purge retains a workspace upgrade receipt, so publishers also stay
# paused until that remaining purge is either completed or otherwise resolved.
DERIVED_PUBLICATION_BLOCKING_DELETION_STATUSES = frozenset(
    {"committed", "cleanup_pending", "normalized_purged"}
)


class DerivedPublicationBlockedError(RuntimeError):
    """A derived-data publisher cannot run while a deletion owns archive state."""

    def __init__(self, *, publisher: str, operation_id: str, status: str) -> None:
        super().__init__(
            f"{publisher} is unavailable while deletion {operation_id} is {status}; "
            "restore or finish hard-purging that deletion first"
        )
        self.publisher = publisher
        self.operation_id = operation_id
        self.status = status


def assert_derived_publication_allowed(engine: Engine, *, publisher: str) -> None:
    """Reject derived publication while exact restore or purge finalization is outstanding.

    The caller must hold the archive deletion lock and its own publisher lock for the whole
    mutation. Locking in that order makes this check atomic with deletion commit, restore, and
    hard purge: a deletion cannot cross the check while the publisher continues to write.
    """

    with Session(engine) as session:
        blocked = session.execute(
            select(DeletionOperation.id, DeletionOperation.status)
            .where(DeletionOperation.status.in_(DERIVED_PUBLICATION_BLOCKING_DELETION_STATUSES))
            .order_by(DeletionOperation.created_at, DeletionOperation.id)
            .limit(1)
        ).one_or_none()
    if blocked is not None:
        operation_id, status = blocked
        raise DerivedPublicationBlockedError(
            publisher=publisher,
            operation_id=str(operation_id),
            status=str(status),
        )
