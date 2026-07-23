from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from fastapi import HTTPException, status
from sqlalchemy import Engine

from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion_lock_lease import DeletionError, deletion_lock
from meta_archive.services.derived_publication_guard import (
    DerivedPublicationBlockedError,
    assert_derived_publication_allowed,
)


def _blocked_detail(exc: DerivedPublicationBlockedError) -> dict[str, str]:
    return {
        "code": "derived_publication_blocked",
        "message": (
            "Derived-data changes are unavailable while a deletion quarantine or purge "
            "is outstanding."
        ),
        "deletion_status": exc.status,
    }


@contextmanager
def guarded_derived_request(
    layout: ArchiveLayout,
    engine: Engine,
    *,
    publisher: str,
) -> Iterator[None]:
    """Serialize an API-side mutation with deletion and expose a stable blocked response."""

    try:
        with deletion_lock(layout):
            assert_derived_publication_allowed(engine, publisher=publisher)
            yield
    except DerivedPublicationBlockedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_blocked_detail(exc),
        ) from exc
    except DeletionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "derived_publication_busy",
                "message": (
                    "Derived-data changes are temporarily unavailable while deletion state "
                    "is changing."
                ),
            },
        ) from exc


def raise_publication_blocked(exc: DerivedPublicationBlockedError) -> None:
    """Translate a synchronous service guard into the public API error contract."""

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=_blocked_detail(exc),
    ) from exc


def raise_publication_busy(exc: DeletionError) -> None:
    """Translate deletion-lock contention into the stable API error contract."""

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "derived_publication_busy",
            "message": (
                "Derived-data changes are temporarily unavailable while deletion state is changing."
            ),
        },
    ) from exc


def safe_derived_status_refresh[Status](
    layout: ArchiveLayout,
    engine: Engine,
    *,
    publisher: str,
    refresh: bool,
    loader: Callable[[bool], Status],
) -> Status:
    """Run a mutating status refresh only when it is serialized with deletion."""

    if not refresh:
        return loader(False)
    try:
        with deletion_lock(layout):
            assert_derived_publication_allowed(engine, publisher=publisher)
            return loader(True)
    except (DerivedPublicationBlockedError, DeletionError):
        return loader(False)
