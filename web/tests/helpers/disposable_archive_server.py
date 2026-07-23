from __future__ import annotations

import json
import secrets
import shutil
import signal
import socket
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from meta_archive.api.main import create_app
from meta_archive.database import create_catalog_engine, run_migrations
from meta_archive.paths import ArchiveLayout
from meta_archive.services.consolidation import build_blob_store
from meta_archive.services.inventory import scan_archive
from meta_archive.services.message_importer import import_messages

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MESSAGE_FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "messages"
READY_PREFIX = "META_ARCHIVE_DISPOSABLE_READY="
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _utc_ms(value: str) -> int:
    instant = datetime.fromisoformat(value).astimezone(UTC)
    if instant.microsecond % 1_000:
        raise ValueError("Destructive-test fixture timestamps must align to whole milliseconds")
    elapsed = instant - UNIX_EPOCH
    return elapsed.days * 86_400_000 + elapsed.seconds * 1_000 + elapsed.microseconds // 1_000


def _write_date_boundary_fixture(root: Path) -> None:
    """Add records immediately around one DST-shortened local deletion day."""

    fixture = (
        root
        / "package-a"
        / "your_facebook_activity"
        / "messages"
        / "archived_threads"
        / "date_boundary_fixture"
        / "message_1.json"
    )
    fixture.parent.mkdir(parents=True, exist_ok=True)
    messages = [
        ("2024-03-11T04:00:00+00:00", "Date boundary exclusive end retained"),
        ("2024-03-11T03:59:59.999000+00:00", "Date boundary final millisecond included"),
        ("2024-03-10T05:00:00+00:00", "Date boundary first millisecond included"),
        ("2024-03-10T04:59:59.999000+00:00", "Date boundary millisecond before retained"),
    ]
    payload = {
        "title": "Date Boundary Contact",
        "participants": [{"name": "Archive Owner"}, {"name": "Date Boundary Contact"}],
        "messages": [
            {
                "sender_name": "Date Boundary Contact",
                "timestamp_ms": _utc_ms(timestamp),
                "content": content,
                "is_geoblocked_for_viewer": False,
                "is_unsent_image_by_messenger_kid_parent": False,
            }
            for timestamp, content in messages
        ],
        "thread_path": "archived_threads/date_boundary_fixture",
        "is_still_participant": True,
        "magic_words": [],
    }
    fixture.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _CleanupAwareServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        # Current Uvicorn deliberately re-raises a captured signal after graceful shutdown. That
        # is correct for a normal service process, but it would bypass the outer temporary-folder
        # context. The helper owns its signals so cleanup completes before the process exits.
        yield


def _build_archive(root: Path) -> ArchiveLayout:
    """Build the destructive-test catalog in a newly allocated temporary directory only."""

    if root.exists() or not root.parent.name.startswith("meta-archive-destructive-e2e-"):
        raise RuntimeError("Refusing to initialize a non-disposable destructive-test archive")
    shutil.copytree(MESSAGE_FIXTURES, root)
    _write_date_boundary_fixture(root)
    layout = ArchiveLayout(root)
    layout.ensure_runtime_directories()
    run_migrations(layout.database)
    engine = create_catalog_engine(layout.database)
    try:
        scan_archive(layout, engine, hash_files=True)
        import_messages(layout, engine, owner_name="Archive Owner")
        build_blob_store(layout, engine)
    finally:
        engine.dispose()
    return layout


def main() -> None:
    # There is deliberately no archive-root argument. This helper can only create and mutate a
    # fresh TemporaryDirectory, so a test typo cannot point it at the real Facebook export.
    if len(sys.argv) != 1:
        raise SystemExit("Disposable destructive-test helper accepts no arguments")
    handled_signals = (signal.SIGINT, signal.SIGTERM)

    def interrupt_initialization(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt

    process_handlers = {
        handled: signal.signal(handled, interrupt_initialization) for handled in handled_signals
    }
    try:
        with tempfile.TemporaryDirectory(prefix="meta-archive-destructive-e2e-") as temporary:
            temporary_root = Path(temporary)
            layout = _build_archive(temporary_root / "meta-backup-disposable-stage5")
            access_token = secrets.token_urlsafe(32)
            application = create_app(layout.root, access_token=access_token)

            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = int(listener.getsockname()[1])
            print(
                READY_PREFIX
                + json.dumps(
                    {
                        "archive_root": str(layout.root),
                        "temporary_root": str(temporary_root),
                        "base_url": f"http://127.0.0.1:{port}",
                        "access_token": access_token,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            config = uvicorn.Config(
                application,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
                timeout_graceful_shutdown=3,
            )
            server = _CleanupAwareServer(config)
            initialization_handlers = {
                handled: signal.signal(handled, server.handle_exit) for handled in handled_signals
            }
            try:
                server.run(sockets=[listener])
            finally:
                for handled, original in initialization_handlers.items():
                    signal.signal(handled, original)
    finally:
        for handled, original in process_handlers.items():
            signal.signal(handled, original)


if __name__ == "__main__":
    main()
