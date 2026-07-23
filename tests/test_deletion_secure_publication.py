from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from meta_archive.services.deletion_secure_publication import (
    SealedStagingTree,
    SecurePublicationError,
    StagingPreparation,
    prepare_staging_directories,
    publish_private_file,
    read_private_file,
    recover_private_publication,
    remove_sealed_staging_tree,
    seal_staging_tree,
)

MANIFEST_RELATIVE = "trash/deletions/operation-id/manifest.json"
STAGING_RELATIVE = "trash/purge-staging/operation-id"


class InjectedCrash(BaseException):
    pass


class CrashAt:
    def __init__(self, event_prefix: str) -> None:
        self.event_prefix = event_prefix
        self.triggered = False

    def __call__(self, event: str) -> None:
        if not self.triggered and event.startswith(self.event_prefix):
            self.triggered = True
            raise InjectedCrash(event)


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir(mode=0o700)
    return root


def _manifest(root: Path) -> Path:
    return root / MANIFEST_RELATIVE


def _wait_and_replace_with_hardlink(
    ready: multiprocessing.synchronize.Event,
    proceed: multiprocessing.synchronize.Event,
    target: str,
    external: str,
) -> None:
    ready.set()
    if not proceed.wait(timeout=10):
        return
    target_path = Path(target)
    target_path.unlink(missing_ok=True)
    os.link(external, target)


def _wait_and_redirect_directory(
    ready: multiprocessing.synchronize.Event,
    proceed: multiprocessing.synchronize.Event,
    target: str,
    saved: str,
    external: str,
) -> None:
    ready.set()
    if not proceed.wait(timeout=10):
        return
    os.rename(target, saved)
    os.symlink(external, target, target_is_directory=True)


def _process_fault(
    event_name: str,
    proceed: multiprocessing.synchronize.Event,
    process: multiprocessing.Process,
) -> Callable[[str], None]:
    def inject(event: str) -> None:
        if event == event_name:
            proceed.set()
            process.join(timeout=10)
            assert process.exitcode == 0

    return inject


def _publish_and_exit_at_boundary(
    root: str,
    relative_path: str,
    payload: bytes,
    expected: object,
    event_prefix: str,
    occurrence: int,
) -> None:
    seen = 0

    def terminate(event: str) -> None:
        nonlocal seen
        if event.startswith(event_prefix):
            seen += 1
            if seen == occurrence:
                os._exit(91)

    publish_private_file(
        Path(root),
        relative_path,
        payload,
        expected_existing=expected,  # type: ignore[arg-type]
        fault_injector=terminate,
    )


def test_first_publication_is_private_exact_and_durable(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    payload = b'{"manifest_version":2,"items":[]}\n'
    events: list[str] = []

    receipt = publish_private_file(
        root,
        MANIFEST_RELATIVE,
        payload,
        expected_existing=None,
        fault_injector=events.append,
    )

    target = _manifest(root)
    assert target.read_bytes() == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert target.stat().st_uid == os.geteuid()
    assert target.stat().st_nlink == 1
    assert receipt.identity.sha256
    assert receipt.identity.inode == target.stat().st_ino
    for directory in (root / "trash", root / "trash/deletions", target.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert directory.stat().st_uid == os.geteuid()

    assert any(event.startswith("before_write:") for event in events)
    assert any(event.startswith("after_write:") for event in events)
    assert f"before_link:{target.name}" in events
    assert f"after_link:{target.name}" in events
    assert "before_fsync:anonymous-file:manifest.json" in events
    assert "after_fsync:anonymous-file:manifest.json" in events
    assert not any(path.name.startswith(".publication-") for path in target.parent.iterdir())
    for label in ("directory:.", "directory:trash", "directory:trash/deletions"):
        assert f"before_fsync:{label}" in events
        assert f"after_fsync:{label}" in events


def test_replacement_is_bound_to_prior_inode_and_exact_bytes(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    first = b'{"status":"previewed","snapshot":{"message_count":1}}\n'
    second = b'{"status":"committed","snapshot":{"message_count":1}}\n'
    publish_private_file(root, MANIFEST_RELATIVE, first, expected_existing=None)
    expected = read_private_file(root, MANIFEST_RELATIVE)
    assert expected is not None
    events: list[str] = []

    receipt = publish_private_file(
        root,
        MANIFEST_RELATIVE,
        second,
        expected_existing=expected,
        fault_injector=events.append,
    )

    target = _manifest(root)
    assert target.read_bytes() == second
    assert target.stat().st_ino == receipt.identity.inode
    assert target.stat().st_ino != expected.identity.inode
    assert target.stat().st_nlink == 1
    assert f"before_rename_exchange:{target.name}" in events
    assert f"after_rename_exchange:{target.name}" in events
    assert not any(
        path.stat().st_ino == expected.identity.inode for path in target.parent.iterdir()
    )


@pytest.mark.parametrize(
    "event_prefix",
    [
        "before_write:",
        "after_write:",
        "before_chmod:manifest.json",
        "after_chmod:manifest.json",
        "before_fsync:anonymous-file:manifest.json",
        "after_fsync:anonymous-file:manifest.json",
        "before_link:manifest.json",
        "after_link:manifest.json",
        "before_fsync:directory:trash/deletions/operation-id:link",
        "after_fsync:directory:trash/deletions/operation-id:link",
        "before_fsync:file:trash/deletions/operation-id/manifest.json",
        "after_fsync:file:trash/deletions/operation-id/manifest.json",
        "before_fsync:directory:trash/deletions/operation-id:verified-target",
        "after_fsync:directory:trash/deletions/operation-id:verified-target",
    ],
)
def test_first_publication_recovers_idempotently_from_every_io_boundary(
    tmp_path: Path, event_prefix: str
) -> None:
    root = _private_root(tmp_path)
    payload = b'{"status":"previewed","items":[]}\n'
    crash = CrashAt(event_prefix)

    with pytest.raises(InjectedCrash):
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            payload,
            expected_existing=None,
            fault_injector=crash,
        )
    assert crash.triggered

    publish_private_file(root, MANIFEST_RELATIVE, payload, expected_existing=None)
    target = _manifest(root)
    assert target.read_bytes() == payload
    assert target.stat().st_nlink == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o400


@pytest.mark.parametrize(
    "event_prefix",
    [
        "before_write:",
        "after_write:",
        "before_fsync:anonymous-file:manifest.json",
        "after_fsync:anonymous-file:manifest.json",
        "before_fsync:anonymous-file:.publication-",
        "after_fsync:anonymous-file:.publication-",
        "before_link:.publication-",
        "after_link:.publication-",
        "before_fsync:directory:trash/deletions/operation-id:marker",
        "after_fsync:directory:trash/deletions/operation-id:marker",
        "before_fsync:directory:trash/deletions/operation-id:stage",
        "after_fsync:directory:trash/deletions/operation-id:stage",
        "before_rename_exchange:manifest.json",
        "after_rename_exchange:manifest.json",
        "before_fsync:directory:trash/deletions/operation-id:replay-target",
        "after_fsync:directory:trash/deletions/operation-id:replay-target",
        "before_unlink:.publication-",
        "after_unlink:.publication-",
        "before_fsync:directory:trash/deletions/operation-id:replay-stage-cleanup",
        "after_fsync:directory:trash/deletions/operation-id:replay-stage-cleanup",
        "before_fsync:directory:trash/deletions/operation-id:replay-marker-cleanup",
        "after_fsync:directory:trash/deletions/operation-id:replay-marker-cleanup",
        "before_fsync:file:trash/deletions/operation-id/manifest.json",
        "after_fsync:file:trash/deletions/operation-id/manifest.json",
        "before_fsync:directory:trash/deletions/operation-id:verified-target",
        "after_fsync:directory:trash/deletions/operation-id:verified-target",
    ],
)
def test_replacement_recovers_idempotently_from_every_io_boundary(
    tmp_path: Path, event_prefix: str
) -> None:
    root = _private_root(tmp_path)
    original = b'{"status":"previewed","items":[]}\n'
    replacement = b'{"status":"published","items":[]}\n'
    publish_private_file(root, MANIFEST_RELATIVE, original, expected_existing=None)
    expected = read_private_file(root, MANIFEST_RELATIVE)
    assert expected is not None
    crash = CrashAt(event_prefix)

    with pytest.raises(InjectedCrash):
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            replacement,
            expected_existing=expected,
            fault_injector=crash,
        )
    assert crash.triggered

    current = read_private_file(root, MANIFEST_RELATIVE)
    if current is None or current.payload != replacement:
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            replacement,
            expected_existing=current,
        )
    else:
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            replacement,
            expected_existing=expected,
        )
    target = _manifest(root)
    assert target.read_bytes() == replacement
    assert target.stat().st_nlink == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o400


@pytest.mark.parametrize(
    ("event_prefix", "occurrence"),
    [
        ("after_link:.publication-", 1),
        ("after_fsync:directory:trash/deletions/operation-id:marker", 1),
        ("after_link:.publication-", 2),
        ("after_fsync:directory:trash/deletions/operation-id:stage", 1),
        ("after_rename_exchange:manifest.json", 1),
        ("after_fsync:directory:trash/deletions/operation-id:replay-target", 1),
        ("after_unlink:.publication-", 1),
        ("after_fsync:directory:trash/deletions/operation-id:replay-stage-cleanup", 1),
        ("after_unlink:.publication-", 2),
    ],
)
def test_real_process_death_replay_removes_every_replacement_artifact(
    tmp_path: Path,
    event_prefix: str,
    occurrence: int,
) -> None:
    root = _private_root(tmp_path)
    original = b'{"snapshot":"OLD_CANARY_SECRET"}\n'
    replacement = b'{"status":"NEW_STATE"}\n'
    publish_private_file(root, MANIFEST_RELATIVE, original, expected_existing=None)
    expected = read_private_file(root, MANIFEST_RELATIVE)
    assert expected is not None
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_publish_and_exit_at_boundary,
        args=(
            str(root),
            MANIFEST_RELATIVE,
            replacement,
            expected,
            event_prefix,
            occurrence,
        ),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91

    replay_events: list[str] = []
    publish_private_file(
        root,
        MANIFEST_RELATIVE,
        replacement,
        expected_existing=expected,
        fault_injector=replay_events.append,
    )

    target = _manifest(root)
    assert target.read_bytes() == replacement
    assert [path.name for path in target.parent.iterdir()] == ["manifest.json"]
    assert not any(b"OLD_CANARY_SECRET" in path.read_bytes() for path in target.parent.iterdir())
    assert any(
        event.startswith("after_fsync:directory:trash/deletions/operation-id:")
        for event in replay_events
    )


@pytest.mark.parametrize(
    ("event_prefix", "occurrence"),
    [
        ("after_link:.publication-", 1),
        ("after_fsync:directory:trash/deletions/operation-id:marker", 1),
        ("after_link:.publication-", 2),
        ("after_fsync:directory:trash/deletions/operation-id:stage", 1),
        ("after_rename_exchange:manifest.json", 1),
        ("after_fsync:directory:trash/deletions/operation-id:replay-target", 1),
        ("after_unlink:.publication-", 1),
        ("after_fsync:directory:trash/deletions/operation-id:replay-stage-cleanup", 1),
        ("after_unlink:.publication-", 2),
    ],
)
def test_real_process_death_can_recover_then_publish_a_different_payload(
    tmp_path: Path,
    event_prefix: str,
    occurrence: int,
) -> None:
    root = _private_root(tmp_path)
    original = b'{"snapshot":"ORIGINAL_CANARY"}\n'
    abandoned = b'{"status":"ABANDONED_ATTEMPT"}\n'
    final = b'{"status":"FINAL_STATE"}\n'
    publish_private_file(root, MANIFEST_RELATIVE, original, expected_existing=None)
    expected = read_private_file(root, MANIFEST_RELATIVE)
    assert expected is not None
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_publish_and_exit_at_boundary,
        args=(
            str(root),
            MANIFEST_RELATIVE,
            abandoned,
            expected,
            event_prefix,
            occurrence,
        ),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91

    recover_private_publication(root, MANIFEST_RELATIVE)
    recovered = read_private_file(root, MANIFEST_RELATIVE)
    assert recovered is not None
    receipt = publish_private_file(
        root,
        MANIFEST_RELATIVE,
        final,
        expected_existing=recovered,
    )

    target = _manifest(root)
    assert target.read_bytes() == final
    assert target.stat().st_ino == receipt.identity.inode
    assert target.stat().st_nlink == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o400
    assert [path.name for path in target.parent.iterdir()] == ["manifest.json"]
    assert b"ORIGINAL_CANARY" not in target.read_bytes()
    assert b"ABANDONED_ATTEMPT" not in target.read_bytes()


def test_real_process_death_after_first_link_replays_with_no_named_stage(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    payload = b'{"status":"FIRST"}\n'
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_publish_and_exit_at_boundary,
        args=(root, MANIFEST_RELATIVE, payload, None, "after_link:manifest.json", 1),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91

    events: list[str] = []
    publish_private_file(
        root,
        MANIFEST_RELATIVE,
        payload,
        expected_existing=None,
        fault_injector=events.append,
    )

    assert _manifest(root).read_bytes() == payload
    assert [path.name for path in _manifest(root).parent.iterdir()] == ["manifest.json"]
    assert "after_fsync:directory:trash/deletions/operation-id:verified-target" in events


def test_real_process_death_after_last_mkdir_replay_fsyncs_full_ancestor_chain(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    payload = b'{"status":"FIRST"}\n'
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_publish_and_exit_at_boundary,
        args=(
            root,
            MANIFEST_RELATIVE,
            payload,
            None,
            "after_mkdir:trash/deletions/operation-id",
            1,
        ),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 91
    assert _manifest(root).parent.is_dir()
    events: list[str] = []

    publish_private_file(
        root,
        MANIFEST_RELATIVE,
        payload,
        expected_existing=None,
        fault_injector=events.append,
    )

    for label in (
        "directory:trash/deletions/operation-id",
        "directory:trash/deletions",
        "directory:trash",
        "directory:.",
    ):
        assert f"before_fsync:{label}" in events
        assert f"after_fsync:{label}" in events


def test_publication_rejects_intermediate_and_leaf_symlinks(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    (root / "trash").symlink_to(external, target_is_directory=True)

    with pytest.raises(SecurePublicationError):
        publish_private_file(root, MANIFEST_RELATIVE, b"safe\n", expected_existing=None)
    assert list(external.iterdir()) == []

    (root / "trash").unlink()
    target = _manifest(root)
    target.parent.mkdir(parents=True, mode=0o700)
    (root / "trash").chmod(0o700)
    (root / "trash/deletions").chmod(0o700)
    target.parent.chmod(0o700)
    outside = external / "outside.json"
    outside.write_bytes(b"outside")
    outside.chmod(0o400)
    target.symlink_to(outside)

    with pytest.raises(SecurePublicationError):
        publish_private_file(root, MANIFEST_RELATIVE, b"safe\n", expected_existing=None)
    assert outside.read_bytes() == b"outside"


def test_publication_rejects_leaf_hardlink_without_unlinking_source(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    target = _manifest(root)
    target.parent.mkdir(parents=True, mode=0o700)
    for directory in (root / "trash", root / "trash/deletions", target.parent):
        directory.chmod(0o700)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    outside.chmod(0o400)
    os.link(outside, target)

    with pytest.raises(SecurePublicationError):
        publish_private_file(root, MANIFEST_RELATIVE, b"safe\n", expected_existing=None)
    assert outside.read_bytes() == b"outside"
    assert outside.stat().st_nlink == 2
    assert target.samefile(outside)


@pytest.mark.parametrize("unsafe_level", ["root", "intermediate"])
def test_rejected_unsafe_directory_does_not_leak_descriptors(
    tmp_path: Path,
    unsafe_level: str,
) -> None:
    root = _private_root(tmp_path)
    if unsafe_level == "root":
        root.chmod(0o755)
    else:
        (root / "trash").mkdir(mode=0o755)
        (root / "trash").chmod(0o755)
    descriptor_count = len(os.listdir("/proc/self/fd"))

    for _attempt in range(20):
        with pytest.raises(SecurePublicationError):
            publish_private_file(root, MANIFEST_RELATIVE, b"safe\n", expected_existing=None)

    assert len(os.listdir("/proc/self/fd")) == descriptor_count


def test_publication_rejects_unbound_recovery_artifact_without_removing_it(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    publish_private_file(root, MANIFEST_RELATIVE, b"old", expected_existing=None)
    expected = read_private_file(root, MANIFEST_RELATIVE)
    assert expected is not None
    parent = _manifest(root).parent
    path_key = hashlib.sha256(b"manifest.json").hexdigest()[:12]
    unbound = parent / f".publication-{path_key}-deadbeefdeadbeefdeadbeef.data"
    unbound.write_bytes(b"attacker")
    unbound.chmod(0o400)

    with pytest.raises(SecurePublicationError, match="Unbound"):
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            b"new",
            expected_existing=expected,
        )
    assert unbound.read_bytes() == b"attacker"


def test_cross_process_leaf_substitution_is_rejected_without_deleting_attacker_file(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    original = b'{"status":"previewed"}\n'
    publish_private_file(root, MANIFEST_RELATIVE, original, expected_existing=None)
    expected = read_private_file(root, MANIFEST_RELATIVE)
    assert expected is not None
    external = tmp_path / "attacker-owned"
    external.write_bytes(b"do-not-delete")
    external.chmod(0o400)
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    proceed = context.Event()
    process = context.Process(
        target=_wait_and_replace_with_hardlink,
        args=(ready, proceed, str(_manifest(root)), str(external)),
    )
    process.start()
    assert ready.wait(timeout=10)

    with pytest.raises(SecurePublicationError):
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            b'{"status":"published"}\n',
            expected_existing=expected,
            fault_injector=_process_fault("before_rename_exchange:manifest.json", proceed, process),
        )
    process.join(timeout=10)
    assert process.exitcode == 0
    assert external.read_bytes() == b"do-not-delete"
    assert _manifest(root).read_bytes() == b'{"status":"published"}\n'
    attacker_links = [
        path
        for path in _manifest(root).parent.iterdir()
        if path.name.endswith(".data") and path.samefile(external)
    ]
    assert len(attacker_links) == 1


def test_cross_process_parent_redirect_is_rejected_without_writing_external_tree(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    target_parent = _manifest(root).parent
    target_parent.mkdir(parents=True, mode=0o700)
    for directory in (root / "trash", root / "trash/deletions", target_parent):
        directory.chmod(0o700)
    saved = target_parent.with_name("operation-id-saved")
    external = tmp_path / "external-directory"
    external.mkdir(mode=0o700)
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    proceed = context.Event()
    process = context.Process(
        target=_wait_and_redirect_directory,
        args=(ready, proceed, str(target_parent), str(saved), str(external)),
    )
    process.start()
    assert ready.wait(timeout=10)

    with pytest.raises(SecurePublicationError):
        publish_private_file(
            root,
            MANIFEST_RELATIVE,
            b"safe\n",
            expected_existing=None,
            fault_injector=_process_fault("before_link:manifest.json", proceed, process),
        )
    process.join(timeout=10)
    assert process.exitcode == 0
    assert list(external.iterdir()) == []
    assert target_parent.is_symlink()


def _prepared_staging(root: Path) -> tuple[Path, StagingPreparation]:
    destinations = [
        f"{STAGING_RELATIVE}/catalog/nested/one.bin",
        f"{STAGING_RELATIVE}/blobs/sha256/aa/two.bin",
    ]
    preparation = prepare_staging_directories(root, STAGING_RELATIVE, destinations)
    staging = root / STAGING_RELATIVE
    first = staging / "catalog/nested/one.bin"
    second = staging / "blobs/sha256/aa/two.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    first.chmod(0o400)
    second.chmod(0o400)
    return staging, preparation


def test_nested_staging_is_precreated_then_sealed_bottom_up(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    destinations = [f"{STAGING_RELATIVE}/catalog/nested/one.bin"]
    prepare_events: list[str] = []
    preparation = prepare_staging_directories(
        root,
        STAGING_RELATIVE,
        destinations,
        fault_injector=prepare_events.append,
    )
    nested = root / STAGING_RELATIVE / "catalog/nested"
    assert nested.is_dir()
    for path in (
        root / "trash",
        root / "trash/purge-staging",
        root / STAGING_RELATIVE,
        root / STAGING_RELATIVE / "catalog",
        nested,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert any(event == "before_fsync:directory:." for event in prepare_events)
    assert any(event == "after_fsync:directory:trash" for event in prepare_events)

    target = nested / "one.bin"
    target.write_bytes(b"one")
    target.chmod(0o400)
    seal_events: list[str] = []
    receipt = seal_staging_tree(root, preparation, fault_injector=seal_events.append)
    file_after = seal_events.index("after_fsync:staging-file:catalog/nested/one.bin")
    nested_before = seal_events.index("before_fsync:staging-directory:catalog/nested")
    catalog_before = seal_events.index("before_fsync:staging-directory:catalog")
    root_before = seal_events.index("before_fsync:staging-directory:.")
    assert file_after < nested_before < catalog_before < root_before
    assert SealedStagingTree.from_json(receipt.to_json()) == receipt


def test_seal_rejects_a_substituted_nested_directory(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    staging, preparation = _prepared_staging(root)
    nested = staging / "catalog/nested"
    saved = staging / "catalog/nested-saved"
    external = tmp_path / "external-staging"
    external.mkdir(mode=0o700)
    nested.rename(saved)
    nested.symlink_to(external, target_is_directory=True)

    with pytest.raises(SecurePublicationError):
        seal_staging_tree(root, preparation)
    assert list(external.iterdir()) == []


def test_seal_rejects_a_same_owner_regular_nested_directory_replacement(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    staging, preparation = _prepared_staging(root)
    nested = staging / "catalog/nested"
    saved = staging / "catalog/nested-saved"
    nested.rename(saved)
    nested.mkdir(mode=0o700)
    replacement = nested / "one.bin"
    replacement.write_bytes(b"one")
    replacement.chmod(0o400)

    with pytest.raises(SecurePublicationError, match="Prepared staging directory was substituted"):
        seal_staging_tree(root, preparation)


def test_seal_rejects_a_file_with_any_link_outside_staging(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    destination = f"{STAGING_RELATIVE}/catalog/item.bin"
    preparation = prepare_staging_directories(root, STAGING_RELATIVE, [destination])
    outside = tmp_path / "outside-hardlink"
    outside.write_bytes(b"retain")
    outside.chmod(0o400)
    staged = root / destination
    os.link(outside, staged)

    with pytest.raises(SecurePublicationError, match="link outside"):
        seal_staging_tree(root, preparation)
    assert outside.read_bytes() == b"retain"
    assert outside.stat().st_nlink == 2


def test_seal_and_cleanup_allow_only_fully_internal_hardlink_topology(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    destinations = [
        f"{STAGING_RELATIVE}/catalog/one.bin",
        f"{STAGING_RELATIVE}/catalog/two.bin",
    ]
    preparation = prepare_staging_directories(root, STAGING_RELATIVE, destinations)
    first, second = (root / path for path in destinations)
    first.write_bytes(b"internal")
    first.chmod(0o400)
    os.link(first, second)
    receipt = seal_staging_tree(root, preparation)

    assert remove_sealed_staging_tree(root, receipt) is True
    assert not (root / STAGING_RELATIVE).exists()


def test_sealed_cleanup_rejects_unknown_entries_before_removing_known_data(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    staging, preparation = _prepared_staging(root)
    receipt = seal_staging_tree(root, preparation)
    expected = staging / "catalog/nested/one.bin"
    unexpected = staging / "unexpected"
    unexpected.write_bytes(b"attacker")
    unexpected.chmod(0o400)

    with pytest.raises(SecurePublicationError):
        remove_sealed_staging_tree(root, receipt)
    assert expected.read_bytes() == b"one"
    assert unexpected.read_bytes() == b"attacker"


def test_cross_process_cleanup_substitution_never_unlinks_attacker_inode(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    staging, preparation = _prepared_staging(root)
    receipt = seal_staging_tree(root, preparation)
    target = staging / "blobs/sha256/aa/two.bin"
    external = tmp_path / "attacker-cleanup"
    external.write_bytes(b"keep")
    external.chmod(0o400)
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    proceed = context.Event()
    process = context.Process(
        target=_wait_and_replace_with_hardlink,
        args=(ready, proceed, str(target), str(external)),
    )
    process.start()
    assert ready.wait(timeout=10)

    with pytest.raises(SecurePublicationError):
        remove_sealed_staging_tree(
            root,
            receipt,
            fault_injector=_process_fault(
                "before_unlink_staging:blobs/sha256/aa/two.bin",
                proceed,
                process,
            ),
        )
    process.join(timeout=10)
    assert process.exitcode == 0
    assert external.read_bytes() == b"keep"
    assert target.samefile(external)


def test_cleanup_replays_after_crash_between_unlink_and_directory_fsync(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    staging, preparation = _prepared_staging(root)
    receipt = seal_staging_tree(root, preparation)
    crash = CrashAt("after_unlink_staging:blobs/sha256/aa/two.bin")

    with pytest.raises(InjectedCrash):
        remove_sealed_staging_tree(root, receipt, fault_injector=crash)
    assert crash.triggered
    assert staging.is_dir()

    assert remove_sealed_staging_tree(root, receipt) is True
    assert not staging.exists()
    assert remove_sealed_staging_tree(root, receipt) is False


def test_cleanup_replay_fsyncs_surviving_ancestry_after_root_rmdir_crash(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    staging, preparation = _prepared_staging(root)
    receipt = seal_staging_tree(root, preparation)
    crash = CrashAt("after_rmdir_staging:.")

    with pytest.raises(InjectedCrash):
        remove_sealed_staging_tree(root, receipt, fault_injector=crash)
    assert not staging.exists()
    events: list[str] = []

    assert remove_sealed_staging_tree(root, receipt, fault_injector=events.append) is False
    assert "before_fsync:cleanup-absence:trash/purge-staging" in events
    assert "after_fsync:cleanup-absence:trash" in events
    assert "after_fsync:cleanup-absence:." in events
