from __future__ import annotations

import base64
import hashlib
import os
import stat
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import orjson
import pytest

from meta_archive.paths import ArchiveLayout
from meta_archive.services.deletion_upgrade_state import (
    DeletionUpgradeStateError,
    DeletionUpgradeStateExistsError,
    SavedDeletionResources,
    WorkspaceUpgradeStateReceipt,
    load_workspace_upgrade_state,
    publish_workspace_upgrade_state,
)
from meta_archive.services.purge_rewrite import (
    BINARY_TOMBSTONE_BYTES,
    FingerprintUpdate,
    RawRewritePlan,
    RawSourceRewrite,
)

OPERATION_ID = str(uuid.UUID("11111111-2222-3333-4444-555555555555"))
SELECTION_SHA256 = "a" * 64
PURGE_REVISION = "b" * 64


@pytest.fixture
def upgrade_fixture(
    tmp_path: Path,
) -> Iterator[tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources]]:
    layout = ArchiveLayout(tmp_path / "meta-backup-upgrade-state")
    target = layout.trash / "deletions" / OPERATION_ID / "workspace-upgrade.json"
    target.parent.mkdir(parents=True, mode=0o700)
    os.chmod(layout.root, 0o700)
    os.chmod(layout.trash, 0o700)
    os.chmod(layout.trash / "deletions", 0o700)
    os.chmod(target.parent, 0o700)

    new_bytes = b'{"retained":true,"value":"rewritten"}'
    new_sha256 = hashlib.sha256(new_bytes).hexdigest()
    relative_path = "messages/inbox/thread/message_1.json"
    source = f"chunk-a/{relative_path}"
    rewrite = RawSourceRewrite(
        source_file_id=7,
        source=source,
        chunk="chunk-a",
        relative_path=relative_path,
        physical_path=layout.root / source,
        mode=0o100600,
        mtime_ns=1_700_000_000_000_000_000,
        media_type="application/json",
        old_sha256="c" * 64,
        old_size_bytes=4_096,
        new_sha256=new_sha256,
        new_size_bytes=len(new_bytes),
        new_bytes=new_bytes,
        binary=False,
        pointers=("/messages/0",),
        selected_source_record_ids=("selected-source-record",),
        fingerprint_updates=(
            FingerprintUpdate(
                source_record_id="retained-source-record",
                message_id=None,
                activity_id="retained-activity",
                old_fingerprint="d" * 64,
                new_fingerprint="e" * 64,
                new_activity_payload={"nested": ["complete", {"ordinal": 1}]},
            ),
        ),
    )
    original_row = {
        "chunk": "chunk-a",
        "relative_path": relative_path,
        "sha256": new_sha256,
        "size_bytes": len(new_bytes),
        "mode": 0o100600,
        "mtime_ns": 1_700_000_000_000_000_000,
    }
    relocation_row = {
        **original_row,
        "source": source,
        "blob": f"blobs/sha256/{new_sha256[:2]}/{new_sha256}",
        "operation": "hard-purge-tombstone-v1",
    }
    plan = RawRewritePlan(
        rewrites=(rewrite,),
        original_manifest_rows=(original_row,),
        relocation_manifest_rows=(relocation_row,),
        obsolete_blob_ids=("f" * 64,),
    )
    owned_blob = "1" * 64
    shared_blob = "2" * 64
    resources = SavedDeletionResources.create(
        blob_ids=(owned_blob, shared_blob),
        shared_blob_ids=(shared_blob,),
        recoverably_referenced_blob_ids=(shared_blob,),
        exclusive_blob_ids=(owned_blob,),
        physically_exclusive_blob_ids=(owned_blob,),
        search_window_ids=("direct-window", "derived-window"),
        purge_search_window_ids=("direct-window",),
        audio_segment_ids=("audio-segment",),
        attachment_extraction_ids=("attachment-extraction",),
        physical_audio_segment_ids=("audio-segment",),
        physical_attachment_extraction_ids=(),
        source_record_ids=("selected-source-record",),
        conversation_participant_ids=("participant-candidate",),
        conversation_ids=("affected-conversation",),
        source_file_ids=(7, "8"),
    )
    yield layout, target, plan, resources


def _publish(
    fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> WorkspaceUpgradeStateReceipt:
    layout, target, plan, resources = fixture
    return publish_workspace_upgrade_state(
        layout,
        target,
        operation_id=OPERATION_ID,
        selection_sha256=SELECTION_SHA256,
        purge_revision=PURGE_REVISION,
        raw_plan=plan,
        resources=resources,
    )


def _rewrite_document(
    target: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    rebuild_state_identity: bool = True,
) -> None:
    value = orjson.loads(target.read_bytes())
    assert isinstance(value, dict)
    mutate(value)
    if rebuild_state_identity:
        body = {key: item for key, item in value.items() if key != "state_sha256"}
        value["state_sha256"] = hashlib.sha256(
            orjson.dumps(body, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()
    target.write_bytes(orjson.dumps(value, option=orjson.OPT_SORT_KEYS))
    os.chmod(target, 0o600)


def test_workspace_upgrade_state_round_trips_every_required_field(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, plan, resources = upgrade_fixture
    receipt = _publish(upgrade_fixture)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.stat().st_uid == os.geteuid()
    assert WorkspaceUpgradeStateReceipt.from_json(receipt.to_json()) == receipt
    loaded = load_workspace_upgrade_state(
        layout,
        target,
        receipt=receipt.to_json(),
        expected_operation_id=OPERATION_ID,
        expected_selection_sha256=SELECTION_SHA256,
        expected_purge_revision=PURGE_REVISION,
    )

    assert loaded.resources == resources
    assert loaded.raw_plan == plan
    assert loaded.raw_plan.rewrites[0].physical_path == plan.rewrites[0].physical_path
    update = loaded.raw_plan.rewrites[0].fingerprint_updates[0]
    assert update.new_activity_payload == {"nested": ["complete", {"ordinal": 1}]}
    assert loaded.raw_plan.original_manifest_rows == plan.original_manifest_rows
    assert loaded.raw_plan.relocation_manifest_rows == plan.relocation_manifest_rows
    assert loaded.raw_plan.obsolete_blob_ids == plan.obsolete_blob_ids
    encoded = target.read_bytes()
    document = orjson.loads(encoded)
    generated = document["raw_rewrite_plan"]["rewrites"][0]["new_bytes_base64"]
    assert base64.b64decode(generated, validate=True) == plan.rewrites[0].new_bytes
    assert receipt.file_sha256 == hashlib.sha256(encoded).hexdigest()


def test_binary_rewrite_with_absent_physical_path_round_trips(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, _plan, _resources = upgrade_fixture
    target = target.with_name("binary-workspace-upgrade.json")
    new_sha256 = hashlib.sha256(BINARY_TOMBSTONE_BYTES).hexdigest()
    source = "chunk-a/media/photo.jpg"
    rewrite = RawSourceRewrite(
        source_file_id=9,
        source=source,
        chunk="chunk-a",
        relative_path="media/photo.jpg",
        physical_path=None,
        mode=0o100600,
        mtime_ns=1_700_000_000_000_000_000,
        media_type="image/jpeg",
        old_sha256="6" * 64,
        old_size_bytes=10_000,
        new_sha256=new_sha256,
        new_size_bytes=len(BINARY_TOMBSTONE_BYTES),
        new_bytes=BINARY_TOMBSTONE_BYTES,
        binary=True,
        pointers=(),
        selected_source_record_ids=(),
        fingerprint_updates=(),
    )
    original = {
        "chunk": "chunk-a",
        "relative_path": "media/photo.jpg",
        "sha256": new_sha256,
        "size_bytes": len(BINARY_TOMBSTONE_BYTES),
    }
    relocation = {
        **original,
        "source": source,
        "blob": f"blobs/sha256/{new_sha256[:2]}/{new_sha256}",
    }
    plan = RawRewritePlan((rewrite,), (original,), (relocation,), ("6" * 64,))
    resources = SavedDeletionResources.create(
        blob_ids=("6" * 64,),
        shared_blob_ids=(),
        recoverably_referenced_blob_ids=(),
        exclusive_blob_ids=("6" * 64,),
        physically_exclusive_blob_ids=("6" * 64,),
        search_window_ids=(),
        purge_search_window_ids=(),
        audio_segment_ids=(),
        attachment_extraction_ids=(),
        physical_audio_segment_ids=(),
        physical_attachment_extraction_ids=(),
        source_record_ids=(),
        conversation_participant_ids=(),
        conversation_ids=(),
        source_file_ids=(9,),
    )
    receipt = publish_workspace_upgrade_state(
        layout,
        target,
        operation_id=OPERATION_ID,
        selection_sha256=SELECTION_SHA256,
        purge_revision=PURGE_REVISION,
        raw_plan=plan,
        resources=resources,
    )

    loaded = load_workspace_upgrade_state(layout, target, receipt=receipt)
    assert loaded.raw_plan == plan
    assert loaded.raw_plan.rewrites[0].physical_path is None
    assert loaded.raw_plan.rewrites[0].new_bytes == BINARY_TOMBSTONE_BYTES


def test_publication_is_create_only_and_preserves_the_first_artifact(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    _publish(upgrade_fixture)
    _layout, target, _plan, _resources = upgrade_fixture
    before = target.read_bytes()
    before_stat = target.stat()

    with pytest.raises(DeletionUpgradeStateExistsError, match="refusing to overwrite"):
        _publish(upgrade_fixture)

    after_stat = target.stat()
    assert target.read_bytes() == before
    assert (after_stat.st_dev, after_stat.st_ino) == (before_stat.st_dev, before_stat.st_ino)
    assert not list(target.parent.glob(".workspace-upgrade-*.tmp"))


@pytest.mark.parametrize(
    ("expected_keyword", "override"),
    [
        ("operation", {"expected_operation_id": str(uuid.uuid4())}),
        ("selection", {"expected_selection_sha256": "3" * 64}),
        ("purge revision", {"expected_purge_revision": "4" * 64}),
    ],
)
def test_load_rejects_wrong_caller_identity(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
    expected_keyword: str,
    override: dict[str, str],
) -> None:
    layout, target, _plan, _resources = upgrade_fixture
    receipt = _publish(upgrade_fixture)
    with pytest.raises(DeletionUpgradeStateError, match=expected_keyword):
        load_workspace_upgrade_state(layout, target, receipt=receipt, **override)


def test_receipt_rejects_same_bytes_published_as_a_different_inode(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, _plan, _resources = upgrade_fixture
    receipt = _publish(upgrade_fixture)
    replacement = target.with_name("replacement.json")
    replacement.write_bytes(target.read_bytes())
    os.chmod(replacement, 0o600)
    os.replace(replacement, target)

    with pytest.raises(DeletionUpgradeStateError, match="file identity"):
        load_workspace_upgrade_state(layout, target, receipt=receipt)
    assert load_workspace_upgrade_state(layout, target).operation_id == OPERATION_ID


def test_load_rejects_non_private_mode_before_parsing(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, _plan, _resources = upgrade_fixture
    _publish(upgrade_fixture)
    os.chmod(target, 0o640)

    with pytest.raises(DeletionUpgradeStateError, match="owner-only"):
        load_workspace_upgrade_state(layout, target)


def test_load_rejects_tampered_state_even_when_outer_identity_is_rebuilt(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, _plan, _resources = upgrade_fixture
    _publish(upgrade_fixture)

    def mutate(value: dict[str, Any]) -> None:
        rewrite = value["raw_rewrite_plan"]["rewrites"][0]
        rewrite["new_bytes_base64"] = base64.b64encode(b'{"different":true}').decode()

    _rewrite_document(target, mutate)
    with pytest.raises(DeletionUpgradeStateError, match="Generated rewrite bytes"):
        load_workspace_upgrade_state(layout, target)


def test_load_rejects_noncanonical_or_unknown_fields(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, _plan, _resources = upgrade_fixture
    _publish(upgrade_fixture)

    def mutate(value: dict[str, Any]) -> None:
        value["unexpected_private_copy"] = True

    _rewrite_document(target, mutate)
    with pytest.raises(DeletionUpgradeStateError, match="Invalid workspace-upgrade state fields"):
        load_workspace_upgrade_state(layout, target)


def test_publication_rejects_paths_outside_or_through_symlinked_directories(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
    tmp_path: Path,
) -> None:
    layout, _target, plan, resources = upgrade_fixture
    outside = tmp_path / "outside.json"
    with pytest.raises(DeletionUpgradeStateError, match="outside the archive"):
        publish_workspace_upgrade_state(
            layout,
            outside,
            operation_id=OPERATION_ID,
            selection_sha256=SELECTION_SHA256,
            purge_revision=PURGE_REVISION,
            raw_plan=plan,
            resources=resources,
        )

    external = tmp_path / "external-upgrade"
    external.mkdir()
    alias = layout.trash / "unsafe-alias"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(DeletionUpgradeStateError, match="destination directory"):
        publish_workspace_upgrade_state(
            layout,
            alias / "state.json",
            operation_id=OPERATION_ID,
            selection_sha256=SELECTION_SHA256,
            purge_revision=PURGE_REVISION,
            raw_plan=plan,
            resources=resources,
        )
    assert not (external / "state.json").exists()


def test_resource_factory_canonicalizes_live_selection_mapping(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    _layout, _target, _plan, expected = upgrade_fixture
    mapping = expected.to_json()
    source_record_ids = mapping.pop("source_record_ids")
    mapping["source_file_ids"] = ["8", "7", "8"]
    mapping["conversation_ids"] = ["affected-conversation", "affected-conversation"]

    assert (
        SavedDeletionResources.from_selection_resources(
            mapping, source_record_ids=source_record_ids
        )
        == expected
    )


def test_inconsistent_resource_ownership_is_rejected() -> None:
    with pytest.raises(DeletionUpgradeStateError, match="exclusivity"):
        SavedDeletionResources.create(
            blob_ids=("1" * 64,),
            shared_blob_ids=(),
            recoverably_referenced_blob_ids=(),
            exclusive_blob_ids=(),
            physically_exclusive_blob_ids=("1" * 64,),
            search_window_ids=(),
            purge_search_window_ids=(),
            audio_segment_ids=(),
            attachment_extraction_ids=(),
            physical_audio_segment_ids=(),
            physical_attachment_extraction_ids=(),
            source_record_ids=(),
            conversation_participant_ids=(),
            conversation_ids=(),
            source_file_ids=(),
        )


def test_plan_and_saved_source_record_ownership_must_match(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
) -> None:
    layout, target, plan, resources = upgrade_fixture
    mismatched = SavedDeletionResources.create(
        **{
            **resources.to_json(),
            "source_record_ids": ["different-selected-source-record"],
        }
    )

    with pytest.raises(DeletionUpgradeStateError, match="do not match the raw rewrite plan"):
        publish_workspace_upgrade_state(
            layout,
            target,
            operation_id=OPERATION_ID,
            selection_sha256=SELECTION_SHA256,
            purge_revision=PURGE_REVISION,
            raw_plan=plan,
            resources=mismatched,
        )
    assert not target.exists()


def test_physical_path_must_be_archive_relative_and_match_source(
    upgrade_fixture: tuple[ArchiveLayout, Path, RawRewritePlan, SavedDeletionResources],
    tmp_path: Path,
) -> None:
    layout, target, plan, resources = upgrade_fixture
    original = plan.rewrites[0]
    unsafe_rewrite = RawSourceRewrite(
        source_file_id=original.source_file_id,
        source=original.source,
        chunk=original.chunk,
        relative_path=original.relative_path,
        physical_path=tmp_path / "outside.json",
        mode=original.mode,
        mtime_ns=original.mtime_ns,
        media_type=original.media_type,
        old_sha256=original.old_sha256,
        old_size_bytes=original.old_size_bytes,
        new_sha256=original.new_sha256,
        new_size_bytes=original.new_size_bytes,
        new_bytes=original.new_bytes,
        binary=original.binary,
        pointers=original.pointers,
        selected_source_record_ids=original.selected_source_record_ids,
        fingerprint_updates=original.fingerprint_updates,
    )
    unsafe_plan = RawRewritePlan(
        rewrites=(unsafe_rewrite,),
        original_manifest_rows=plan.original_manifest_rows,
        relocation_manifest_rows=plan.relocation_manifest_rows,
        obsolete_blob_ids=plan.obsolete_blob_ids,
    )

    with pytest.raises(DeletionUpgradeStateError, match="outside the archive"):
        publish_workspace_upgrade_state(
            layout,
            target,
            operation_id=OPERATION_ID,
            selection_sha256=SELECTION_SHA256,
            purge_revision=PURGE_REVISION,
            raw_plan=unsafe_plan,
            resources=resources,
        )
    assert not target.exists()
