from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

from meta_archive.services import attachment_worker


def _office_zip(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return output.getvalue()


def _run_office(tmp_path: Path, payload: bytes, suffix: str = ".docx") -> dict[str, object]:
    source = tmp_path / f"fixture{suffix}"
    source.write_bytes(payload)
    return attachment_worker.run(source, source.name, None)


def _patch_central_member(
    payload: bytes,
    member_name: str,
    *,
    encrypted: bool = False,
    compressed_size: int | None = None,
) -> bytes:
    mutated = bytearray(payload)
    signature = b"PK\x01\x02"
    offset = 0
    encoded_name = member_name.encode()
    while (offset := mutated.find(signature, offset)) >= 0:
        name_length = int.from_bytes(mutated[offset + 28 : offset + 30], "little")
        extra_length = int.from_bytes(mutated[offset + 30 : offset + 32], "little")
        comment_length = int.from_bytes(mutated[offset + 32 : offset + 34], "little")
        name = bytes(mutated[offset + 46 : offset + 46 + name_length])
        if name == encoded_name:
            if encrypted:
                flags = int.from_bytes(mutated[offset + 8 : offset + 10], "little") | 0x1
                mutated[offset + 8 : offset + 10] = flags.to_bytes(2, "little")
            if compressed_size is not None:
                mutated[offset + 20 : offset + 24] = compressed_size.to_bytes(4, "little")
            return bytes(mutated)
        offset += 46 + name_length + extra_length + comment_length
    raise AssertionError(f"missing central-directory member {member_name!r}")


@pytest.mark.parametrize(
    ("suffix", "target", "xml", "expected"),
    (
        (".docx", "word/document.xml", b"<document>docx text</document>", "docx text"),
        (".pptx", "ppt/slides/slide1.xml", b"<slide>pptx text</slide>", "pptx text"),
        (
            ".xlsx",
            "xl/sharedStrings.xml",
            b"<strings>xlsx text</strings>",
            "xlsx text",
        ),
        (".odt", "content.xml", b"<document>odt text</document>", "odt text"),
    ),
)
def test_office_worker_preserves_valid_extraction_and_safe_directories(
    tmp_path: Path,
    suffix: str,
    target: str,
    xml: bytes,
    expected: str,
) -> None:
    payload = _office_zip([("metadata/", b""), (target, xml), ("unused.bin", b"unused")])

    result = _run_office(tmp_path, payload, suffix)

    assert result["status"] == "completed"
    assert result["text"] == expected
    assert result["entries"] == 3
    assert result["expanded_bytes"] == len(xml) + len(b"unused")


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../unused.bin",
        "/unused.bin",
        r"\unused.bin",
        r"..\unused.bin",
        r"C:\unused.bin",
    ),
)
def test_office_worker_rejects_unsafe_unused_member_paths(tmp_path: Path, unsafe_name: str) -> None:
    payload = _office_zip(
        [("word/document.xml", b"<document>visible</document>"), (unsafe_name, b"unused")]
    )

    result = _run_office(tmp_path, payload)

    assert result["status"] == "unsupported"
    assert "traversal" in str(result["diagnostic"])
    assert result["text"] is None


def test_office_worker_rejects_duplicate_normalized_unused_paths(tmp_path: Path) -> None:
    payload = _office_zip(
        [
            ("word/document.xml", b"<document>visible</document>"),
            ("word/./document.xml", b"<document>shadow</document>"),
        ]
    )

    result = _run_office(tmp_path, payload)

    assert result["status"] == "unsupported"
    assert "duplicate normalized" in str(result["diagnostic"])
    assert result["text"] is None


@pytest.mark.parametrize(
    "member_type",
    (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK, stat.S_IFDIR),
)
def test_office_worker_rejects_non_regular_unused_member_types(
    tmp_path: Path, member_type: int
) -> None:
    special = zipfile.ZipInfo("unused.bin")
    special.create_system = 3
    special.external_attr = (member_type | 0o600) << 16
    payload = _office_zip(
        [("word/document.xml", b"<document>visible</document>"), (special, b"unused")]
    )

    result = _run_office(tmp_path, payload)

    assert result["status"] == "unsupported"
    assert "link, device, or FIFO" in str(result["diagnostic"])
    assert result["text"] is None


def test_office_worker_rejects_an_encrypted_unused_member(tmp_path: Path) -> None:
    payload = _office_zip(
        [("word/document.xml", b"<document>visible</document>"), ("unused.bin", b"unused")]
    )
    payload = _patch_central_member(payload, "unused.bin", encrypted=True)

    result = _run_office(tmp_path, payload)

    assert result["status"] == "unsupported"
    assert "encrypted Office content" in str(result["diagnostic"])
    assert result["text"] is None


def test_office_worker_counts_every_member_before_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attachment_worker, "MAX_ARCHIVE_ENTRIES", 2)
    payload = _office_zip(
        [
            ("word/document.xml", b"<document>visible</document>"),
            ("unused-a.bin", b"a"),
            ("unused-b.bin", b"b"),
        ]
    )

    result = _run_office(tmp_path, payload)

    assert result["status"] == "limit_exceeded"
    assert "2 entries" in str(result["diagnostic"])
    assert result["entries"] == 3
    assert result["text"] is None


def test_office_worker_limits_each_unused_members_expanded_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = b"<x/>"
    monkeypatch.setattr(attachment_worker, "MAX_EXPANDED_BYTES", len(xml))
    payload = _office_zip([("word/document.xml", xml), ("unused.bin", b"12345")])

    result = _run_office(tmp_path, payload)

    assert result["status"] == "limit_exceeded"
    assert "member exceeds" in str(result["diagnostic"])
    assert result["text"] is None


def test_office_worker_limits_cumulative_unused_expanded_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = b"<x/>"
    monkeypatch.setattr(attachment_worker, "MAX_EXPANDED_BYTES", len(xml) + 1)
    payload = _office_zip([("word/document.xml", xml), ("unused.bin", b"12")])

    result = _run_office(tmp_path, payload)

    assert result["status"] == "limit_exceeded"
    assert "expanded data" in str(result["diagnostic"])
    assert result["expanded_bytes"] == len(xml) + 2
    assert result["text"] is None


def test_office_worker_rejects_nonempty_member_with_zero_compressed_size(
    tmp_path: Path,
) -> None:
    payload = _office_zip(
        [("word/document.xml", b"<document>visible</document>"), ("unused.bin", b"x")]
    )
    payload = _patch_central_member(payload, "unused.bin", compressed_size=0)

    result = _run_office(tmp_path, payload)

    assert result["status"] == "limit_exceeded"
    assert "compression-ratio" in str(result["diagnostic"])
    assert result["text"] is None


def test_office_worker_rejects_high_ratio_unused_member(tmp_path: Path) -> None:
    compressed = zipfile.ZipInfo("unused.bin")
    compressed.compress_type = zipfile.ZIP_DEFLATED
    payload = _office_zip(
        [
            ("word/document.xml", b"<document>visible</document>"),
            (compressed, b"0" * 16_384),
        ]
    )

    result = _run_office(tmp_path, payload)

    assert result["status"] == "limit_exceeded"
    assert "100:1 compression-ratio" in str(result["diagnostic"])
    assert result["text"] is None
