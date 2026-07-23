"""In-sandbox attachment text worker.

This module is executed as a standalone file inside Bubblewrap. Keep it independent from the main
application so the sandbox needs only Python's standard library plus pypdf.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

MAX_ARCHIVE_DEPTH = 3
MAX_ARCHIVE_ENTRIES = 1_000
MAX_EXPANDED_BYTES = 500 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100
MAX_OUTPUT_CHARACTERS = 16 * 1024 * 1024
MAX_PLAIN_INPUT_BYTES = 64 * 1024 * 1024

CREDENTIAL_EXTENSIONS = {
    ".asc",
    ".gpg",
    ".kdbx",
    ".key",
    ".p12",
    ".pfx",
    ".pem",
    ".ppk",
}
EXECUTABLE_EXTENSIONS = {
    ".apk",
    ".app",
    ".bat",
    ".bin",
    ".cmd",
    ".com",
    ".dll",
    ".dmg",
    ".exe",
    ".jar",
    ".msi",
    ".ps1",
    ".scr",
    ".sh",
}
MACRO_EXTENSIONS = {".docm", ".dotm", ".potm", ".ppam", ".pptm", ".xlam", ".xlsm"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}
PLAIN_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".rtf",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
KEY_PATTERN = re.compile(
    rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|"
    rb"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}",
    re.IGNORECASE,
)


class LimitExceeded(RuntimeError):
    pass


class UnsafeAttachment(RuntimeError):
    pass


@dataclass(slots=True)
class Budget:
    entries: int = 0
    expanded_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def consume(self, size: int) -> None:
        self.entries += 1
        self.expanded_bytes += size
        if self.entries > MAX_ARCHIVE_ENTRIES:
            raise LimitExceeded(f"archive exceeds {MAX_ARCHIVE_ENTRIES} entries")
        if self.expanded_bytes > MAX_EXPANDED_BYTES:
            raise LimitExceeded("archive exceeds 500 MB expanded data")


class InertHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "template", "noscript"}:
            self.ignored_depth += 1
        elif tag.casefold() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "template", "noscript"}:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif tag.casefold() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "\n".join(part.strip() for part in "".join(self.parts).splitlines() if part.strip())


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in normalized
        or re.match(r"^[A-Za-z]:/", normalized)
    ):
        raise UnsafeAttachment("archive contains a traversal path")
    canonical = path.as_posix()
    if canonical == ".":
        raise UnsafeAttachment("archive contains an empty member path")
    return canonical


def _validated_office_members(
    archive: zipfile.ZipFile, budget: Budget
) -> list[tuple[str, zipfile.ZipInfo]]:
    """Validate and charge every OOXML ZIP member before any member is read."""

    validated: list[tuple[str, zipfile.ZipInfo]] = []
    normalized_names: set[str] = set()
    for info in archive.infolist():
        name = _safe_member_name(info.filename)
        if name in normalized_names:
            raise UnsafeAttachment("archive contains duplicate normalized member paths")
        normalized_names.add(name)

        file_type = (info.external_attr >> 16) & 0o170000
        expected_type = 0o040000 if info.is_dir() else 0o100000
        if file_type not in {0, expected_type}:
            raise UnsafeAttachment("archive contains a link, device, or FIFO")
        if info.is_dir() and info.file_size:
            raise UnsafeAttachment("archive directory member contains data")
        if info.flag_bits & 0x1 or info.compress_type == 99:
            raise UnsafeAttachment("encrypted Office content is unavailable")
        if info.file_size < 0 or info.compress_size < 0:
            raise UnsafeAttachment("archive contains invalid member sizes")
        if info.file_size > MAX_EXPANDED_BYTES:
            raise LimitExceeded("archive member exceeds the expanded-data limit")
        if info.file_size and (
            info.compress_size == 0 or info.file_size > info.compress_size * MAX_COMPRESSION_RATIO
        ):
            raise LimitExceeded(
                f"archive member exceeds the {MAX_COMPRESSION_RATIO}:1 compression-ratio limit"
            )

        budget.consume(info.file_size)
        validated.append((name, info))
    return validated


def _decode(data: bytes) -> str:
    if b"\x00" in data[:8_192]:
        raise UnsafeAttachment("binary content is not eligible for text indexing")
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _xml_text(data: bytes) -> str:
    root = ElementTree.fromstring(data)
    return "\n".join(text.strip() for text in root.itertext() if text and text.strip())


def _office_zip(data: bytes, suffix: str, budget: Budget) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = _validated_office_members(archive, budget)
        if any(name.casefold().endswith("vbaproject.bin") for name, _info in members):
            raise UnsafeAttachment("macro-bearing Office content is excluded")

        if suffix == ".docx":
            targets = {"word/document.xml"}
        elif suffix == ".pptx":
            targets = {
                name for name, _info in members if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            }
        elif suffix == ".xlsx":
            targets = {
                name
                for name, _info in members
                if name == "xl/sharedStrings.xml"
                or (name.startswith("xl/worksheets/") and name.endswith(".xml"))
            }
        elif suffix == ".odt":
            targets = {"content.xml"}
        else:
            return ""

        for name, info in sorted(members, key=lambda item: item[0]):
            if name not in targets:
                continue
            parts.append(_xml_text(archive.read(info)))
    return "\n".join(parts)


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise UnsafeAttachment("encrypted PDF text is unavailable")
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except UnsafeAttachment:
        raise
    except Exception as exc:
        raise UnsafeAttachment("unsupported or invalid PDF container") from exc


def _extract_archive(data: bytes, filename: str, depth: int, budget: Budget) -> str:
    if depth >= MAX_ARCHIVE_DEPTH:
        raise LimitExceeded(f"archive nesting exceeds depth {MAX_ARCHIVE_DEPTH}")
    parts: list[str] = []
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for zip_info in archive.infolist():
                name = _safe_member_name(zip_info.filename)
                mode = (zip_info.external_attr >> 16) & 0o170000
                if mode in {0o120000, 0o010000, 0o020000, 0o060000}:
                    raise UnsafeAttachment("archive contains a link, device, or FIFO")
                if zip_info.is_dir():
                    budget.consume(0)
                    continue
                budget.consume(zip_info.file_size)
                if (
                    zip_info.compress_size
                    and zip_info.file_size / zip_info.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise LimitExceeded("archive member exceeds the 100:1 compression-ratio limit")
                member = archive.read(zip_info)
                text = extract_bytes(member, name, None, depth + 1, budget)
                if text:
                    parts.append(f"[{name}]\n{text}")
    else:
        declared_bytes = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                for tar_info in archive:
                    name = _safe_member_name(tar_info.name)
                    if (
                        tar_info.issym()
                        or tar_info.islnk()
                        or tar_info.isdev()
                        or tar_info.isfifo()
                    ):
                        raise UnsafeAttachment("archive contains a link, device, or FIFO")
                    budget.consume(tar_info.size if tar_info.isfile() else 0)
                    if not tar_info.isfile():
                        continue
                    declared_bytes += tar_info.size
                    if declared_bytes > max(len(data), 1) * MAX_COMPRESSION_RATIO:
                        raise LimitExceeded("archive exceeds the 100:1 compression-ratio limit")
                    extracted = archive.extractfile(tar_info)
                    if extracted is None:
                        continue
                    member = extracted.read(MAX_EXPANDED_BYTES + 1)
                    if len(member) != tar_info.size:
                        raise LimitExceeded(
                            "archive member could not be read within its declared size"
                        )
                    text = extract_bytes(member, name, None, depth + 1, budget)
                    if text:
                        parts.append(f"[{name}]\n{text}")
        except (tarfile.ReadError, EOFError) as exc:
            raise UnsafeAttachment("unsupported or invalid archive container") from exc
    return "\n\n".join(parts)


def extract_bytes(
    data: bytes, filename: str, media_type: str | None, depth: int, budget: Budget
) -> str:
    suffix = Path(filename).suffix.casefold()
    lowered_type = (media_type or "").casefold()
    if suffix in CREDENTIAL_EXTENSIONS or KEY_PATTERN.search(data):
        raise UnsafeAttachment("credential or key material is metadata-only and excluded")
    if suffix in MACRO_EXTENSIONS:
        raise UnsafeAttachment("macro-bearing documents are excluded")
    if suffix in EXECUTABLE_EXTENSIONS or data.startswith((b"MZ", b"\x7fELF", b"#!")):
        raise UnsafeAttachment("executable content is excluded")
    if suffix in {".docx", ".pptx", ".xlsx", ".odt"}:
        return _office_zip(data, suffix, budget)
    if suffix == ".pdf" or lowered_type == "application/pdf":
        return _pdf_text(data)
    if suffix in ARCHIVE_EXTENSIONS or lowered_type in {
        "application/zip",
        "application/x-tar",
        "application/gzip",
    }:
        return _extract_archive(data, filename, depth, budget)
    if suffix in {".html", ".htm"} or lowered_type == "text/html":
        parser = InertHtmlParser()
        parser.feed(_decode(data))
        return parser.text()
    if suffix in PLAIN_EXTENSIONS or lowered_type.startswith("text/"):
        if len(data) > MAX_PLAIN_INPUT_BYTES:
            raise LimitExceeded("plain-text input exceeds the 64 MB parse limit")
        decoded = _decode(data)
        if suffix == ".rtf":
            decoded = re.sub(r"\\[a-z]+\d* ?|[{}]", " ", decoded, flags=re.IGNORECASE)
        if suffix in {".xml"}:
            return _xml_text(data)
        return html.unescape(decoded)
    raise UnsafeAttachment("unsupported attachment type")


def run(source: Path, filename: str, media_type: str | None) -> dict[str, object]:
    data = source.read_bytes()
    budget = Budget()
    try:
        text = extract_bytes(data, filename, media_type, 0, budget)
        text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        truncated = len(text) > MAX_OUTPUT_CHARACTERS
        if truncated:
            text = text[:MAX_OUTPUT_CHARACTERS]
        if not text:
            return {
                "status": "unsupported",
                "text": None,
                "truncated": False,
                "index_eligible": False,
                "diagnostic": "supported container contained no extractable text",
                "warnings": budget.warnings,
                "entries": budget.entries,
                "expanded_bytes": budget.expanded_bytes,
            }
        return {
            "status": "completed",
            "text": text,
            "truncated": truncated,
            "index_eligible": True,
            "diagnostic": None,
            "warnings": budget.warnings,
            "entries": budget.entries,
            "expanded_bytes": budget.expanded_bytes,
        }
    except LimitExceeded as exc:
        return {
            "status": "limit_exceeded",
            "text": None,
            "truncated": False,
            "index_eligible": False,
            "diagnostic": str(exc),
            "warnings": budget.warnings,
            "entries": budget.entries,
            "expanded_bytes": budget.expanded_bytes,
        }
    except UnsafeAttachment as exc:
        diagnostic = str(exc)
        status = (
            "excluded"
            if "excluded" in diagnostic or "metadata-only" in diagnostic
            else "unsupported"
        )
        return {
            "status": status,
            "text": None,
            "truncated": False,
            "index_eligible": False,
            "diagnostic": diagnostic,
            "warnings": budget.warnings,
            "entries": budget.entries,
            "expanded_bytes": budget.expanded_bytes,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--media-type")
    args = parser.parse_args()
    try:
        payload = run(args.source, args.filename, args.media_type)
    except Exception as exc:
        payload = {
            "status": "failed",
            "text": None,
            "truncated": False,
            "index_eligible": False,
            # Do not echo parser-controlled content into the parent process's durable job row.
            "diagnostic": f"worker exception: {type(exc).__name__}",
            "warnings": [],
            "entries": 0,
            "expanded_bytes": 0,
        }
    json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
