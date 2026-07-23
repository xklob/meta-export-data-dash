#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from meta_archive.paths import ArchiveLayout
from meta_archive.services.semantic_bakeoff import (
    DEFAULT_CORPUS_CAP,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_HARD_DISTRACTOR_CAP,
    DEFAULT_JUDGMENT_CAP_PER_QUERY,
    DEFAULT_QUERY_CAP,
    BakeoffError,
    available_model_aliases,
    build_bakeoff_manifest,
    evaluate_bakeoff,
    production_embedder_factory,
    write_private_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally execute a bounded, development-only semantic-model bakeoff. "
            "Output contains aggregate metrics and opaque manifest digests only."
        )
    )
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="ALIAS",
        help=(
            "Explicit model roster. It must include multilingual-e5-small plus at least one "
            f"candidate. Available: {', '.join(available_model_aliases())}"
        ),
    )
    parser.add_argument(
        "--manifest-only",
        "--dry-run",
        action="store_true",
        dest="manifest_only",
        help=(
            "Validate inputs and emit the privacy-safe deterministic manifest without "
            "loading models."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus-cap", type=int, default=DEFAULT_CORPUS_CAP)
    parser.add_argument(
        "--hard-distractor-cap",
        type=int,
        default=DEFAULT_HARD_DISTRACTOR_CAP,
    )
    parser.add_argument("--query-cap", type=int, default=DEFAULT_QUERY_CAP)
    parser.add_argument(
        "--judgment-cap-per-query",
        type=int,
        default=DEFAULT_JUDGMENT_CAP_PER_QUERY,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail rather than downloading any missing pinned model snapshot.",
    )
    return parser


def _safe_error(exc: BakeoffError) -> dict[str, Any]:
    return {
        "schema_version": "semantic-model-bakeoff-v1",
        "status": "failed",
        "error_code": exc.code,
        "counts": exc.counts,
        "private_query_details_persisted": False,
        "private_result_details_persisted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        layout = ArchiveLayout.discover(args.archive_root)
        manifest = build_bakeoff_manifest(
            layout,
            model_aliases=args.models,
            corpus_cap=args.corpus_cap,
            hard_distractor_cap=args.hard_distractor_cap,
            query_cap=args.query_cap,
            judgment_cap_per_query=args.judgment_cap_per_query,
        )
        if args.manifest_only:
            report = {**manifest.safe_summary(), "status": "manifest_ready"}
        else:
            report = evaluate_bakeoff(
                manifest,
                embedder_factory=production_embedder_factory(
                    layout,
                    batch_size=args.batch_size,
                    local_files_only=args.local_files_only,
                ),
            )
        if args.output is not None:
            write_private_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except BakeoffError as exc:
        print(json.dumps(_safe_error(exc), sort_keys=True), file=sys.stderr)
        return 2
    except Exception:
        # Third-party model and SQLite errors can include input values. Never reflect them.
        print(
            json.dumps(
                {
                    "schema_version": "semantic-model-bakeoff-v1",
                    "status": "failed",
                    "error_code": "unexpected_failure",
                    "counts": {},
                    "private_query_details_persisted": False,
                    "private_result_details_persisted": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
