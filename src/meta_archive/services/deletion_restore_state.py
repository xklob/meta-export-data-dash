from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, String, cast, func, select, update
from sqlalchemy.orm import Session

from meta_archive.models import (
    ConflictAggregateRun,
    ConflictHourlyRollup,
    ConflictWindowScore,
    Job,
    NlpAnnotation,
    NlpHourlyRollup,
    SearchWindow,
    SearchWindowMessage,
    SemanticIndexConfig,
    SemanticWindowEmbedding,
)

RESTORE_STATE_VERSION = 1
_ID_CHUNK = 500
_CONFLICT_RUN_MUTABLE_FIELDS = frozenset(
    {"active", "status", "error_text", "provenance_json", "updated_at"}
)
_NLP_ROLLUP_GENERATION_FIELDS = (
    "task",
    "pipeline_version",
    "model_name",
    "model_revision",
    "configuration_sha256",
    "generated_at",
)


class DerivedRestoreStateError(RuntimeError):
    """The published derived state can no longer be restored exactly."""


def _chunks(values: Sequence[str]) -> Iterator[Sequence[str]]:
    for offset in range(0, len(values), _ID_CHUNK):
        yield values[offset : offset + _ID_CHUNK]


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat()
    return copy.deepcopy(value)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _row_snapshot(row: Any, *, excluding: Collection[str] = ()) -> dict[str, Any]:
    omitted = set(excluding)
    return {
        column.name: _json_value(getattr(row, column.name))
        for column in row.__table__.columns
        if column.name not in omitted
    }


def _row_sha256(row: Any, *, excluding: Collection[str] = ()) -> str:
    return _canonical_sha256(_row_snapshot(row, excluding=excluding))


def _entry(
    row: Any,
    *,
    stable_excluding: Collection[str] = ("active",),
    include_snapshot: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": str(row.id),
        "pre_sha256": _row_sha256(row),
        "stable_sha256": _row_sha256(row, excluding=stable_excluding),
    }
    if include_snapshot:
        result["pre_snapshot"] = _row_snapshot(row)
    return result


def _state_digest(state: Mapping[str, Any], *, omitted: Collection[str]) -> str:
    return _canonical_sha256({key: value for key, value in state.items() if key not in omitted})


def _require_digest(state: Mapping[str, Any], field: str, *, omitted: Collection[str]) -> None:
    expected = state.get(field)
    if not isinstance(expected, str) or expected != _state_digest(state, omitted=omitted):
        raise DerivedRestoreStateError("Logical derived restore state failed integrity validation")


def _rows_by_ids(session: Session, model: Any, identifiers: Collection[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    ordered = sorted(set(identifiers))
    for chunk in _chunks(ordered):
        rows.update(
            {str(row.id): row for row in session.scalars(select(model).where(model.id.in_(chunk)))}
        )
    return rows


def _rowset_stats(
    session: Session,
    model: Any,
    conditions: Sequence[Any],
) -> tuple[int, int, str]:
    """Return count, active count, and stable digest with bounded Python memory."""

    digest = hashlib.sha256()
    row_count = 0
    active_count = 0
    statement = (
        select(*model.__table__.columns)
        .where(*conditions)
        .order_by(model.id)
        .execution_options(yield_per=512)
    )
    for row in session.execute(statement).mappings():
        snapshot = {key: _json_value(value) for key, value in row.items() if key != "active"}
        digest.update(
            json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        row_count += 1
        active_count += int(bool(row["active"]))
    return row_count, active_count, digest.hexdigest()


def _database_datetime_text(value: object) -> str:
    if not isinstance(value, str):
        raise DerivedRestoreStateError("Derived generation timestamp is invalid")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise DerivedRestoreStateError("Derived generation timestamp is invalid") from exc
    return value


def _generation_column(model: Any, field: str) -> Any:
    column = getattr(model, field)
    # SQLite's CURRENT_TIMESTAMP stores second-precision text while SQLAlchemy binds the
    # equivalent datetime with a fractional suffix. Preserve and compare the raw stored text so
    # generation identity remains exact across both representations.
    return cast(column, String) if isinstance(column.type, DateTime) else column


def _generation_conditions(
    model: Any,
    identity: Mapping[str, Any],
    fields: Sequence[str],
) -> list[Any]:
    if set(identity) != set(fields):
        raise DerivedRestoreStateError("Derived generation identity is invalid")
    conditions: list[Any] = []
    for field in fields:
        column = getattr(model, field)
        value = identity[field]
        if isinstance(column.type, DateTime):
            value = _database_datetime_text(value)
        conditions.append(_generation_column(model, field) == value)
    return conditions


def _capture_nlp_rollup_generations(session: Session) -> list[dict[str, Any]]:
    columns = [
        _generation_column(NlpHourlyRollup, field).label(field)
        for field in _NLP_ROLLUP_GENERATION_FIELDS
    ]
    groups = list(
        session.execute(
            select(*columns, func.count(NlpHourlyRollup.id))
            .where(NlpHourlyRollup.active.is_(True))
            .group_by(*columns)
            .order_by(*columns)
        )
    )
    generations: list[dict[str, Any]] = []
    for row in groups:
        identity = {
            field: _json_value(row[index])
            for index, field in enumerate(_NLP_ROLLUP_GENERATION_FIELDS)
        }
        conditions = _generation_conditions(
            NlpHourlyRollup,
            identity,
            _NLP_ROLLUP_GENERATION_FIELDS,
        )
        row_count, active_count, stable_sha256 = _rowset_stats(session, NlpHourlyRollup, conditions)
        if row_count != int(row[-1]) or active_count != row_count:
            raise DerivedRestoreStateError(
                "Published NLP rollup generation contains mixed active state"
            )
        generations.append(
            {
                "identity": identity,
                "row_count": row_count,
                "stable_sha256": stable_sha256,
            }
        )
    return generations


def _capture_conflict_generation(session: Session, model: Any, run_id: str) -> dict[str, Any]:
    conditions = [model.run_id == run_id]
    row_count, active_count, stable_sha256 = _rowset_stats(session, model, conditions)
    if active_count not in {0, row_count}:
        raise DerivedRestoreStateError(
            "Conflict publication generation contains mixed active state"
        )
    return {
        "row_count": row_count,
        "active_mode": "all" if active_count else "none",
        "stable_sha256": stable_sha256,
    }


def _require_generation_payload(raw: object, *, conflict: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DerivedRestoreStateError("Derived generation state is invalid")
    row_count = raw.get("row_count")
    stable_sha256 = raw.get("stable_sha256")
    if (
        not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 0
        or not isinstance(stable_sha256, str)
        or len(stable_sha256) != 64
    ):
        raise DerivedRestoreStateError("Derived generation state is invalid")
    if conflict and raw.get("active_mode") not in {"all", "none"}:
        raise DerivedRestoreStateError("Conflict generation active mode is invalid")
    return raw


def _require_nlp_rollup_generations(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise DerivedRestoreStateError("NLP rollup generation state is invalid")
    generations: list[dict[str, Any]] = []
    identities: set[str] = set()
    for value in raw:
        generation = _require_generation_payload(value, conflict=False)
        identity = generation.get("identity")
        if not isinstance(identity, dict):
            raise DerivedRestoreStateError("NLP rollup generation identity is invalid")
        _generation_conditions(
            NlpHourlyRollup,
            identity,
            _NLP_ROLLUP_GENERATION_FIELDS,
        )
        identity_sha256 = _canonical_sha256(identity)
        if identity_sha256 in identities:
            raise DerivedRestoreStateError("NLP rollup generations contain duplicate identities")
        identities.add(identity_sha256)
        generations.append(generation)
    return generations


def _validate_nlp_rollup_generations(
    session: Session,
    raw: object,
    *,
    expected_active: bool,
    activate: bool = False,
) -> int:
    generations = _require_nlp_rollup_generations(raw)
    restored = 0
    for generation in generations:
        identity = generation["identity"]
        conditions = _generation_conditions(
            NlpHourlyRollup,
            identity,
            _NLP_ROLLUP_GENERATION_FIELDS,
        )
        row_count, active_count, stable_sha256 = _rowset_stats(session, NlpHourlyRollup, conditions)
        expected_count = int(generation["row_count"])
        expected_active_count = expected_count if expected_active else 0
        if (
            row_count != expected_count
            or active_count != expected_active_count
            or stable_sha256 != generation["stable_sha256"]
        ):
            raise DerivedRestoreStateError(
                "NLP rollup generation changed while rows were quarantined"
            )
        if activate and expected_count:
            result = session.execute(update(NlpHourlyRollup).where(*conditions).values(active=True))
            if int(getattr(result, "rowcount", 0) or 0) != expected_count:
                raise DerivedRestoreStateError("NLP rollup generation restoration was incomplete")
            restored += expected_count
    return restored


def _validate_conflict_generation(
    session: Session,
    model: Any,
    run_id: str,
    raw: object,
    *,
    quarantined: bool,
    activate: bool = False,
) -> int:
    generation = _require_generation_payload(raw, conflict=True)
    conditions = [model.run_id == run_id]
    row_count, active_count, stable_sha256 = _rowset_stats(session, model, conditions)
    expected_count = int(generation["row_count"])
    expected_active_count = (
        0 if quarantined else expected_count if generation["active_mode"] == "all" else 0
    )
    if (
        row_count != expected_count
        or active_count != expected_active_count
        or stable_sha256 != generation["stable_sha256"]
    ):
        raise DerivedRestoreStateError("Conflict generation changed while rows were quarantined")
    if activate and generation["active_mode"] == "all" and expected_count:
        result = session.execute(update(model).where(*conditions).values(active=True))
        if int(getattr(result, "rowcount", 0) or 0) != expected_count:
            raise DerivedRestoreStateError("Conflict generation restoration was incomplete")
        return expected_count
    return 0


def _restore_encoding_modes(state: Mapping[str, Any]) -> tuple[bool, bool]:
    """Validate that compact and legacy restore encodings cannot be mixed."""

    compact_nlp = "nlp_rollup_generations" in state
    legacy_nlp = _require_unique_entries(state.get("nlp_hourly_rollups"), "nlp_hourly_rollups")
    if compact_nlp and legacy_nlp:
        raise DerivedRestoreStateError("NLP restore state mixes compact and legacy encodings")

    run_entries = _require_unique_entries(state.get("conflict_runs"), "conflict_runs")
    generation_flags: list[bool] = []
    for entry in run_entries:
        has_score_generation = "score_generation" in entry
        has_rollup_generation = "rollup_generation" in entry
        if has_score_generation != has_rollup_generation:
            raise DerivedRestoreStateError("Conflict restore generation state is incomplete")
        generation_flags.append(has_score_generation)
    if any(generation_flags) and not all(generation_flags):
        raise DerivedRestoreStateError("Conflict restore state mixes compact and legacy runs")

    legacy_scores = _require_unique_entries(state.get("conflict_scores"), "conflict_scores")
    legacy_rollups = _require_unique_entries(state.get("conflict_rollups"), "conflict_rollups")
    compact_conflict = bool(run_entries) and all(generation_flags)
    if compact_conflict and (legacy_scores or legacy_rollups):
        raise DerivedRestoreStateError("Conflict restore state mixes compact and legacy encodings")
    return compact_nlp, compact_conflict


def _require_unique_entries(raw: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise DerivedRestoreStateError(f"Logical restore field {field} is invalid")
    entries: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for value in raw:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise DerivedRestoreStateError(f"Logical restore field {field} is invalid")
        identifier = value["id"]
        if identifier in identifiers:
            raise DerivedRestoreStateError(f"Logical restore field {field} has duplicate IDs")
        identifiers.add(identifier)
        entries.append(value)
    return entries


def _require_entry_hashes(entry: Mapping[str, Any], *, snapshot: bool = False) -> None:
    required = ("pre_sha256", "stable_sha256", "quarantine_sha256")
    if any(not isinstance(entry.get(field), str) for field in required):
        raise DerivedRestoreStateError("Logical restore row identity is invalid")
    if snapshot:
        pre_snapshot = entry.get("pre_snapshot")
        if not isinstance(pre_snapshot, dict):
            raise DerivedRestoreStateError("Logical restore row snapshot is invalid")
        if _canonical_sha256(pre_snapshot) != entry["pre_sha256"]:
            raise DerivedRestoreStateError("Logical restore row snapshot failed validation")


def _active_annotation_entries(
    session: Session,
    nlp_targets: Mapping[str, Collection[str]],
) -> list[dict[str, Any]]:
    rows: dict[str, NlpAnnotation] = {}
    for target_type, raw_identifiers in sorted(nlp_targets.items()):
        identifiers = sorted(set(raw_identifiers))
        if not isinstance(target_type, str) or not target_type:
            raise ValueError("NLP restore target types must be non-empty strings")
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError("NLP restore target IDs must be non-empty strings")
        for chunk in _chunks(identifiers):
            rows.update(
                {
                    row.id: row
                    for row in session.scalars(
                        select(NlpAnnotation).where(
                            NlpAnnotation.target_type == target_type,
                            NlpAnnotation.target_id.in_(chunk),
                            NlpAnnotation.active.is_(True),
                        )
                    )
                }
            )
    return [
        {
            **_entry(row),
            "target_type": row.target_type,
            "target_id": row.target_id,
            "task": row.task,
        }
        for row in sorted(rows.values(), key=lambda value: value.id)
    ]


def _search_state(session: Session, search_window_ids: Collection[str]) -> dict[str, Any]:
    identifiers = sorted(set(search_window_ids))
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("Search-window restore IDs must be non-empty strings")
    rows = _rows_by_ids(session, SearchWindow, identifiers)
    if set(rows) != set(identifiers):
        raise DerivedRestoreStateError("Affected search windows disappeared before quarantine")

    windows: list[dict[str, Any]] = []
    for identifier in identifiers:
        links = [
            {"message_id": str(message_id), "ordinal": int(ordinal)}
            for message_id, ordinal in session.execute(
                select(SearchWindowMessage.message_id, SearchWindowMessage.ordinal)
                .where(SearchWindowMessage.window_id == identifier)
                .order_by(SearchWindowMessage.ordinal, SearchWindowMessage.message_id)
            )
        ]
        memberships = sorted(
            str(config_id)
            for config_id in session.scalars(
                select(SemanticWindowEmbedding.config_id).where(
                    SemanticWindowEmbedding.window_id == identifier
                )
            )
        )
        windows.append(
            {
                "id": identifier,
                "snapshot_sha256": _row_sha256(rows[identifier]),
                "message_links": links,
                "semantic_config_ids": memberships,
            }
        )

    active_configs = list(
        session.scalars(
            select(SemanticIndexConfig)
            .where(SemanticIndexConfig.active.is_(True))
            .order_by(SemanticIndexConfig.id)
        )
    )
    return {
        "windows": windows,
        "active_semantic_configs": [
            {"id": row.id, "snapshot_sha256": _row_sha256(row)} for row in active_configs
        ],
    }


def capture_derived_restore_state(
    session: Session,
    *,
    nlp_targets: Mapping[str, Collection[str]],
    passthrough_nlp_targets: Mapping[str, Collection[str]] | None = None,
    search_window_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Capture the exact published state before logical-quarantine invalidation.

    Call :func:`seal_quarantined_derived_state` in the same transaction after the caller has
    invalidated the captured rows. The returned value is JSON-serializable.
    """

    active_annotations = [
        {**entry, "passthrough": False}
        for entry in _active_annotation_entries(session, nlp_targets)
    ]
    passthrough_annotations = _active_annotation_entries(session, passthrough_nlp_targets or {})
    active_annotation_ids = {entry["id"] for entry in active_annotations}
    passthrough_annotation_ids = {entry["id"] for entry in passthrough_annotations}
    if active_annotation_ids & passthrough_annotation_ids:
        raise ValueError(
            "NLP annotations cannot be both quarantined and passthrough restore targets"
        )

    conflict_runs = list(
        session.scalars(
            select(ConflictAggregateRun)
            .where(ConflictAggregateRun.status.in_(("building", "ready")))
            .order_by(ConflictAggregateRun.id)
        )
    )
    topic_jobs = list(
        session.scalars(select(Job).where(Job.kind == "topic_clusters").order_by(Job.id))
    )
    state: dict[str, Any] = {
        "version": RESTORE_STATE_VERSION,
        "sealed": False,
        "nlp_annotations": sorted(
            active_annotations
            + [{**entry, "passthrough": True} for entry in passthrough_annotations],
            key=lambda entry: entry["id"],
        ),
        "nlp_rollup_generations": _capture_nlp_rollup_generations(session),
        # Retained as empty compatibility fields so older consumers fail closed instead of
        # interpreting a missing key as an unbounded legacy payload.
        "nlp_hourly_rollups": [],
        "conflict_runs": [
            {
                **_entry(
                    row,
                    stable_excluding=_CONFLICT_RUN_MUTABLE_FIELDS,
                    include_snapshot=True,
                ),
                "score_generation": _capture_conflict_generation(
                    session, ConflictWindowScore, row.id
                ),
                "rollup_generation": _capture_conflict_generation(
                    session, ConflictHourlyRollup, row.id
                ),
            }
            for row in conflict_runs
        ],
        "conflict_scores": [],
        "conflict_rollups": [],
        "topic_jobs": [
            _entry(row, stable_excluding=row.__table__.columns.keys(), include_snapshot=True)
            for row in topic_jobs
        ],
        "search": _search_state(session, search_window_ids),
    }
    state["capture_sha256"] = _state_digest(state, omitted=("capture_sha256",))
    return state


def _validate_search_state(
    session: Session,
    raw: object,
    *,
    validate_windows: bool = True,
    validate_active_configs: bool = True,
) -> None:
    if not isinstance(raw, dict):
        raise DerivedRestoreStateError("Logical search restore state is invalid")
    windows = _require_unique_entries(raw.get("windows"), "search.windows")
    for entry in windows:
        if not isinstance(entry.get("snapshot_sha256"), str):
            raise DerivedRestoreStateError("Logical search restore state is invalid")
        links = entry.get("message_links")
        memberships = entry.get("semantic_config_ids")
        if (
            not isinstance(links, list)
            or any(
                not isinstance(link, dict)
                or set(link) != {"message_id", "ordinal"}
                or not isinstance(link["message_id"], str)
                or not link["message_id"]
                or not isinstance(link["ordinal"], int)
                or isinstance(link["ordinal"], bool)
                for link in links
            )
            or not isinstance(memberships, list)
            or any(not isinstance(value, str) for value in memberships)
        ):
            raise DerivedRestoreStateError("Logical search restore state is invalid")

    configs = _require_unique_entries(
        raw.get("active_semantic_configs"), "search.active_semantic_configs"
    )
    if any(not isinstance(entry.get("snapshot_sha256"), str) for entry in configs):
        raise DerivedRestoreStateError("Logical search restore state is invalid")

    # A later hard purge can intentionally remove shared windows and their semantic memberships.
    # The caller records that incompatibility and requires a rebuild, so only the sealed payload
    # shape remains relevant in that mode. Exact restores continue through every identity check.
    if not validate_windows and not validate_active_configs:
        return

    rows = _rows_by_ids(session, SearchWindow, {entry["id"] for entry in windows})
    if set(rows) != {entry["id"] for entry in windows}:
        raise DerivedRestoreStateError("Search windows changed while rows were quarantined")
    for entry in windows:
        if _row_sha256(rows[entry["id"]]) != entry["snapshot_sha256"]:
            raise DerivedRestoreStateError("Search windows changed while rows were quarantined")
        links = entry["message_links"]
        memberships = entry["semantic_config_ids"]
        current_links = [
            {"message_id": str(message_id), "ordinal": int(ordinal)}
            for message_id, ordinal in session.execute(
                select(SearchWindowMessage.message_id, SearchWindowMessage.ordinal)
                .where(SearchWindowMessage.window_id == entry["id"])
                .order_by(SearchWindowMessage.ordinal, SearchWindowMessage.message_id)
            )
        ]
        current_memberships = sorted(
            str(config_id)
            for config_id in session.scalars(
                select(SemanticWindowEmbedding.config_id).where(
                    SemanticWindowEmbedding.window_id == entry["id"]
                )
            )
        )
        if current_links != links or current_memberships != memberships:
            raise DerivedRestoreStateError(
                "Search-window links changed while rows were quarantined"
            )

    if not validate_active_configs:
        return
    current_configs = list(
        session.scalars(
            select(SemanticIndexConfig)
            .where(SemanticIndexConfig.active.is_(True))
            .order_by(SemanticIndexConfig.id)
        )
    )
    by_id = {row.id: row for row in current_configs}
    if set(by_id) != {entry["id"] for entry in configs}:
        raise DerivedRestoreStateError(
            "The active semantic configuration changed during quarantine"
        )
    for entry in configs:
        if (
            not isinstance(entry.get("snapshot_sha256"), str)
            or _row_sha256(by_id[entry["id"]]) != entry["snapshot_sha256"]
        ):
            raise DerivedRestoreStateError(
                "The active semantic configuration changed during quarantine"
            )


def seal_quarantined_derived_state(session: Session, state: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the captured pre-state to the exact invalidated rows now in SQLite."""

    if state.get("version") != RESTORE_STATE_VERSION or state.get("sealed") is not False:
        raise DerivedRestoreStateError("Logical derived restore state is not sealable")
    _require_digest(state, "capture_sha256", omitted=("capture_sha256",))
    sealed = copy.deepcopy(dict(state))
    compact_nlp_rollups, compact_conflict_generations = _restore_encoding_modes(sealed)

    specifications = (
        ("nlp_hourly_rollups", NlpHourlyRollup, False),
        ("conflict_scores", ConflictWindowScore, False),
        ("conflict_rollups", ConflictHourlyRollup, False),
    )
    for field, model, _snapshot in specifications:
        entries = _require_unique_entries(sealed.get(field), field)
        rows = _rows_by_ids(session, model, {entry["id"] for entry in entries})
        if set(rows) != {entry["id"] for entry in entries}:
            raise DerivedRestoreStateError("Derived rows disappeared during quarantine")
        for entry in entries:
            row = rows[entry["id"]]
            if row.active is not False or _row_sha256(row, excluding=("active",)) != entry.get(
                "stable_sha256"
            ):
                raise DerivedRestoreStateError("Derived rows did not enter the expected quarantine")
            entry["quarantine_sha256"] = _row_sha256(row)

    if compact_nlp_rollups:
        _validate_nlp_rollup_generations(
            session,
            sealed.get("nlp_rollup_generations"),
            expected_active=False,
        )

    annotation_entries = _require_unique_entries(sealed.get("nlp_annotations"), "nlp_annotations")
    annotation_rows = _rows_by_ids(
        session,
        NlpAnnotation,
        {entry["id"] for entry in annotation_entries},
    )
    if set(annotation_rows) != {entry["id"] for entry in annotation_entries}:
        raise DerivedRestoreStateError("NLP annotations disappeared during quarantine")
    for entry in annotation_entries:
        passthrough = entry.get("passthrough", False)
        if not isinstance(passthrough, bool):
            raise DerivedRestoreStateError("NLP annotation passthrough state is invalid")
        row = annotation_rows[entry["id"]]
        if row.active is not passthrough or _row_sha256(row, excluding=("active",)) != entry.get(
            "stable_sha256"
        ):
            kind = "Passthrough NLP annotations" if passthrough else "NLP annotations"
            raise DerivedRestoreStateError(f"{kind} changed during quarantine")
        entry["quarantine_sha256"] = _row_sha256(row)

    run_entries = _require_unique_entries(sealed.get("conflict_runs"), "conflict_runs")
    runs = _rows_by_ids(session, ConflictAggregateRun, {entry["id"] for entry in run_entries})
    if set(runs) != {entry["id"] for entry in run_entries}:
        raise DerivedRestoreStateError("Conflict aggregate rows disappeared during quarantine")
    for entry in run_entries:
        _require_entry_hashes({**entry, "quarantine_sha256": "pending"}, snapshot=True)
        row = runs[entry["id"]]
        if (
            row.active is not False
            or row.status != "stale"
            or _row_sha256(row, excluding=_CONFLICT_RUN_MUTABLE_FIELDS)
            != entry.get("stable_sha256")
        ):
            raise DerivedRestoreStateError("Conflict aggregates did not enter quarantine")
        if compact_conflict_generations:
            _validate_conflict_generation(
                session,
                ConflictWindowScore,
                entry["id"],
                entry.get("score_generation"),
                quarantined=True,
            )
            _validate_conflict_generation(
                session,
                ConflictHourlyRollup,
                entry["id"],
                entry.get("rollup_generation"),
                quarantined=True,
            )
        entry["quarantine_sha256"] = _row_sha256(row)

    topic_entries = _require_unique_entries(sealed.get("topic_jobs"), "topic_jobs")
    topics = _rows_by_ids(session, Job, {entry["id"] for entry in topic_entries})
    current_topic_ids = set(session.scalars(select(Job.id).where(Job.kind == "topic_clusters")))
    if set(topics) != {entry["id"] for entry in topic_entries} or current_topic_ids != set(topics):
        raise DerivedRestoreStateError("Topic-cluster jobs changed during quarantine")
    for entry in topic_entries:
        _require_entry_hashes({**entry, "quarantine_sha256": "pending"}, snapshot=True)
        row = topics[entry["id"]]
        if row.status != "stale" or not bool((row.counters_json or {}).get("invalidated")):
            raise DerivedRestoreStateError("Topic-cluster jobs did not enter quarantine")
        entry["quarantine_sha256"] = _row_sha256(row)

    if session.scalar(
        select(func.count(NlpHourlyRollup.id)).where(NlpHourlyRollup.active.is_(True))
    ):
        raise DerivedRestoreStateError("NLP rollups remained active after quarantine")
    if session.scalar(
        select(func.count(ConflictAggregateRun.id)).where(
            ConflictAggregateRun.status.in_(("building", "ready"))
        )
    ):
        raise DerivedRestoreStateError("Conflict aggregates remained published after quarantine")
    _validate_search_state(session, sealed.get("search"))
    sealed["sealed"] = True
    sealed["sealed_sha256"] = _state_digest(sealed, omitted=("capture_sha256", "sealed_sha256"))
    return sealed


def _deserialize_snapshot(row: Any, snapshot: Mapping[str, Any]) -> None:
    expected_columns = {column.name for column in row.__table__.columns}
    if set(snapshot) != expected_columns or snapshot.get("id") != str(row.id):
        raise DerivedRestoreStateError("Logical restore row snapshot has the wrong schema")
    for column in row.__table__.columns:
        value = copy.deepcopy(snapshot[column.name])
        if isinstance(column.type, DateTime) and value is not None:
            if not isinstance(value, str):
                raise DerivedRestoreStateError("Logical restore row timestamp is invalid")
            try:
                value = datetime.fromisoformat(value)
            except ValueError as exc:
                raise DerivedRestoreStateError("Logical restore row timestamp is invalid") from exc
        setattr(row, column.name, value)


def _validate_sealed_entries(
    session: Session,
    state: Mapping[str, Any],
    field: str,
    model: Any,
    *,
    snapshot: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = _require_unique_entries(state.get(field), field)
    for entry in entries:
        _require_entry_hashes(entry, snapshot=snapshot)
    rows = _rows_by_ids(session, model, {entry["id"] for entry in entries})
    if set(rows) != {entry["id"] for entry in entries}:
        raise DerivedRestoreStateError(f"{field} changed while rows were quarantined")
    for entry in entries:
        if _row_sha256(rows[entry["id"]]) != entry["quarantine_sha256"]:
            raise DerivedRestoreStateError(f"{field} changed while rows were quarantined")
    return entries, rows


def restore_derived_restore_state(
    session: Session,
    state: Mapping[str, Any],
    *,
    delegated_annotation_ids: Collection[str] = (),
    restore_global_derived: bool = True,
) -> dict[str, int]:
    """Validate every seal first, then reactivate the exact pre-quarantine publication state.

    The caller must run this inside the same transaction that restores normalized rows and must
    serialize against the NLP, semantic-index, conflict-aggregate, and topic-cluster workers.
    """

    if state.get("version") != RESTORE_STATE_VERSION or state.get("sealed") is not True:
        raise DerivedRestoreStateError("Deletion operation lacks sealed logical restore state")
    if not isinstance(restore_global_derived, bool):
        raise DerivedRestoreStateError("Global derived restore mode is invalid")
    _require_digest(
        state,
        "sealed_sha256",
        omitted=("capture_sha256", "sealed_sha256"),
    )
    compact_nlp_rollups, compact_conflict_generations = _restore_encoding_modes(state)
    annotation_entries = _require_unique_entries(state.get("nlp_annotations"), "nlp_annotations")
    for entry in annotation_entries:
        _require_entry_hashes(entry)
        passthrough = entry.get("passthrough", False)
        if not isinstance(passthrough, bool):
            raise DerivedRestoreStateError("NLP annotation passthrough state is invalid")
    annotations = _rows_by_ids(
        session,
        NlpAnnotation,
        {entry["id"] for entry in annotation_entries},
    )
    if set(annotations) != {entry["id"] for entry in annotation_entries}:
        raise DerivedRestoreStateError("NLP annotations changed while rows were quarantined")
    delegated_ids = set(delegated_annotation_ids)
    if any(not isinstance(identifier, str) or not identifier for identifier in delegated_ids):
        raise DerivedRestoreStateError("Delegated NLP annotation IDs are invalid")
    passthrough_entries = [entry for entry in annotation_entries if entry.get("passthrough", False)]
    direct_entries = [entry for entry in annotation_entries if not entry.get("passthrough", False)]
    passthrough_ids = {entry["id"] for entry in passthrough_entries}
    if not delegated_ids <= passthrough_ids:
        raise DerivedRestoreStateError(
            "Delegated NLP annotation IDs are not captured passthrough rows"
        )
    passthrough_to_reactivate: list[dict[str, Any]] = []
    for entry in direct_entries:
        if _row_sha256(annotations[entry["id"]]) != entry["quarantine_sha256"]:
            raise DerivedRestoreStateError("NLP annotations changed while rows were quarantined")
    for entry in passthrough_entries:
        row = annotations[entry["id"]]
        if row.active is True:
            if _row_sha256(row) != entry["quarantine_sha256"]:
                raise DerivedRestoreStateError(
                    "Passthrough NLP annotations changed while rows were quarantined"
                )
            continue
        if (
            row.active is not False
            or entry["id"] not in delegated_ids
            or _row_sha256(row, excluding=("active",)) != entry["stable_sha256"]
        ):
            raise DerivedRestoreStateError(
                "Passthrough NLP annotation retirement lacks exact delegated provenance"
            )
        passthrough_to_reactivate.append(entry)
    _validate_search_state(
        session,
        state.get("search"),
        validate_windows=restore_global_derived,
        validate_active_configs=restore_global_derived,
    )

    rollup_entries: list[dict[str, Any]] = []
    rollups: dict[str, Any] = {}
    run_entries: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    score_entries: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    conflict_rollup_entries: list[dict[str, Any]] = []
    conflict_rollups: dict[str, Any] = {}
    topic_entries: list[dict[str, Any]] = []
    topics: dict[str, Any] = {}
    nlp_rollup_restored = 0
    conflict_score_restored = 0
    conflict_rollup_restored = 0
    if restore_global_derived:
        if compact_nlp_rollups:
            _validate_nlp_rollup_generations(
                session,
                state.get("nlp_rollup_generations"),
                expected_active=False,
            )
        else:
            rollup_entries, rollups = _validate_sealed_entries(
                session, state, "nlp_hourly_rollups", NlpHourlyRollup
            )
        run_entries, runs = _validate_sealed_entries(
            session, state, "conflict_runs", ConflictAggregateRun, snapshot=True
        )
        if compact_conflict_generations:
            for entry in run_entries:
                _validate_conflict_generation(
                    session,
                    ConflictWindowScore,
                    entry["id"],
                    entry.get("score_generation"),
                    quarantined=True,
                )
                _validate_conflict_generation(
                    session,
                    ConflictHourlyRollup,
                    entry["id"],
                    entry.get("rollup_generation"),
                    quarantined=True,
                )
        else:
            score_entries, scores = _validate_sealed_entries(
                session, state, "conflict_scores", ConflictWindowScore
            )
            conflict_rollup_entries, conflict_rollups = _validate_sealed_entries(
                session, state, "conflict_rollups", ConflictHourlyRollup
            )
        topic_entries, topics = _validate_sealed_entries(
            session, state, "topic_jobs", Job, snapshot=True
        )

        if session.scalar(
            select(func.count(NlpHourlyRollup.id)).where(NlpHourlyRollup.active.is_(True))
        ):
            raise DerivedRestoreStateError("New NLP rollups were published during quarantine")
        if session.scalar(
            select(func.count(ConflictAggregateRun.id)).where(
                ConflictAggregateRun.status.in_(("building", "ready"))
            )
        ):
            raise DerivedRestoreStateError(
                "New conflict aggregates were published during quarantine"
            )
        if session.scalar(
            select(func.count(ConflictWindowScore.id)).where(ConflictWindowScore.active.is_(True))
        ) or session.scalar(
            select(func.count(ConflictHourlyRollup.id)).where(ConflictHourlyRollup.active.is_(True))
        ):
            raise DerivedRestoreStateError(
                "New conflict aggregate rows were published during quarantine"
            )
        current_topic_ids = set(session.scalars(select(Job.id).where(Job.kind == "topic_clusters")))
        if current_topic_ids != set(topics):
            raise DerivedRestoreStateError("Topic-cluster jobs changed during quarantine")

    captured_annotation_ids = {entry["id"] for entry in annotation_entries}
    for entry in direct_entries + passthrough_to_reactivate:
        competitor = session.scalar(
            select(NlpAnnotation.id)
            .where(
                NlpAnnotation.target_type == entry.get("target_type"),
                NlpAnnotation.target_id == entry.get("target_id"),
                NlpAnnotation.task == entry.get("task"),
                NlpAnnotation.active.is_(True),
                NlpAnnotation.id.not_in(captured_annotation_ids),
            )
            .limit(1)
        )
        if competitor is not None:
            raise DerivedRestoreStateError(
                "A replacement NLP annotation was published during quarantine"
            )

    for entry in direct_entries:
        annotations[entry["id"]].active = True
    for entry in passthrough_to_reactivate:
        annotations[entry["id"]].active = True
    if restore_global_derived:
        if compact_nlp_rollups:
            nlp_rollup_restored = _validate_nlp_rollup_generations(
                session,
                state.get("nlp_rollup_generations"),
                expected_active=False,
                activate=True,
            )
        else:
            for entry in rollup_entries:
                rollups[entry["id"]].active = True
            nlp_rollup_restored = len(rollup_entries)
        if compact_conflict_generations:
            for entry in run_entries:
                conflict_score_restored += _validate_conflict_generation(
                    session,
                    ConflictWindowScore,
                    entry["id"],
                    entry.get("score_generation"),
                    quarantined=True,
                    activate=True,
                )
                conflict_rollup_restored += _validate_conflict_generation(
                    session,
                    ConflictHourlyRollup,
                    entry["id"],
                    entry.get("rollup_generation"),
                    quarantined=True,
                    activate=True,
                )
        else:
            for entry in score_entries:
                scores[entry["id"]].active = True
            for entry in conflict_rollup_entries:
                conflict_rollups[entry["id"]].active = True
            conflict_score_restored = len(score_entries)
            conflict_rollup_restored = len(conflict_rollup_entries)
    for entry in run_entries:
        _deserialize_snapshot(runs[entry["id"]], entry["pre_snapshot"])
    for entry in topic_entries:
        _deserialize_snapshot(topics[entry["id"]], entry["pre_snapshot"])
    session.flush()

    for entries, rows in (
        (annotation_entries, annotations),
        (rollup_entries, rollups),
        (run_entries, runs),
        (score_entries, scores),
        (conflict_rollup_entries, conflict_rollups),
        (topic_entries, topics),
    ):
        for entry in entries:
            if _row_sha256(rows[entry["id"]]) != entry["pre_sha256"]:
                raise DerivedRestoreStateError("Exact derived-state restoration failed validation")

    if restore_global_derived and compact_nlp_rollups:
        _validate_nlp_rollup_generations(
            session,
            state.get("nlp_rollup_generations"),
            expected_active=True,
        )
    if restore_global_derived and compact_conflict_generations:
        for entry in run_entries:
            _validate_conflict_generation(
                session,
                ConflictWindowScore,
                entry["id"],
                entry.get("score_generation"),
                quarantined=False,
            )
            _validate_conflict_generation(
                session,
                ConflictHourlyRollup,
                entry["id"],
                entry.get("rollup_generation"),
                quarantined=False,
            )

    affected = {
        "nlp_annotations_reactivated": len(direct_entries) + len(passthrough_to_reactivate),
        "nlp_hourly_rollups_reactivated": nlp_rollup_restored,
        "conflict_aggregate_runs_restored": len(run_entries),
        "conflict_window_scores_reactivated": conflict_score_restored,
        "conflict_hourly_rollups_reactivated": conflict_rollup_restored,
        "topic_cluster_jobs_restored": len(topic_entries),
    }
    if not restore_global_derived:
        affected["global_derived_rebuild_required"] = 1
    return affected


def captured_search_window_ids(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the affected window IDs for caller-side LanceDB identity validation."""

    raw_search = state.get("search")
    if not isinstance(raw_search, dict):
        raise DerivedRestoreStateError("Logical search restore state is invalid")
    return tuple(
        entry["id"]
        for entry in _require_unique_entries(raw_search.get("windows"), "search.windows")
    )
