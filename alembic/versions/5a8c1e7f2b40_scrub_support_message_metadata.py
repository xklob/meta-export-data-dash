"""scrub sensitive support-message metadata

Revision ID: 5a8c1e7f2b40
Revises: 9f2a4c7d8b31
Create Date: 2026-07-22

"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa

from alembic import op

revision: str = "5a8c1e7f2b40"
down_revision: str | Sequence[str] | None = "9f2a4c7d8b31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Keep this migration-local denylist aligned with the importer version at the time of the scrub.
# Matching is exact after case/separator normalization. Broad substring rules would incorrectly
# classify ordinary support metadata such as ``shipping``, ``topic``, or ``participant``.
_SENSITIVE_FIELD_KEYS = frozenset(
    {
        "accountcredentials",
        "bankaccount",
        "bankaccountdetails",
        "bankaccountinfo",
        "bankaccountnumber",
        "billing",
        "billingdetails",
        "billinginfo",
        "billinginformation",
        "cardcredentials",
        "carddetails",
        "cardinfo",
        "cardinformation",
        "cardnumber",
        "clientip",
        "creditcard",
        "creditcarddetails",
        "creditcardinfo",
        "cvc",
        "cvv",
        "iban",
        "ip",
        "ipaddress",
        "payment",
        "paymentdetails",
        "paymentinfo",
        "paymentinformation",
        "paymentmethod",
        "routingnumber",
        "sourceip",
    }
)
_IP_FIELD_KEYS = frozenset({"clientip", "ip", "ipaddress", "sourceip"})
_BATCH_SIZE = 500


def _field_kind(key: str) -> str | None:
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    if normalized not in _SENSITIVE_FIELD_KEYS:
        return None
    return "ip" if normalized in _IP_FIELD_KEYS else "payment"


def _partition(value: Any) -> tuple[Any, Any | None, bool, set[str]]:
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        sensitive: dict[str, Any] = {}
        field_names: set[str] = set()
        for raw_key, item in value.items():
            key = str(raw_key)
            if _field_kind(key) is not None:
                sensitive[key] = item
                field_names.add(key)
                continue
            child_public, child_sensitive, child_has_sensitive, child_names = _partition(item)
            public[key] = child_public
            field_names.update(child_names)
            if child_has_sensitive:
                sensitive[key] = child_sensitive
        return public, sensitive or None, bool(sensitive), field_names

    if isinstance(value, list):
        public_items: list[Any] = []
        sensitive_items: list[Any] = []
        list_field_names: set[str] = set()
        has_sensitive = False
        for item in value:
            child_public, child_sensitive, child_has_sensitive, child_names = _partition(item)
            public_items.append(child_public)
            sensitive_items.append(child_sensitive if child_has_sensitive else None)
            list_field_names.update(child_names)
            has_sensitive = has_sensitive or child_has_sensitive
        return (
            public_items,
            sensitive_items if has_sensitive else None,
            has_sensitive,
            list_field_names,
        )

    return value, None, False, set()


def _merge_sensitive(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, Mapping) and isinstance(incoming, Mapping):
        merged = dict(existing)
        for key, value in incoming.items():
            merged[key] = _merge_sensitive(merged[key], value) if key in merged else value
        return merged
    # The historical importer duplicated canonical sensitive values in both columns, so equal
    # leaves are expected. If a hand-edited catalog differs, retain the already-gated value while
    # still removing the public duplicate rather than overwriting private data during migration.
    return existing


def _sensitive_kinds(value: Any) -> set[str]:
    kinds: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            kind = _field_kind(str(raw_key))
            if kind is not None:
                kinds.add(kind)
            else:
                kinds.update(_sensitive_kinds(item))
    elif isinstance(value, list):
        for item in value:
            kinds.update(_sensitive_kinds(item))
    return kinds


def _json_object(raw: Any, *, column: str, message_pk: int) -> dict[str, Any]:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
    except (TypeError, ValueError) as error:
        raise ValueError(f"messages.{column} is invalid JSON at pk={message_pk}") from error
    if not isinstance(value, dict):
        raise ValueError(f"messages.{column} is not a JSON object at pk={message_pk}")
    return value


def _dump_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def upgrade() -> None:
    connection = op.get_bind()
    last_pk = 0
    while True:
        rows = (
            connection.execute(
                sa.text(
                    """
                SELECT m.pk, m.extra_json, m.sensitive_json,
                       m.has_sensitive_ip, m.has_payment
                FROM messages AS m
                JOIN conversation_segments AS segment ON segment.id = m.segment_id
                WHERE segment.source_category = 'support' AND m.pk > :last_pk
                ORDER BY m.pk
                LIMIT :batch_size
                """
                ),
                {"last_pk": last_pk, "batch_size": _BATCH_SIZE},
            )
            .mappings()
            .all()
        )
        if not rows:
            break

        for row in rows:
            message_pk = int(row["pk"])
            extra = _json_object(row["extra_json"], column="extra_json", message_pk=message_pk)
            existing_sensitive = _json_object(
                row["sensitive_json"], column="sensitive_json", message_pk=message_pk
            )
            public, extracted, has_extracted, _field_names = _partition(extra)
            merged_sensitive = (
                _merge_sensitive(existing_sensitive, extracted)
                if has_extracted
                else existing_sensitive
            )
            kinds = _sensitive_kinds(merged_sensitive)
            has_sensitive_ip = bool(row["has_sensitive_ip"]) or "ip" in kinds
            has_payment = bool(row["has_payment"]) or "payment" in kinds

            if (
                public != extra
                or merged_sensitive != existing_sensitive
                or has_sensitive_ip != bool(row["has_sensitive_ip"])
                or has_payment != bool(row["has_payment"])
            ):
                connection.execute(
                    sa.text(
                        """
                        UPDATE messages
                        SET extra_json = :extra_json,
                            sensitive_json = :sensitive_json,
                            has_sensitive_ip = :has_sensitive_ip,
                            has_payment = :has_payment
                        WHERE pk = :message_pk
                        """
                    ),
                    {
                        "extra_json": _dump_json(public),
                        "sensitive_json": _dump_json(merged_sensitive),
                        "has_sensitive_ip": has_sensitive_ip,
                        "has_payment": has_payment,
                        "message_pk": message_pk,
                    },
                )
            last_pk = message_pk


def downgrade() -> None:
    # Privacy relocation is intentionally irreversible. Sensitive values remain preserved in the
    # gated column, and restoring public duplicates would recreate the disclosure this fixes.
    pass
