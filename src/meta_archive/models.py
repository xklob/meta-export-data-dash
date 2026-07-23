from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )


class ArchiveSetting(Base, TimestampMixin):
    __tablename__ = "archive_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ImportRun(Base):
    __tablename__ = "import_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    parser_version: Mapped[str] = mapped_column(String(40), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    counters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    checkpoint_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    progress_current: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    progress_total: Mapped[int | None] = mapped_column(BigInteger)
    stage: Mapped[str | None] = mapped_column(String(120))
    checkpoint_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    counters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SourceChunk(Base):
    __tablename__ = "source_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    original_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class Blob(Base):
    __tablename__ = "blobs"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(255))
    extension: Mapped[str | None] = mapped_column(String(32))
    relative_path: Mapped[str | None] = mapped_column(Text, unique=True)
    state: Mapped[str] = mapped_column(String(24), default="catalogued", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class MediaDerivative(Base, TimestampMixin):
    __tablename__ = "media_derivatives"
    __table_args__ = (
        UniqueConstraint(
            "blob_sha256",
            "kind",
            "pipeline_version",
            name="uq_media_derivative_blob_kind_version",
        ),
        Index("ix_media_derivatives_blob_status", "blob_sha256", "status"),
        Index("ix_media_derivatives_kind_status", "kind", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    blob_sha256: Mapped[str] = mapped_column(
        ForeignKey("blobs.sha256", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    media_type: Mapped[str] = mapped_column(String(80), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    output_sha256: Mapped[str | None] = mapped_column(String(64))
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SourceFile(Base, TimestampMixin):
    __tablename__ = "source_files"
    __table_args__ = (
        UniqueConstraint("chunk_id", "relative_path", name="uq_source_file_coordinate"),
        Index("ix_source_files_normalized_path", "normalized_path"),
        Index("ix_source_files_sha256", "sha256"),
        Index("ix_source_files_category", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("source_chunks.id", ondelete="RESTRICT"), nullable=False
    )
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_path: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mode: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(255))
    extension: Mapped[str | None] = mapped_column(String(32))
    sha256: Mapped[str | None] = mapped_column(
        ForeignKey("blobs.sha256", ondelete="RESTRICT"), nullable=True
    )
    json_status: Mapped[str] = mapped_column(String(24), default="not_json", nullable=False)
    json_error: Mapped[str | None] = mapped_column(Text)
    scan_status: Mapped[str] = mapped_column(String(24), default="discovered", nullable=False)
    imported_parser_version: Mapped[str | None] = mapped_column(String(40))
    imported_sha256: Mapped[str | None] = mapped_column(String(64))


class SourceRecord(Base):
    __tablename__ = "source_records"
    __table_args__ = (
        UniqueConstraint("source_file_id", "json_pointer", name="uq_source_record_coordinate"),
        Index("ix_source_records_fingerprint", "logical_fingerprint"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("source_files.id", ondelete="RESTRICT"), nullable=False
    )
    json_pointer: Mapped[str] = mapped_column(Text, nullable=False)
    record_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    logical_fingerprint: Mapped[str | None] = mapped_column(String(64))
    duplicate_of_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_records.id", ondelete="SET NULL")
    )
    import_run_id: Mapped[str] = mapped_column(
        ForeignKey("import_runs.id", ondelete="RESTRICT"), nullable=False
    )


class CoverageArea(Base, TimestampMixin):
    __tablename__ = "coverage_areas"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    raw_count: Mapped[int | None] = mapped_column(BigInteger)
    logical_count: Mapped[int | None] = mapped_column(BigInteger)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_paths_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class IntegrityAnomaly(Base, TimestampMixin):
    __tablename__ = "integrity_anomalies"
    __table_args__ = (
        Index("ix_integrity_anomalies_code", "code"),
        Index("ix_integrity_anomalies_severity", "severity"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    code: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(24), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_files.id", ondelete="SET NULL")
    )
    source_record_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_records.id", ondelete="SET NULL")
    )
    conversation_segment_id: Mapped[str | None] = mapped_column(String(64))
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_type_activity", "conversation_type", "last_activity_ms"),
        Index("ix_conversations_message_count", "message_count"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title_raw: Mapped[str] = mapped_column(Text, nullable=False)
    title_display: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    title_search: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_type: Mapped[str] = mapped_column(String(24), nullable=False)
    primary_category: Mapped[str] = mapped_column(String(80), nullable=False)
    participant_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_activity_ms: Mapped[int | None] = mapped_column(BigInteger)
    last_activity_ms: Mapped[int | None] = mapped_column(BigInteger)
    message_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    sent_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    received_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    attachment_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    unique_attachment_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    attachment_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    active_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completeness: Mapped[str] = mapped_column(String(24), default="unknown", nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationSegment(Base):
    __tablename__ = "conversation_segments"
    __table_args__ = (
        Index("ix_conversation_segments_conversation", "conversation_id", "first_activity_ms"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    thread_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    title_raw: Mapped[str] = mapped_column(Text, nullable=False)
    title_display: Mapped[str] = mapped_column(Text, nullable=False)
    title_search: Mapped[str] = mapped_column(Text, nullable=False)
    source_category: Mapped[str] = mapped_column(String(80), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_still_participant: Mapped[bool | None] = mapped_column(Boolean)
    first_activity_ms: Mapped[int | None] = mapped_column(BigInteger)
    last_activity_ms: Mapped[int | None] = mapped_column(BigInteger)
    completeness: Mapped[str] = mapped_column(String(24), default="unknown", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name_raw: Mapped[str] = mapped_column(Text, nullable=False)
    name_display: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name_search: Mapped[str] = mapped_column(Text, nullable=False)
    is_owner: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class SegmentParticipant(Base):
    __tablename__ = "segment_participants"
    __table_args__ = (
        UniqueConstraint("segment_id", "ordinal", name="uq_segment_participant_ordinal"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_segments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="RESTRICT"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class ConversationCorrection(Base):
    __tablename__ = "conversation_corrections"
    __table_args__ = (Index("ix_conversation_corrections_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    before_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    after_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    catalog_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    result_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    revert_revision: Mapped[str | None] = mapped_column(String(64))
    effect_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )
    reverted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationCorrectionSegment(Base):
    __tablename__ = "conversation_correction_segments"
    __table_args__ = (
        UniqueConstraint(
            "correction_id",
            "segment_id",
            name="uq_conversation_correction_segment",
        ),
        Index("ix_conversation_correction_segments_segment", "segment_id", "correction_id"),
        Index(
            "ix_conversation_correction_segments_conversations",
            "before_conversation_id",
            "after_conversation_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    correction_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_corrections.id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    thread_path: Mapped[str] = mapped_column(Text, nullable=False)
    before_conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    after_conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_correction_id: Mapped[str | None] = mapped_column(String(36))


class ConversationCorrectionAudit(Base):
    __tablename__ = "conversation_correction_audits"
    __table_args__ = (
        Index(
            "ix_conversation_correction_audits_correction_created",
            "correction_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    correction_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_corrections.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class ConversationSegmentOverride(Base):
    __tablename__ = "conversation_segment_overrides"
    __table_args__ = (Index("ix_conversation_segment_overrides_conversation", "conversation_id"),)

    segment_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_segments.id", ondelete="CASCADE"), primary_key=True
    )
    # The provisional ID is deliberately not a foreign key: an empty provisional conversation is
    # removed while a correction is active and recreated from the audit snapshot on revert.
    base_conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="RESTRICT"), nullable=False
    )
    correction_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_corrections.id", ondelete="RESTRICT"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("source_file_id", "json_pointer", name="uq_message_source_coordinate"),
        Index("ix_messages_conversation_time", "conversation_id", "timestamp_ms", "id"),
        Index("ix_messages_segment_time", "segment_id", "timestamp_ms", "id"),
        Index("ix_messages_deleted_time_id", "deleted_at", "timestamp_ms", "id"),
        Index("ix_messages_sender", "sender_display"),
        Index("ix_messages_deleted", "deleted_at"),
    )

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    source_record_id: Mapped[str] = mapped_column(
        ForeignKey("source_records.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("source_files.id", ondelete="RESTRICT"), nullable=False
    )
    json_pointer: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_array_index: Mapped[int] = mapped_column(Integer, nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    segment_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_segments.id", ondelete="CASCADE"), nullable=False
    )
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sender_raw: Mapped[str] = mapped_column(Text, nullable=False)
    sender_display: Mapped[str] = mapped_column(Text, nullable=False)
    sender_search: Mapped[str] = mapped_column(Text, nullable=False)
    sender_is_owner: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    content_raw: Mapped[str | None] = mapped_column(Text)
    content_display: Mapped[str | None] = mapped_column(Text)
    content_search: Mapped[str | None] = mapped_column(Text)
    message_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    call_duration_seconds: Mapped[int | None] = mapped_column(Integer)
    is_missed_call: Mapped[bool | None] = mapped_column(Boolean)
    has_payment: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_sensitive_ip: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_transcription: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    transcription_raw: Mapped[str | None] = mapped_column(Text)
    transcription_display: Mapped[str | None] = mapped_column(Text)
    transcription_search: Mapped[str | None] = mapped_column(Text)
    is_unsent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_unavailable_body: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    logical_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    duplicate_of_id: Mapped[str | None] = mapped_column(String(64))
    extra_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    sensitive_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AttachmentOccurrence(Base):
    __tablename__ = "attachment_occurrences"
    __table_args__ = (
        UniqueConstraint("message_id", "kind", "ordinal", name="uq_attachment_occurrence"),
        Index("ix_attachment_conversation_time", "conversation_id", "timestamp_ms"),
        Index("ix_attachment_blob", "blob_sha256"),
        Index("ix_attachment_availability", "availability"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=True
    )
    activity_record_id: Mapped[str | None] = mapped_column(
        ForeignKey("activity_records.id", ondelete="CASCADE"), nullable=True
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True
    )
    source_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_files.id", ondelete="RESTRICT")
    )
    resolved_source_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_files.id", ondelete="RESTRICT")
    )
    blob_sha256: Mapped[str | None] = mapped_column(ForeignKey("blobs.sha256", ondelete="RESTRICT"))
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    media_type: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    timestamp_ms: Mapped[int | None] = mapped_column(BigInteger)
    availability: Mapped[str] = mapped_column(String(24), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MessageReaction(Base):
    __tablename__ = "message_reactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_raw: Mapped[str] = mapped_column(Text, nullable=False)
    actor_display: Mapped[str] = mapped_column(Text, nullable=False)
    reaction_raw: Mapped[str] = mapped_column(Text, nullable=False)
    reaction_display: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[int | None] = mapped_column(BigInteger)
    extra_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class AttachmentCandidate(Base):
    __tablename__ = "attachment_candidates"
    __table_args__ = (
        UniqueConstraint("attachment_id", "source_file_id", name="uq_attachment_candidate_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attachment_id: Mapped[str] = mapped_column(
        ForeignKey("attachment_occurrences.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("source_files.id", ondelete="RESTRICT"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    match_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ActivityRecord(Base):
    __tablename__ = "activity_records"
    __table_args__ = (
        UniqueConstraint("source_file_id", "json_pointer", name="uq_activity_source_coordinate"),
        Index("ix_activity_category_time", "category", "timestamp_ms"),
        Index("ix_activity_deleted_time_id", "deleted_at", "timestamp_ms", "id"),
        Index("ix_activity_subtype", "subtype"),
    )

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    source_record_id: Mapped[str] = mapped_column(
        ForeignKey("source_records.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("activity_records.id", ondelete="CASCADE"), nullable=True
    )
    source_file_id: Mapped[int] = mapped_column(
        ForeignKey("source_files.id", ondelete="RESTRICT"), nullable=False
    )
    json_pointer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    subtype: Mapped[str] = mapped_column(String(80), nullable=False)
    source_fbid: Mapped[str | None] = mapped_column(String(100), index=True)
    timestamp_ms: Mapped[int | None] = mapped_column(BigInteger)
    title_raw: Mapped[str | None] = mapped_column(Text)
    title_display: Mapped[str | None] = mapped_column(Text)
    title_search: Mapped[str | None] = mapped_column(Text)
    content_raw: Mapped[str | None] = mapped_column(Text)
    content_display: Mapped[str | None] = mapped_column(Text)
    content_search: Mapped[str | None] = mapped_column(Text)
    actor_raw: Mapped[str | None] = mapped_column(Text)
    actor_display: Mapped[str | None] = mapped_column(Text)
    actor_search: Mapped[str | None] = mapped_column(Text)
    target_uri: Mapped[str | None] = mapped_column(Text)
    availability: Mapped[str] = mapped_column(String(24), default="available", nullable=False)
    extra_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SearchWindow(Base, TimestampMixin):
    __tablename__ = "search_windows"
    __table_args__ = (
        Index("ix_search_windows_entity_time", "entity_type", "first_timestamp_ms"),
        Index("ix_search_windows_conversation_time", "conversation_id", "first_timestamp_ms"),
        Index("ix_search_windows_embedding", "embedding_config_hash"),
        Index("ix_search_windows_generation", "build_generation"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    window_version: Mapped[str] = mapped_column(String(40), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversation_segments.id", ondelete="CASCADE")
    )
    activity_id: Mapped[str | None] = mapped_column(
        ForeignKey("activity_records.id", ondelete="CASCADE")
    )
    audio_segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("audio_transcript_segments.id", ondelete="CASCADE")
    )
    attachment_extraction_id: Mapped[str | None] = mapped_column(
        ForeignKey("attachment_text_extractions.id", ondelete="CASCADE")
    )
    attachment_occurrence_id: Mapped[str | None] = mapped_column(
        ForeignKey("attachment_occurrences.id", ondelete="CASCADE")
    )
    first_message_id: Mapped[str | None] = mapped_column(String(64))
    last_message_id: Mapped[str | None] = mapped_column(String(64))
    first_timestamp_ms: Mapped[int | None] = mapped_column(BigInteger)
    last_timestamp_ms: Mapped[int | None] = mapped_column(BigInteger)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False)
    approx_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text_display: Mapped[str] = mapped_column(Text, nullable=False)
    text_search: Mapped[str] = mapped_column(Text, nullable=False)
    source_category: Mapped[str | None] = mapped_column(String(80))
    participants_search: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_ids_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    build_generation: Mapped[str] = mapped_column(String(36), nullable=False)
    embedding_config_hash: Mapped[str | None] = mapped_column(String(64))
    embedding_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SearchWindowMessage(Base):
    __tablename__ = "search_window_messages"
    __table_args__ = (
        UniqueConstraint("window_id", "ordinal", name="uq_search_window_message_ordinal"),
        UniqueConstraint("window_id", "message_id", name="uq_search_window_message"),
        Index("ix_search_window_messages_message", "message_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    window_id: Mapped[str] = mapped_column(
        ForeignKey("search_windows.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class SemanticIndexConfig(Base, TimestampMixin):
    __tablename__ = "semantic_index_configs"
    __table_args__ = (Index("ix_semantic_index_configs_active", "active"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    backend: Mapped[str] = mapped_column(String(40), nullable=False)
    backend_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(24), nullable=False)
    document_prefix: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    query_prefix: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    vector_table: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    window_version: Mapped[str] = mapped_column(String(40), nullable=False)
    window_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SemanticWindowEmbedding(Base):
    """Durable per-index membership for one embedded search window.

    ``SearchWindow.embedding_config_hash`` predates side-by-side candidate builds and can point
    at only one configuration.  This relation is the authoritative checkpoint: a candidate can
    be built and evaluated without taking membership away from the currently active index.
    """

    __tablename__ = "semantic_window_embeddings"
    __table_args__ = (Index("ix_semantic_window_embeddings_window", "window_id"),)

    config_id: Mapped[str] = mapped_column(
        ForeignKey("semantic_index_configs.id", ondelete="CASCADE"), primary_key=True
    )
    window_id: Mapped[str] = mapped_column(
        ForeignKey("search_windows.id", ondelete="CASCADE"), primary_key=True
    )
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class RagAudit(Base):
    __tablename__ = "rag_audits"
    __table_args__ = (Index("ix_rag_audits_created_at", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    query_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_mode: Mapped[str] = mapped_column(String(24), nullable=False)
    retrieved_ids_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    citation_ids_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    excerpt_hashes_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    remote_response_id: Mapped[str | None] = mapped_column(String(255))
    usage_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class SavedSearch(Base, TimestampMixin):
    __tablename__ = "saved_searches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    query_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class Star(Base):
    """A durable user bookmark for one stable archive identity.

    Target foreign keys are intentionally omitted. Importers replace deterministic
    catalog rows transactionally, and a cascading foreign key would erase a star
    during that replacement instead of reconnecting it to the recreated identity.
    """

    __tablename__ = "stars"
    __table_args__ = (
        CheckConstraint(
            "target_type IN ("
            "'conversation', 'message', 'activity', 'attachment', 'blob', "
            "'messenger_contact'"
            ")",
            name="ck_stars_target_type",
        ),
        UniqueConstraint("target_type", "target_id", name="uq_stars_target"),
        Index("ix_stars_starred_at_id", "starred_at", "id"),
        Index("ix_stars_target_type_starred_at_id", "target_type", "starred_at", "id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(128), nullable=False)
    starred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class RelevanceFeedback(Base):
    __tablename__ = "relevance_feedback"
    __table_args__ = (
        Index("ix_relevance_feedback_query_result", "query_sha256", "result_id"),
        Index(
            "uq_relevance_feedback_query_mode_scope_result",
            "query_sha256",
            "search_mode",
            "filter_scope_sha256",
            "result_id",
            unique=True,
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_mode: Mapped[str] = mapped_column(String(24), nullable=False)
    filter_scope_sha256: Mapped[str] = mapped_column(
        String(64),
        default="44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        server_default="44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        nullable=False,
    )
    filter_scope_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        server_default="{}",
        nullable=False,
    )
    result_id: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[str] = mapped_column(String(24), nullable=False)
    evaluation_split: Mapped[str | None] = mapped_column(String(24))
    evaluation_intent: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class AudioTranscription(Base, TimestampMixin):
    __tablename__ = "audio_transcriptions"
    __table_args__ = (
        UniqueConstraint(
            "blob_sha256",
            "pipeline_version",
            "model_revision",
            name="uq_audio_transcription_blob_pipeline_model",
        ),
        Index("ix_audio_transcriptions_status", "status"),
        Index("ix_audio_transcriptions_blob", "blob_sha256"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    blob_sha256: Mapped[str] = mapped_column(
        ForeignKey("blobs.sha256", ondelete="CASCADE"), nullable=False
    )
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    device: Mapped[str] = mapped_column(String(32), nullable=False)
    compute_type: Mapped[str] = mapped_column(String(32), nullable=False)
    language: Mapped[str | None] = mapped_column(String(32))
    language_probability: Mapped[float | None] = mapped_column(Float)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger)
    text_display: Mapped[str | None] = mapped_column(Text)
    text_search: Mapped[str | None] = mapped_column(Text)
    segment_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    index_eligible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    warnings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AudioTranscriptSegment(Base):
    __tablename__ = "audio_transcript_segments"
    __table_args__ = (
        UniqueConstraint("transcription_id", "ordinal", name="uq_audio_segment_ordinal"),
        Index("ix_audio_segments_transcription_time", "transcription_id", "start_ms"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    transcription_id: Mapped[str] = mapped_column(
        ForeignKey("audio_transcriptions.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text_display: Mapped[str] = mapped_column(Text, nullable=False)
    text_search: Mapped[str] = mapped_column(Text, nullable=False)
    avg_log_probability: Mapped[float | None] = mapped_column(Float)
    no_speech_probability: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    words_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class AttachmentTextExtraction(Base, TimestampMixin):
    __tablename__ = "attachment_text_extractions"
    __table_args__ = (
        UniqueConstraint(
            "blob_sha256", "pipeline_version", name="uq_attachment_text_blob_pipeline"
        ),
        Index("ix_attachment_text_status", "status"),
        Index("ix_attachment_text_blob", "blob_sha256"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    blob_sha256: Mapped[str] = mapped_column(
        ForeignKey("blobs.sha256", ondelete="CASCADE"), nullable=False
    )
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    extractor_name: Mapped[str] = mapped_column(String(80), nullable=False)
    extractor_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    filename: Mapped[str | None] = mapped_column(Text)
    media_type: Mapped[str | None] = mapped_column(String(255))
    text_display: Mapped[str | None] = mapped_column(Text)
    text_search: Mapped[str | None] = mapped_column(Text)
    character_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    index_eligible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sandboxed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sandbox_status: Mapped[str] = mapped_column(String(40), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    limits_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    warnings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NlpAnnotation(Base):
    __tablename__ = "nlp_annotations"
    __table_args__ = (
        UniqueConstraint(
            "target_type",
            "target_id",
            "task",
            "pipeline_version",
            "model_name",
            "model_revision",
            "input_sha256",
            name="uq_nlp_annotation_versioned_target",
        ),
        Index("ix_nlp_annotations_task_label", "task", "predicted_label"),
        Index("ix_nlp_annotations_target", "target_type", "target_id"),
        Index("ix_nlp_annotations_active", "active"),
        Index(
            "ix_nlp_annotations_active_completed_task_job",
            "active",
            "status",
            "task",
            "job_id",
        ),
        Index(
            "ix_nlp_annotations_inference_reuse",
            "task",
            "pipeline_version",
            "model_name",
            "model_revision",
            "input_sha256",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    language: Mapped[str | None] = mapped_column(String(32))
    predicted_label: Mapped[str | None] = mapped_column(String(80))
    confidence: Mapped[float | None] = mapped_column(Float)
    probabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    attribution_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    warnings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class NlpHourlyRollup(Base):
    __tablename__ = "nlp_hourly_rollups"
    __table_args__ = (
        UniqueConstraint(
            "task",
            "hour_start_ms",
            "direction",
            "pipeline_version",
            "model_name",
            "model_revision",
            "configuration_sha256",
            "predicted_label",
            name="uq_nlp_hourly_rollup_versioned_bucket",
        ),
        Index("ix_nlp_hourly_rollups_active_hour", "active", "hour_start_ms"),
        Index("ix_nlp_hourly_rollups_task_direction", "task", "direction"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    hour_start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    direction: Mapped[str] = mapped_column(String(24), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    configuration_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(80), nullable=False)
    sample_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    confidence_sum: Mapped[float] = mapped_column(Float, nullable=False)
    probability_sums_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class ConflictLabel(Base):
    __tablename__ = "conflict_labels"
    __table_args__ = (Index("ix_conflict_labels_window_split", "window_id", "evaluation_split"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    window_id: Mapped[str] = mapped_column(
        ForeignKey("search_windows.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    evaluation_split: Mapped[str] = mapped_column(String(24), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class ConflictCalibration(Base, TimestampMixin):
    __tablename__ = "conflict_calibrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    direct_conflict_precision: Mapped[float | None] = mapped_column(Float)
    macro_f1: Mapped[float | None] = mapped_column(Float)
    aggregation_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    warnings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConflictAggregateRun(Base, TimestampMixin):
    """A versioned, calibration-gated full-corpus conflict classification run.

    Deliberately contains no person, participant, conversation, or relationship
    dimensions.  Aggregate conflict is a corpus/time signal only.
    """

    __tablename__ = "conflict_aggregate_runs"
    __table_args__ = (
        UniqueConstraint("configuration_sha256", name="uq_conflict_aggregate_run_config"),
        Index("ix_conflict_aggregate_runs_active_status", "active", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_revision: Mapped[str] = mapped_column(String(80), nullable=False)
    calibration_id: Mapped[str | None] = mapped_column(
        ForeignKey("conflict_calibrations.id", ondelete="SET NULL")
    )
    semantic_config_id: Mapped[str | None] = mapped_column(
        ForeignKey("semantic_index_configs.id", ondelete="SET NULL")
    )
    corpus_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    configuration_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    window_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    scored_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    direct_conflict_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"))
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    warnings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConflictWindowScore(Base):
    """Per-window model output for an aggregate run; inactive until publication."""

    __tablename__ = "conflict_window_scores"
    __table_args__ = (
        UniqueConstraint("run_id", "window_id", name="uq_conflict_score_run_window"),
        Index("ix_conflict_window_scores_active_time", "active", "first_timestamp_ms"),
        Index("ix_conflict_window_scores_run_label", "run_id", "predicted_label"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("conflict_aggregate_runs.id", ondelete="CASCADE"), nullable=False
    )
    window_id: Mapped[str] = mapped_column(
        ForeignKey("search_windows.id", ondelete="CASCADE"), nullable=False
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    first_timestamp_ms: Mapped[int | None] = mapped_column(BigInteger)
    predicted_label: Mapped[str] = mapped_column(String(32), nullable=False)
    direct_conflict_score: Mapped[float] = mapped_column(Float, nullable=False)
    scores_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class ConflictHourlyRollup(Base):
    """Time-only conflict counts derived from one completed aggregate run."""

    __tablename__ = "conflict_hourly_rollups"
    __table_args__ = (
        UniqueConstraint("run_id", "hour_start_ms", name="uq_conflict_rollup_run_hour"),
        Index("ix_conflict_hourly_rollups_active_hour", "active", "hour_start_ms"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("conflict_aggregate_runs.id", ondelete="CASCADE"), nullable=False
    )
    hour_start_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sample_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    direct_conflict_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )


class DeletionOperation(Base, TimestampMixin):
    __tablename__ = "deletion_operations"
    __table_args__ = (Index("ix_deletion_operations_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    selector_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    impact_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    catalog_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    selection_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    commit_token_hash: Mapped[str | None] = mapped_column(String(64))
    purge_token_hash: Mapped[str | None] = mapped_column(String(64))
    purge_revision: Mapped[str | None] = mapped_column(String(64))
    purge_plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    manifest_relative_path: Mapped[str | None] = mapped_column(Text)
    item_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    restored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DeletionItem(Base):
    __tablename__ = "deletion_items"
    __table_args__ = (
        UniqueConstraint(
            "operation_id",
            "entity_type",
            "entity_id",
            name="uq_deletion_item_operation_entity",
        ),
        Index("ix_deletion_items_entity", "entity_type", "entity_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("deletion_operations.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(24), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DeletionAudit(Base):
    __tablename__ = "deletion_audits"
    __table_args__ = (Index("ix_deletion_audits_operation_created", "operation_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("deletion_operations.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    scope: Mapped[str | None] = mapped_column(String(32))
    outcome: Mapped[str] = mapped_column(String(24), nullable=False)
    detail_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False
    )
