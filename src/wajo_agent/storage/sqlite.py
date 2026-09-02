"""Versioned SQLite event store and preference-state projection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from wajo_agent.domain import (
    ApprovalRecord,
    ApprovalStatus,
    AuditEvent,
    FeedbackType,
    PreferenceState,
)
from wajo_agent.domain.models import new_id, utc_now

SCHEMA_VERSION = 2
FEEDBACK_LIST_ADAPTER = TypeAdapter(list[str])

MIGRATION_V1 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    stream_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    event_type TEXT NOT NULL CHECK (length(event_type) BETWEEN 1 AND 100),
    event_version INTEGER NOT NULL CHECK (event_version >= 1),
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    occurred_at TEXT NOT NULL,
    UNIQUE (stream_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_audit_events_stream_sequence
    ON audit_events (stream_id, sequence);

CREATE TRIGGER IF NOT EXISTS audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN
    SELECT RAISE(ABORT, 'audit events are append-only');
END;

CREATE TABLE IF NOT EXISTS preference_states (
    context_key TEXT PRIMARY KEY,
    alpha INTEGER NOT NULL CHECK (alpha >= 1),
    beta INTEGER NOT NULL CHECK (beta >= 1),
    observations INTEGER NOT NULL CHECK (observations >= 0),
    recent_feedback_json TEXT NOT NULL CHECK (json_valid(recent_feedback_json)),
    cooldown_remaining INTEGER NOT NULL CHECK (cooldown_remaining >= 0),
    updated_at TEXT NOT NULL
);

PRAGMA user_version = 1;
COMMIT;
"""

MIGRATION_V2 = """
BEGIN IMMEDIATE;

CREATE TABLE approvals (
    approval_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    proposal_version INTEGER NOT NULL CHECK (proposal_version >= 1),
    payload_hash TEXT NOT NULL CHECK (
        length(payload_hash) = 64 AND payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'granted', 'rejected', 'consumed', 'expired', 'invalidated')
    ),
    actor TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    granted_at TEXT,
    consumed_at TEXT,
    superseded_by_approval_id TEXT REFERENCES approvals(approval_id),
    CHECK (expires_at > created_at),
    CHECK (updated_at >= created_at),
    CHECK (status = 'consumed' OR consumed_at IS NULL),
    CHECK (status != 'consumed' OR consumed_at IS NOT NULL),
    CHECK (status NOT IN ('granted', 'consumed') OR (actor IS NOT NULL AND granted_at IS NOT NULL)),
    CHECK (status NOT IN ('rejected', 'invalidated') OR actor IS NOT NULL),
    CHECK (superseded_by_approval_id IS NULL OR status = 'invalidated'),
    CHECK (superseded_by_approval_id IS NULL OR superseded_by_approval_id != approval_id)
);

CREATE INDEX idx_approvals_proposal
    ON approvals (proposal_id, proposal_version);
CREATE INDEX idx_approvals_status_expiry
    ON approvals (status, expires_at);

PRAGMA user_version = 2;
COMMIT;
"""

MIGRATIONS = {
    1: MIGRATION_V1,
    2: MIGRATION_V2,
}


class StorageError(RuntimeError):
    """Base class for safe storage failures."""


class SchemaVersionError(StorageError):
    """The database schema cannot be safely opened by this application version."""


class DuplicateEventError(StorageError):
    """An event identifier has already been committed."""


class ApprovalNotFoundError(StorageError):
    """The requested approval does not exist."""


class ApprovalStateConflictError(StorageError):
    """An approval changed between observation and attempted transition."""


class DuplicateApprovalError(StorageError):
    """An approval identifier has already been committed."""


class SQLiteStore:
    """Single-process SQLite store with short explicit write transactions."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure()
            self._initialize_schema()
        except Exception:
            self._connection.close()
            raise

    def _configure(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")

    def _initialize_schema(self) -> None:
        version = self.schema_version
        if version > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"database schema {version} is newer than supported version {SCHEMA_VERSION}"
            )
        for target_version in range(version + 1, SCHEMA_VERSION + 1):
            migration = MIGRATIONS.get(target_version)
            if migration is None:
                raise SchemaVersionError(f"no migration path from schema {target_version - 1}")
            try:
                self._connection.executescript(migration)
            except sqlite3.DatabaseError as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise StorageError(
                    f"failed to migrate SQLite schema to version {target_version}"
                ) from exc

    @property
    def schema_version(self) -> int:
        row = self._connection.execute("PRAGMA user_version").fetchone()
        if row is None:
            raise StorageError("SQLite did not return a schema version")
        return int(row[0])

    @contextmanager
    def _write_transaction(self) -> Generator[None, None, None]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def append_event(
        self,
        *,
        stream_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        event_version: int = 1,
        event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> AuditEvent:
        """Append one immutable event and allocate its stream sequence atomically."""
        with self._write_transaction():
            return self._insert_event(
                stream_id=stream_id,
                event_type=event_type,
                payload=payload,
                event_version=event_version,
                event_id=event_id,
                occurred_at=occurred_at,
            )

    def _insert_event(
        self,
        *,
        stream_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        event_version: int,
        event_id: str | None,
        occurred_at: datetime | None,
    ) -> AuditEvent:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM audit_events WHERE stream_id = ?",
            (stream_id.strip(),),
        ).fetchone()
        if row is None:
            raise StorageError("could not allocate the next event sequence")

        event = AuditEvent(
            event_id=event_id or new_id("event"),
            stream_id=stream_id,
            sequence=int(row[0]),
            event_type=event_type,
            event_version=event_version,
            payload=payload,
            occurred_at=occurred_at or utc_now(),
        )
        payload_json = _canonical_json(event.payload)

        try:
            self._connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, stream_id, sequence, event_type, event_version,
                    payload_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.stream_id,
                    event.sequence,
                    event.event_type,
                    event.event_version,
                    payload_json,
                    event.occurred_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "event_id" in str(exc).casefold():
                raise DuplicateEventError(f"event already exists: {event.event_id}") from exc
            raise StorageError("audit event violated a database invariant") from exc
        return event

    def read_stream(self, stream_id: str, *, after_sequence: int = 0) -> tuple[AuditEvent, ...]:
        """Load one event stream in deterministic sequence order."""
        if after_sequence < 0:
            raise ValueError("after_sequence cannot be negative")
        rows = self._connection.execute(
            """
            SELECT event_id, stream_id, sequence, event_type, event_version,
                   payload_json, occurred_at
            FROM audit_events
            WHERE stream_id = ? AND sequence > ?
            ORDER BY sequence ASC
            """,
            (stream_id.strip(), after_sequence),
        ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def get_preference(self, context_key: str) -> PreferenceState:
        """Satisfy PreferenceRepository with a cold prior when no row exists."""
        cleaned_key = context_key.strip()
        row = self._connection.execute(
            """
            SELECT context_key, alpha, beta, observations,
                   recent_feedback_json, cooldown_remaining
            FROM preference_states
            WHERE context_key = ?
            """,
            (cleaned_key,),
        ).fetchone()
        if row is None:
            return PreferenceState(context_key=cleaned_key)
        return _preference_from_row(row)

    def save_preference(self, state: PreferenceState) -> None:
        """Persist one current preference projection in a short transaction."""
        with self._write_transaction():
            self._upsert_preference(state)

    def save_preference_and_append_event(
        self,
        state: PreferenceState,
        *,
        stream_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        event_version: int = 1,
        event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> AuditEvent:
        """Atomically commit a preference projection and its explaining audit event."""
        with self._write_transaction():
            self._upsert_preference(state)
            return self._insert_event(
                stream_id=stream_id,
                event_type=event_type,
                payload=payload,
                event_version=event_version,
                event_id=event_id,
                occurred_at=occurred_at,
            )

    def _upsert_preference(self, state: PreferenceState) -> None:
        feedback_json = _canonical_json([item.value for item in state.recent_feedback])
        self._connection.execute(
            """
            INSERT INTO preference_states (
                context_key, alpha, beta, observations,
                recent_feedback_json, cooldown_remaining, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(context_key) DO UPDATE SET
                alpha = excluded.alpha,
                beta = excluded.beta,
                observations = excluded.observations,
                recent_feedback_json = excluded.recent_feedback_json,
                cooldown_remaining = excluded.cooldown_remaining,
                updated_at = excluded.updated_at
            """,
            (
                state.context_key,
                state.alpha,
                state.beta,
                state.observations,
                feedback_json,
                state.cooldown_remaining,
                utc_now().isoformat(),
            ),
        )

    def create_approval(
        self,
        record: ApprovalRecord,
        *,
        event_payload: dict[str, JsonValue],
    ) -> ApprovalRecord:
        """Atomically persist a new approval request and its audit event."""
        if record.status != ApprovalStatus.PENDING:
            raise ValueError("new approval must start pending")
        with self._write_transaction():
            self._insert_approval(record)
            self._insert_event(
                stream_id=_approval_stream_id(record.approval_id),
                event_type="approval.requested",
                payload=event_payload,
                event_version=1,
                event_id=None,
                occurred_at=record.created_at,
            )
        return record

    def get_approval(self, approval_id: str) -> ApprovalRecord:
        """Load one approval or raise a typed not-found error."""
        row = self._connection.execute(
            """
            SELECT approval_id, proposal_id, proposal_version, payload_hash, status,
                   actor, created_at, expires_at, updated_at, granted_at, consumed_at,
                   superseded_by_approval_id
            FROM approvals
            WHERE approval_id = ?
            """,
            (approval_id.strip(),),
        ).fetchone()
        if row is None:
            raise ApprovalNotFoundError(f"approval not found: {approval_id.strip()}")
        return _approval_from_row(row)

    def transition_approval(
        self,
        record: ApprovalRecord,
        *,
        expected_statuses: tuple[ApprovalStatus, ...],
        event_type: str,
        event_payload: dict[str, JsonValue],
    ) -> ApprovalRecord:
        """Compare-and-set one approval state and append its event in one transaction."""
        with self._write_transaction():
            self._update_approval(record, expected_statuses=expected_statuses)
            self._insert_event(
                stream_id=_approval_stream_id(record.approval_id),
                event_type=event_type,
                payload=event_payload,
                event_version=1,
                event_id=None,
                occurred_at=record.updated_at,
            )
        return record

    def replace_approval_for_edit(
        self,
        invalidated: ApprovalRecord,
        replacement: ApprovalRecord,
        *,
        expected_statuses: tuple[ApprovalStatus, ...],
        invalidated_event_payload: dict[str, JsonValue],
        replacement_event_payload: dict[str, JsonValue],
    ) -> tuple[ApprovalRecord, ApprovalRecord]:
        """Invalidate an old approval and create its edited replacement atomically."""
        if invalidated.status != ApprovalStatus.INVALIDATED:
            raise ValueError("edited approval must invalidate the previous request")
        if invalidated.superseded_by_approval_id != replacement.approval_id:
            raise ValueError("old approval must reference its exact replacement")
        if replacement.status != ApprovalStatus.PENDING:
            raise ValueError("replacement approval must start pending")

        with self._write_transaction():
            self._insert_approval(replacement)
            self._update_approval(invalidated, expected_statuses=expected_statuses)
            self._insert_event(
                stream_id=_approval_stream_id(invalidated.approval_id),
                event_type="approval.invalidated",
                payload=invalidated_event_payload,
                event_version=1,
                event_id=None,
                occurred_at=invalidated.updated_at,
            )
            self._insert_event(
                stream_id=_approval_stream_id(replacement.approval_id),
                event_type="approval.requested",
                payload=replacement_event_payload,
                event_version=1,
                event_id=None,
                occurred_at=replacement.created_at,
            )
        return invalidated, replacement

    def _insert_approval(self, record: ApprovalRecord) -> None:
        try:
            self._connection.execute(
                """
                INSERT INTO approvals (
                    approval_id, proposal_id, proposal_version, payload_hash, status,
                    actor, created_at, expires_at, updated_at, granted_at, consumed_at,
                    superseded_by_approval_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _approval_values(record),
            )
        except sqlite3.IntegrityError as exc:
            if "approval_id" in str(exc).casefold():
                raise DuplicateApprovalError(
                    f"approval already exists: {record.approval_id}"
                ) from exc
            raise StorageError("approval violated a database invariant") from exc

    def _update_approval(
        self,
        record: ApprovalRecord,
        *,
        expected_statuses: tuple[ApprovalStatus, ...],
    ) -> None:
        if not expected_statuses:
            raise ValueError("approval transition requires an expected source status")
        placeholders = ", ".join("?" for _ in expected_statuses)
        cursor = self._connection.execute(
            f"""
            UPDATE approvals
            SET status = ?, actor = ?, updated_at = ?, granted_at = ?, consumed_at = ?,
                superseded_by_approval_id = ?
            WHERE approval_id = ?
              AND proposal_id = ?
              AND proposal_version = ?
              AND payload_hash = ?
              AND status IN ({placeholders})
            """,
            (
                record.status.value,
                record.actor,
                record.updated_at.isoformat(),
                _optional_time(record.granted_at),
                _optional_time(record.consumed_at),
                record.superseded_by_approval_id,
                record.approval_id,
                record.proposal_id,
                record.proposal_version,
                record.payload_hash,
                *(status.value for status in expected_statuses),
            ),
        )
        if cursor.rowcount == 1:
            return
        exists = self._connection.execute(
            "SELECT 1 FROM approvals WHERE approval_id = ?",
            (record.approval_id,),
        ).fetchone()
        if exists is None:
            raise ApprovalNotFoundError(f"approval not found: {record.approval_id}")
        raise ApprovalStateConflictError(
            f"approval is no longer in an expected state: {record.approval_id}"
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise StorageError("value is not valid canonical JSON") from exc


def _event_from_row(row: sqlite3.Row) -> AuditEvent:
    try:
        raw_payload: object = json.loads(str(row["payload_json"]))
        return AuditEvent.model_validate(
            {
                "event_id": str(row["event_id"]),
                "stream_id": str(row["stream_id"]),
                "sequence": int(row["sequence"]),
                "event_type": str(row["event_type"]),
                "event_version": int(row["event_version"]),
                "payload": raw_payload,
                "occurred_at": str(row["occurred_at"]),
            }
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StorageError("stored audit event is malformed") from exc


def _preference_from_row(row: sqlite3.Row) -> PreferenceState:
    try:
        feedback_items = FEEDBACK_LIST_ADAPTER.validate_json(str(row["recent_feedback_json"]))
        feedback = tuple(FeedbackType(item) for item in feedback_items)
        return PreferenceState(
            context_key=str(row["context_key"]),
            alpha=int(row["alpha"]),
            beta=int(row["beta"]),
            observations=int(row["observations"]),
            recent_feedback=feedback,
            cooldown_remaining=int(row["cooldown_remaining"]),
        )
    except (ValidationError, ValueError) as exc:
        raise StorageError("stored preference feedback is malformed") from exc


def _approval_values(record: ApprovalRecord) -> tuple[object, ...]:
    return (
        record.approval_id,
        record.proposal_id,
        record.proposal_version,
        record.payload_hash,
        record.status.value,
        record.actor,
        record.created_at.isoformat(),
        record.expires_at.isoformat(),
        record.updated_at.isoformat(),
        _optional_time(record.granted_at),
        _optional_time(record.consumed_at),
        record.superseded_by_approval_id,
    )


def _approval_from_row(row: sqlite3.Row) -> ApprovalRecord:
    try:
        return ApprovalRecord.model_validate(
            {
                "approval_id": str(row["approval_id"]),
                "proposal_id": str(row["proposal_id"]),
                "proposal_version": int(row["proposal_version"]),
                "payload_hash": str(row["payload_hash"]),
                "status": str(row["status"]),
                "actor": row["actor"],
                "created_at": str(row["created_at"]),
                "expires_at": str(row["expires_at"]),
                "updated_at": str(row["updated_at"]),
                "granted_at": row["granted_at"],
                "consumed_at": row["consumed_at"],
                "superseded_by_approval_id": row["superseded_by_approval_id"],
            }
        )
    except ValidationError as exc:
        raise StorageError("stored approval is malformed") from exc


def _approval_stream_id(approval_id: str) -> str:
    return f"approval:{approval_id}"


def _optional_time(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
