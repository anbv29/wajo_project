"""Versioned SQLite event store and preference-state projection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from wajo_agent.domain import AuditEvent, FeedbackType, PreferenceState
from wajo_agent.domain.models import new_id, utc_now

SCHEMA_VERSION = 1
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


class StorageError(RuntimeError):
    """Base class for safe storage failures."""


class SchemaVersionError(StorageError):
    """The database schema cannot be safely opened by this application version."""


class DuplicateEventError(StorageError):
    """An event identifier has already been committed."""


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
        if version == 0:
            try:
                self._connection.executescript(MIGRATION_V1)
            except sqlite3.DatabaseError as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise StorageError("failed to initialize SQLite schema") from exc
        elif version != SCHEMA_VERSION:
            raise SchemaVersionError(f"no migration path from schema {version}")

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
