"""Versioned SQLite event store and preference-state projection."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from wajo_agent.domain import (
    AgentOutcome,
    ApprovalRecord,
    ApprovalStatus,
    AuditEvent,
    AutonomyTier,
    ExecutionCommand,
    ExecutionRecord,
    ExecutionState,
    FeedbackRecord,
    FeedbackSubmission,
    FeedbackType,
    OutcomeRoute,
    PreferenceState,
)
from wajo_agent.domain.models import new_id, utc_now

SCHEMA_VERSION = 6
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

MIGRATION_V3 = """
BEGIN IMMEDIATE;

CREATE TABLE executions (
    execution_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE CHECK (
        length(idempotency_key) = 64 AND idempotency_key NOT GLOB '*[^0-9a-f]*'
    ),
    command_id TEXT NOT NULL UNIQUE,
    proposal_id TEXT NOT NULL,
    proposal_version INTEGER NOT NULL CHECK (proposal_version >= 1),
    approval_id TEXT REFERENCES approvals(approval_id),
    state TEXT NOT NULL CHECK (state IN ('executing', 'succeeded', 'failed_safe', 'unknown')),
    command_json TEXT NOT NULL CHECK (json_valid(command_json)),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 1),
    detail TEXT NOT NULL CHECK (length(detail) BETWEEN 1 AND 1000),
    provider_operation_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (updated_at >= created_at),
    CHECK (started_at >= created_at),
    CHECK (state = 'executing' OR completed_at IS NOT NULL),
    CHECK (state != 'executing' OR completed_at IS NULL),
    CHECK (completed_at IS NULL OR completed_at >= started_at)
);

CREATE INDEX idx_executions_proposal
    ON executions (proposal_id, proposal_version);
CREATE INDEX idx_executions_state
    ON executions (state, updated_at);

PRAGMA user_version = 3;
COMMIT;
"""

MIGRATION_V4 = """
BEGIN IMMEDIATE;

CREATE TABLE agent_runs (
    run_id TEXT PRIMARY KEY,
    mailbox_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    email_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'completed')),
    outcome_route TEXT,
    decision_tier TEXT,
    execution_state TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (mailbox_id, provider_message_id),
    CHECK (updated_at >= created_at),
    CHECK (status != 'completed' OR (outcome_route IS NOT NULL AND decision_tier IS NOT NULL))
);

CREATE INDEX idx_agent_runs_status
    ON agent_runs (status, updated_at);

PRAGMA user_version = 4;
COMMIT;
"""

MIGRATION_V5 = """
BEGIN IMMEDIATE;

CREATE TABLE feedback_records (
    feedback_id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE CHECK (
        length(dedupe_key) = 64 AND dedupe_key NOT GLOB '*[^0-9a-f]*'
    ),
    decision_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    proposal_version INTEGER NOT NULL CHECK (proposal_version >= 1),
    context_key TEXT NOT NULL,
    feedback_type TEXT NOT NULL CHECK (
        feedback_type IN ('approved', 'correct', 'edited', 'rejected', 'undone')
    ),
    actor TEXT NOT NULL,
    source_reference TEXT NOT NULL,
    record_json TEXT NOT NULL CHECK (json_valid(record_json)),
    created_at TEXT NOT NULL
);

CREATE INDEX idx_feedback_context_created
    ON feedback_records (context_key, created_at);
CREATE INDEX idx_feedback_decision
    ON feedback_records (decision_id, proposal_version);

PRAGMA user_version = 5;
COMMIT;
"""

MIGRATION_V6 = """
BEGIN IMMEDIATE;

CREATE TABLE agent_outcomes (
    run_id TEXT PRIMARY KEY REFERENCES agent_runs(run_id),
    email_id TEXT NOT NULL,
    provider_message_id TEXT NOT NULL,
    decision_id TEXT NOT NULL UNIQUE,
    approval_id TEXT UNIQUE,
    outcome_json TEXT NOT NULL CHECK (json_valid(outcome_json)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (updated_at >= created_at)
);

CREATE INDEX idx_agent_outcomes_provider_message
    ON agent_outcomes (provider_message_id);

PRAGMA user_version = 6;
COMMIT;
"""

MIGRATIONS = {
    1: MIGRATION_V1,
    2: MIGRATION_V2,
    3: MIGRATION_V3,
    4: MIGRATION_V4,
    5: MIGRATION_V5,
    6: MIGRATION_V6,
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


class ExecutionStateConflictError(StorageError):
    """An execution changed between observation and attempted transition."""


class AgentRunStateConflictError(StorageError):
    """An agent run changed or completed before the requested transition."""


class FeedbackConflictError(StorageError):
    """A feedback identity was reused for contradictory evidence."""


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
        except sqlite3.DatabaseError as exc:
            if self._connection.in_transaction:
                with suppress(sqlite3.DatabaseError):
                    self._connection.execute("ROLLBACK")
            raise StorageError("SQLite write transaction failed") from exc
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

    def read_recent_events(self, *, limit: int = 100) -> tuple[AuditEvent, ...]:
        """Load the newest audit events across streams, newest first."""
        if not 1 <= limit <= 1_000:
            raise ValueError("event limit must be between 1 and 1000")
        rows = self._connection.execute(
            """
            SELECT event_id, stream_id, sequence, event_type, event_version,
                   payload_json, occurred_at
            FROM audit_events
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (limit,),
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

    def list_preferences(self, *, limit: int = 100) -> tuple[PreferenceState, ...]:
        """List learned contexts with the most observed evidence first."""
        if not 1 <= limit <= 1_000:
            raise ValueError("preference limit must be between 1 and 1000")
        rows = self._connection.execute(
            """
            SELECT context_key, alpha, beta, observations,
                   recent_feedback_json, cooldown_remaining
            FROM preference_states
            ORDER BY observations DESC, context_key ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(_preference_from_row(row) for row in rows)

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

    def get_execution_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> ExecutionRecord | None:
        """Return the durable execution for an effect key, when one exists."""
        row = self._connection.execute(
            """
            SELECT execution_id, idempotency_key, command_json, state, attempt_count,
                   detail, provider_operation_id, created_at, updated_at, started_at,
                   completed_at
            FROM executions
            WHERE idempotency_key = ?
            """,
            (idempotency_key.strip(),),
        ).fetchone()
        return None if row is None else _execution_from_row(row)

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Return one durable execution by its application identity."""
        row = self._connection.execute(
            """
            SELECT execution_id, idempotency_key, command_json, state, attempt_count,
                   detail, provider_operation_id, created_at, updated_at, started_at,
                   completed_at
            FROM executions
            WHERE execution_id = ?
            """,
            (execution_id.strip(),),
        ).fetchone()
        return None if row is None else _execution_from_row(row)

    def claim_execution(
        self,
        record: ExecutionRecord,
        *,
        consumed_approval: ApprovalRecord | None,
    ) -> tuple[ExecutionRecord, bool]:
        """Claim a new effect and consume ASK authority in the same transaction."""
        if record.state != ExecutionState.EXECUTING:
            raise ValueError("new execution claim must be in EXECUTING state")
        if record.command.authorized_tier == AutonomyTier.ASK:
            if consumed_approval is None:
                raise ValueError("ASK execution claim requires consumed approval evidence")
            if consumed_approval.status != ApprovalStatus.CONSUMED:
                raise ValueError("execution claim requires a prepared consumed approval")
            if record.command.approval_id != consumed_approval.approval_id:
                raise ValueError("command and consumed approval identities differ")
            if (
                record.command.proposal_id != consumed_approval.proposal_id
                or record.command.proposal_version != consumed_approval.proposal_version
            ):
                raise ValueError("command and consumed approval proposals differ")
        elif consumed_approval is not None or record.command.approval_id is not None:
            raise ValueError("non-ASK execution cannot consume an approval")

        with self._write_transaction():
            existing = self.get_execution_by_idempotency_key(record.command.idempotency_key)
            if existing is not None:
                return existing, False

            if consumed_approval is not None:
                self._update_approval(
                    consumed_approval,
                    expected_statuses=(ApprovalStatus.GRANTED,),
                )
                self._insert_event(
                    stream_id=_approval_stream_id(consumed_approval.approval_id),
                    event_type="approval.consumed",
                    payload={
                        "approval_id": consumed_approval.approval_id,
                        "payload_hash": consumed_approval.payload_hash,
                        "execution_id": record.execution_id,
                    },
                    event_version=1,
                    event_id=None,
                    occurred_at=consumed_approval.updated_at,
                )

            self._insert_execution(record)
            self._insert_event(
                stream_id=_execution_stream_id(record.execution_id),
                event_type="execution.started",
                payload={
                    "execution_id": record.execution_id,
                    "idempotency_key": record.command.idempotency_key,
                    "proposal_id": record.command.proposal_id,
                    "proposal_version": record.command.proposal_version,
                    "action_type": record.command.action_type.value,
                },
                event_version=1,
                event_id=None,
                occurred_at=record.started_at,
            )
        return record, True

    def complete_execution(self, record: ExecutionRecord) -> ExecutionRecord:
        """Move one claimed execution to a terminal state and append its result event."""
        if record.state == ExecutionState.EXECUTING:
            raise ValueError("execution completion requires a terminal state")
        with self._write_transaction():
            cursor = self._connection.execute(
                """
                UPDATE executions
                SET state = ?, detail = ?, provider_operation_id = ?, updated_at = ?,
                    completed_at = ?
                WHERE execution_id = ?
                  AND idempotency_key = ?
                  AND state = 'executing'
                """,
                (
                    record.state.value,
                    record.detail,
                    record.provider_operation_id,
                    record.updated_at.isoformat(),
                    _optional_time(record.completed_at),
                    record.execution_id,
                    record.command.idempotency_key,
                ),
            )
            if cursor.rowcount != 1:
                raise ExecutionStateConflictError(
                    f"execution is no longer active: {record.execution_id}"
                )
            self._insert_event(
                stream_id=_execution_stream_id(record.execution_id),
                event_type=f"execution.{record.state.value}",
                payload={
                    "execution_id": record.execution_id,
                    "state": record.state.value,
                    "detail": record.detail,
                    "provider_operation_id": record.provider_operation_id,
                },
                event_version=1,
                event_id=None,
                occurred_at=record.completed_at or record.updated_at,
            )
        return record

    def _insert_execution(self, record: ExecutionRecord) -> None:
        command_json = _canonical_json(record.command.model_dump(mode="json"))
        try:
            self._connection.execute(
                """
                INSERT INTO executions (
                    execution_id, idempotency_key, command_id, proposal_id,
                    proposal_version, approval_id, state, command_json, attempt_count,
                    detail, provider_operation_id, created_at, updated_at, started_at,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.execution_id,
                    record.command.idempotency_key,
                    record.command.command_id,
                    record.command.proposal_id,
                    record.command.proposal_version,
                    record.command.approval_id,
                    record.state.value,
                    command_json,
                    record.attempt_count,
                    record.detail,
                    record.provider_operation_id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.started_at.isoformat(),
                    _optional_time(record.completed_at),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ExecutionStateConflictError(
                "execution identity or idempotency key already exists"
            ) from exc

    def claim_agent_run(
        self,
        *,
        run_id: str,
        mailbox_id: str,
        provider_message_id: str,
        email_id: str,
        source: str,
        occurred_at: datetime,
    ) -> tuple[str, bool]:
        """Claim one provider message before planning and reject duplicate deliveries."""
        identities = tuple(
            value.strip() for value in (run_id, mailbox_id, provider_message_id, email_id)
        )
        if any(not value for value in identities):
            raise ValueError("agent run identities cannot be blank")
        cleaned_run, cleaned_mailbox, cleaned_provider, cleaned_email = identities
        with self._write_transaction():
            existing = self._connection.execute(
                """
                SELECT run_id
                FROM agent_runs
                WHERE mailbox_id = ? AND provider_message_id = ?
                """,
                (cleaned_mailbox, cleaned_provider),
            ).fetchone()
            if existing is not None:
                return str(existing["run_id"]), False

            self._connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, mailbox_id, provider_message_id, email_id, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'processing', ?, ?)
                """,
                (
                    cleaned_run,
                    cleaned_mailbox,
                    cleaned_provider,
                    cleaned_email,
                    occurred_at.isoformat(),
                    occurred_at.isoformat(),
                ),
            )
            self._insert_event(
                stream_id=f"run:{cleaned_run}",
                event_type="email.received",
                payload={
                    "email_id": cleaned_email,
                    "provider_message_id": cleaned_provider,
                    "source": source,
                },
                event_version=1,
                event_id=None,
                occurred_at=occurred_at,
            )
        return cleaned_run, True

    def complete_agent_run(
        self,
        *,
        run_id: str,
        route: OutcomeRoute,
        decision_tier: AutonomyTier,
        execution_state: ExecutionState | None,
        occurred_at: datetime,
    ) -> None:
        """Complete a claimed run and append its final event atomically."""
        with self._write_transaction():
            self._complete_agent_run(
                run_id=run_id,
                route=route,
                decision_tier=decision_tier,
                execution_state=execution_state,
                occurred_at=occurred_at,
            )

    def complete_agent_run_with_outcome(
        self,
        outcome: AgentOutcome,
        *,
        occurred_at: datetime,
    ) -> None:
        """Atomically complete a run and save its typed CLI/read-model snapshot."""
        with self._write_transaction():
            self._complete_agent_run(
                run_id=outcome.run_id,
                route=outcome.route,
                decision_tier=outcome.decision.tier,
                execution_state=(
                    outcome.execution.state if outcome.execution is not None else None
                ),
                occurred_at=occurred_at,
            )
            self._insert_agent_outcome(outcome, occurred_at=occurred_at)

    def _complete_agent_run(
        self,
        *,
        run_id: str,
        route: OutcomeRoute,
        decision_tier: AutonomyTier,
        execution_state: ExecutionState | None,
        occurred_at: datetime,
    ) -> None:
        cleaned_run_id = run_id.strip()
        cursor = self._connection.execute(
            """
            UPDATE agent_runs
            SET status = 'completed', outcome_route = ?, decision_tier = ?,
                execution_state = ?, updated_at = ?
            WHERE run_id = ? AND status = 'processing'
            """,
            (
                route.value,
                decision_tier.value,
                execution_state.value if execution_state is not None else None,
                occurred_at.isoformat(),
                cleaned_run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise AgentRunStateConflictError(
                f"agent run is missing or already complete: {cleaned_run_id}"
            )
        self._insert_event(
            stream_id=f"run:{cleaned_run_id}",
            event_type="run.completed",
            payload={
                "route": route.value,
                "decision_tier": decision_tier.value,
                "execution_state": (execution_state.value if execution_state is not None else None),
            },
            event_version=1,
            event_id=None,
            occurred_at=occurred_at,
        )

    def _insert_agent_outcome(
        self,
        outcome: AgentOutcome,
        *,
        occurred_at: datetime,
    ) -> None:
        outcome_json = _canonical_json(
            outcome.model_dump(mode="json", exclude_computed_fields=True)
        )
        approval_id = outcome.approval.approval_id if outcome.approval is not None else None
        try:
            self._connection.execute(
                """
                INSERT INTO agent_outcomes (
                    run_id, email_id, provider_message_id, decision_id, approval_id,
                    outcome_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.run_id,
                    outcome.email.email_id,
                    outcome.email.provider_message_id,
                    outcome.decision.decision_id,
                    approval_id,
                    outcome_json,
                    occurred_at.isoformat(),
                    occurred_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise AgentRunStateConflictError(
                "run outcome identity already exists or is not attached to its run"
            ) from exc

    def replace_agent_outcome(
        self,
        outcome: AgentOutcome,
        *,
        expected_decision_id: str,
        occurred_at: datetime,
    ) -> None:
        """Replace the current read model after an approved workflow edit."""
        if outcome.approval is None or outcome.proposal is None:
            raise ValueError("replacement outcome must contain its proposal and new approval")
        outcome_json = _canonical_json(
            outcome.model_dump(mode="json", exclude_computed_fields=True)
        )
        with self._write_transaction():
            cursor = self._connection.execute(
                """
                UPDATE agent_outcomes
                SET email_id = ?, provider_message_id = ?, decision_id = ?,
                    approval_id = ?, outcome_json = ?, updated_at = ?
                WHERE run_id = ? AND decision_id = ?
                """,
                (
                    outcome.email.email_id,
                    outcome.email.provider_message_id,
                    outcome.decision.decision_id,
                    outcome.approval.approval_id,
                    outcome_json,
                    occurred_at.isoformat(),
                    outcome.run_id,
                    expected_decision_id.strip(),
                ),
            )
            if cursor.rowcount != 1:
                raise AgentRunStateConflictError("current run outcome changed before replacement")
            self._insert_event(
                stream_id=f"run:{outcome.run_id}",
                event_type="proposal.edited",
                payload={
                    "decision_id": outcome.decision.decision_id,
                    "proposal_id": outcome.proposal.proposal_id,
                    "proposal_version": outcome.proposal.version,
                    "approval_id": outcome.approval.approval_id,
                },
                event_version=1,
                event_id=None,
                occurred_at=occurred_at,
            )

    def get_agent_outcome(self, run_id: str) -> AgentOutcome | None:
        """Load one completed run snapshot by run identity."""
        return self._get_agent_outcome("run_id", run_id)

    def get_agent_outcome_by_decision(self, decision_id: str) -> AgentOutcome | None:
        """Load the current run snapshot carrying a decision."""
        return self._get_agent_outcome("decision_id", decision_id)

    def get_agent_outcome_by_approval(self, approval_id: str) -> AgentOutcome | None:
        """Load the current run snapshot carrying an approval request."""
        return self._get_agent_outcome("approval_id", approval_id)

    def _get_agent_outcome(self, column: str, identity: str) -> AgentOutcome | None:
        if column not in {"run_id", "decision_id", "approval_id"}:
            raise ValueError("unsupported outcome lookup")
        row = self._connection.execute(
            f"SELECT outcome_json FROM agent_outcomes WHERE {column} = ?",
            (identity.strip(),),
        ).fetchone()
        return None if row is None else _agent_outcome_from_row(row)

    def list_agent_outcomes(self, *, limit: int = 100) -> tuple[AgentOutcome, ...]:
        """List recent completed run snapshots for the CLI inbox."""
        if not 1 <= limit <= 1_000:
            raise ValueError("outcome limit must be between 1 and 1000")
        rows = self._connection.execute(
            """
            SELECT outcome_json
            FROM agent_outcomes
            ORDER BY updated_at DESC, run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(_agent_outcome_from_row(row) for row in rows)

    def commit_feedback(
        self,
        submission: FeedbackSubmission,
        *,
        update: Callable[[PreferenceState], PreferenceState],
    ) -> tuple[FeedbackRecord, bool]:
        """Deduplicate and atomically commit feedback, preference state, and events."""
        with self._write_transaction():
            rows = self._connection.execute(
                """
                SELECT record_json
                FROM feedback_records
                WHERE feedback_id = ? OR dedupe_key = ?
                """,
                (submission.feedback_id, submission.dedupe_key),
            ).fetchall()
            if len(rows) > 1:
                raise FeedbackConflictError("feedback ID and dedupe key identify different records")
            if rows:
                existing = _feedback_from_row(rows[0])
                if not _same_feedback_semantics(existing, submission):
                    raise FeedbackConflictError(
                        "feedback identity was reused for different evidence"
                    )
                return existing, False

            previous = self.get_preference(submission.context_key)
            updated = update(previous)
            record = FeedbackRecord.model_validate(
                {
                    **submission.model_dump(mode="python"),
                    "previous_state": previous,
                    "updated_state": updated,
                }
            )
            record_json = _canonical_json(record.model_dump(mode="json"))
            try:
                self._connection.execute(
                    """
                    INSERT INTO feedback_records (
                        feedback_id, dedupe_key, decision_id, proposal_id,
                        proposal_version, context_key, feedback_type, actor,
                        source_reference, record_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.feedback_id,
                        record.dedupe_key,
                        record.decision_id,
                        record.proposal_id,
                        record.proposal_version,
                        record.context_key,
                        record.feedback_type.value,
                        record.actor,
                        record.source_reference,
                        record_json,
                        record.created_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise FeedbackConflictError("feedback identity already exists") from exc

            self._upsert_preference(updated)
            self._insert_event(
                stream_id=f"feedback:{record.feedback_id}",
                event_type="feedback.recorded",
                payload={
                    "feedback_id": record.feedback_id,
                    "decision_id": record.decision_id,
                    "proposal_id": record.proposal_id,
                    "proposal_version": record.proposal_version,
                    "context_key": record.context_key,
                    "feedback_type": record.feedback_type.value,
                    "actor": record.actor,
                    "source_reference": record.source_reference,
                },
                event_version=1,
                event_id=None,
                occurred_at=record.created_at,
            )
            self._insert_event(
                stream_id=f"preference:{record.context_key}",
                event_type="preference.updated",
                payload={
                    "feedback_id": record.feedback_id,
                    "feedback_type": record.feedback_type.value,
                    "previous_alpha": previous.alpha,
                    "previous_beta": previous.beta,
                    "updated_alpha": updated.alpha,
                    "updated_beta": updated.beta,
                    "observations": updated.observations,
                    "cooldown_remaining": updated.cooldown_remaining,
                },
                event_version=1,
                event_id=None,
                occurred_at=record.created_at,
            )
        return record, True

    def get_feedback(self, feedback_id: str) -> FeedbackRecord | None:
        """Load one durable feedback receipt."""
        row = self._connection.execute(
            "SELECT record_json FROM feedback_records WHERE feedback_id = ?",
            (feedback_id.strip(),),
        ).fetchone()
        return None if row is None else _feedback_from_row(row)

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


def _execution_from_row(row: sqlite3.Row) -> ExecutionRecord:
    try:
        command = ExecutionCommand.model_validate_json(str(row["command_json"]))
        if command.idempotency_key != str(row["idempotency_key"]):
            raise StorageError("stored execution key differs from its command")
        return ExecutionRecord.model_validate(
            {
                "execution_id": str(row["execution_id"]),
                "command": command,
                "state": str(row["state"]),
                "attempt_count": int(row["attempt_count"]),
                "detail": str(row["detail"]),
                "provider_operation_id": row["provider_operation_id"],
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "started_at": str(row["started_at"]),
                "completed_at": row["completed_at"],
            }
        )
    except ValidationError as exc:
        raise StorageError("stored execution is malformed") from exc


def _execution_stream_id(execution_id: str) -> str:
    return f"execution:{execution_id}"


def _feedback_from_row(row: sqlite3.Row) -> FeedbackRecord:
    try:
        return FeedbackRecord.model_validate_json(str(row["record_json"]))
    except ValidationError as exc:
        raise StorageError("stored feedback is malformed") from exc


def _agent_outcome_from_row(row: sqlite3.Row) -> AgentOutcome:
    try:
        return AgentOutcome.model_validate_json(str(row["outcome_json"]))
    except ValidationError as exc:
        raise StorageError("stored agent outcome is malformed") from exc


def _same_feedback_semantics(
    existing: FeedbackRecord,
    submission: FeedbackSubmission,
) -> bool:
    return (
        existing.dedupe_key == submission.dedupe_key
        and existing.decision_id == submission.decision_id
        and existing.proposal_id == submission.proposal_id
        and existing.proposal_version == submission.proposal_version
        and existing.context_key == submission.context_key
        and existing.feedback_type == submission.feedback_type
    )
