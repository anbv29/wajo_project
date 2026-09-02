"""Fast behavioral checks for versioned SQLite state and immutable audit events."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from wajo_agent.domain import FeedbackType, PreferenceState
from wajo_agent.storage import (
    SCHEMA_VERSION,
    DuplicateEventError,
    SchemaVersionError,
    SQLiteStore,
    StorageError,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _assert_append_only(db_path: Path, statement: str) -> None:
    connection = sqlite3.connect(db_path)
    try:
        try:
            connection.execute(statement)
        except sqlite3.IntegrityError as exc:
            _require("append-only" in str(exc), "database trigger gave the wrong error")
        else:
            raise RuntimeError("database allowed an audit event to be changed")
    finally:
        connection.close()


@contextmanager
def _workspace_database_files() -> Generator[Path, None, None]:
    """Use ordinary workspace files; Windows sandboxing blocks tempfile ACLs."""
    workspace = Path.cwd()
    database_names = ("storage_check.sqlite3", "storage_check_future.sqlite3")
    artifacts = tuple(
        workspace / f"{name}{suffix}" for name in database_names for suffix in ("", "-wal", "-shm")
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("storage-check artifact already exists; refusing to overwrite it")
    try:
        yield workspace
    finally:
        for path in artifacts:
            path.unlink(missing_ok=True)


def main() -> None:
    checks = 0
    with _workspace_database_files() as temp_dir:
        db_path = temp_dir / "storage_check.sqlite3"

        with SQLiteStore(db_path) as store:
            _require(store.schema_version == SCHEMA_VERSION == 4, "wrong schema version")
            checks += 1

            cold = store.get_preference("ctx_newsletter")
            _require(
                (cold.alpha, cold.beta, cold.observations) == (1, 1, 0),
                "cold preference did not use the safe Beta(1, 1) prior",
            )
            checks += 1

            learned = PreferenceState(
                context_key="ctx_newsletter",
                alpha=7,
                beta=2,
                observations=7,
                recent_feedback=(FeedbackType.APPROVED, FeedbackType.EDITED),
                cooldown_remaining=3,
            )
            store.save_preference(learned)
            _require(store.get_preference("ctx_newsletter") == learned, "upsert lost state")
            checks += 1

            first = store.append_event(
                stream_id="email_123",
                event_type="email.received",
                event_version=2,
                payload={"subject": "Café receipt", "nested": {"safe": True}},
                event_id="event_first",
            )
            second = store.append_event(
                stream_id="email_123",
                event_type="proposal.created",
                payload={"action": "archive", "confidence": 0.91},
                event_id="event_second",
            )
            other = store.append_event(
                stream_id="email_456",
                event_type="email.received",
                payload={"subject": "Another message"},
            )
            _require((first.sequence, second.sequence, other.sequence) == (1, 2, 1), "bad sequence")
            _require(first.event_version == 2, "event version was not retained")
            _require(store.read_stream("email_123") == (first, second), "stream order changed")
            _require(store.read_stream("email_123", after_sequence=1) == (second,), "bad cursor")
            checks += 4

            try:
                store.append_event(
                    stream_id="email_789",
                    event_type="email.received",
                    payload={},
                    event_id="event_first",
                )
            except DuplicateEventError:
                pass
            else:
                raise RuntimeError("duplicate event identifier was accepted")
            checks += 1

            atomic_state = PreferenceState(
                context_key="ctx_atomic",
                alpha=2,
                observations=1,
                recent_feedback=(FeedbackType.APPROVED,),
            )
            atomic_event = store.save_preference_and_append_event(
                atomic_state,
                stream_id="preference_ctx_atomic",
                event_type="preference.updated",
                payload={"feedback": "approved"},
            )
            _require(store.get_preference("ctx_atomic") == atomic_state, "atomic state missing")
            _require(
                store.read_stream("preference_ctx_atomic") == (atomic_event,),
                "atomic event missing",
            )
            checks += 2

            rollback_state = PreferenceState(
                context_key="ctx_must_rollback",
                alpha=2,
                observations=1,
                recent_feedback=(FeedbackType.APPROVED,),
            )
            try:
                store.save_preference_and_append_event(
                    rollback_state,
                    stream_id="preference_ctx_must_rollback",
                    event_type="preference.updated",
                    payload={"feedback": "approved"},
                    event_id="event_first",
                )
            except DuplicateEventError:
                pass
            else:
                raise RuntimeError("atomic operation accepted a duplicate event")
            _require(
                store.get_preference("ctx_must_rollback").observations == 0,
                "failed event left a partial preference update",
            )
            checks += 1

            invalid_payload = cast(dict[str, JsonValue], {"score": float("nan")})
            try:
                store.append_event(
                    stream_id="email_nan",
                    event_type="invalid.number",
                    payload=invalid_payload,
                )
            except StorageError:
                pass
            else:
                raise RuntimeError("non-standard NaN JSON was stored")
            _require(store.read_stream("email_nan") == (), "failed event partially committed")
            checks += 2

        with SQLiteStore(db_path) as reopened:
            _require(reopened.get_preference("ctx_newsletter") == learned, "state was not durable")
            _require(
                reopened.read_stream("email_123") == (first, second),
                "events were not durable",
            )
            checks += 2

        _assert_append_only(
            db_path,
            "UPDATE audit_events SET event_type = 'tampered' WHERE event_id = 'event_first'",
        )
        _assert_append_only(db_path, "DELETE FROM audit_events WHERE event_id = 'event_first'")
        checks += 2

        future_path = temp_dir / "storage_check_future.sqlite3"
        connection = sqlite3.connect(future_path)
        try:
            connection.execute("PRAGMA user_version = 99")
        finally:
            connection.close()
        try:
            SQLiteStore(future_path)
        except SchemaVersionError:
            pass
        else:
            raise RuntimeError("newer database schema was opened unsafely")
        checks += 1

    print(f"Storage checks passed: {checks}")


if __name__ == "__main__":
    main()
