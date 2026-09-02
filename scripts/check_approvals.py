"""Behavioral checks for exact, expiring, single-use approval authorization."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wajo_agent.approvals import (
    ApprovalBindingError,
    ApprovalDecisionError,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalService,
    ApprovalStateError,
    approval_payload_hash,
    canonical_approval_payload,
)
from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    ApprovalStatus,
    AutonomyTier,
    Intent,
    ReplyPayload,
    RiskAssessment,
)
from wajo_agent.policy import PolicyEngine
from wajo_agent.storage import ApprovalStateConflictError, SQLiteStore
from wajo_agent.storage.sqlite import MIGRATION_V1


class MutableClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _reply(*, body: str = "Yes, that works for me.", version: int = 1) -> ActionProposal:
    return ActionProposal(
        proposal_id="proposal_reply_1",
        version=version,
        email_id="email_1",
        action_type=ActionType.SEND_REPLY,
        intent=Intent.REQUEST,
        summary="Reply to the sender",
        payload=ReplyPayload(
            kind=ActionType.SEND_REPLY,
            recipients=("sender@example.com",),
            body=body,
        ),
    )


def _ask_decision(proposal: ActionProposal):
    return PolicyEngine().decide(
        proposal,
        risk=RiskAssessment(),
        preference_tier=AutonomyTier.SILENT,
    )


@contextmanager
def _workspace_database_files() -> Generator[tuple[Path, Path], None, None]:
    workspace = Path.cwd()
    names = ("approval_check.sqlite3", "approval_check_legacy.sqlite3")
    artifacts = tuple(
        workspace / f"{name}{suffix}" for name in names for suffix in ("", "-wal", "-shm")
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("approval-check artifact already exists; refusing to overwrite it")
    try:
        yield workspace / names[0], workspace / names[1]
    finally:
        for path in artifacts:
            path.unlink(missing_ok=True)


def main() -> None:
    checks = 0
    clock = MutableClock()
    original = _reply()
    decision = _ask_decision(original)

    canonical = canonical_approval_payload(original)
    _require(canonical == canonical_approval_payload(original), "canonical bytes changed")
    _require(len(approval_payload_hash(original)) == 64, "approval hash is not SHA-256")
    _require(
        approval_payload_hash(_reply(body="Changed body")) != approval_payload_hash(original),
        "execution-relevant edit did not change the hash",
    )
    _require(
        approval_payload_hash(_reply(version=2)) != approval_payload_hash(original),
        "proposal version was not bound into the hash",
    )
    checks += 4

    with _workspace_database_files() as (db_path, legacy_path):
        with SQLiteStore(db_path) as store:
            service = ApprovalService(store, clock=clock)

            request = service.request(original, decision)
            _require(request.status == ApprovalStatus.PENDING, "request did not start pending")
            _require(request.expires_at == clock.current + timedelta(minutes=15), "wrong expiry")
            _require(store.get_approval(request.approval_id) == request, "request was not stored")
            checks += 3

            try:
                service.grant(
                    request.approval_id,
                    _reply(body="Tampered after display"),
                    actor="user@example.com",
                )
            except ApprovalBindingError:
                pass
            else:
                raise RuntimeError("changed payload reused an approval")
            _require(
                service.get(request.approval_id).status == ApprovalStatus.PENDING,
                "binding failure changed state",
            )
            checks += 2

            granted = service.grant(
                request.approval_id,
                original,
                actor=" user@example.com ",
            )
            _require(granted.status == ApprovalStatus.GRANTED, "grant did not persist")
            _require(granted.actor == "user@example.com", "actor was not normalized")
            checks += 2

            consumed = service.consume(request.approval_id, original)
            _require(consumed.status == ApprovalStatus.CONSUMED, "approval was not consumed")
            _require(consumed.consumed_at == clock.current, "consumption time is wrong")
            try:
                service.consume(request.approval_id, original)
            except ApprovalStateError:
                pass
            else:
                raise RuntimeError("approval was consumed twice")
            event_types = tuple(
                event.event_type for event in store.read_stream(f"approval:{request.approval_id}")
            )
            _require(
                event_types == ("approval.requested", "approval.granted", "approval.consumed"),
                "approval audit trail is incomplete",
            )
            checks += 3

            expiring = service.request(original, decision, ttl=timedelta(minutes=1))
            clock.advance(timedelta(minutes=2))
            expired = service.get(expiring.approval_id)
            _require(expired.status == ApprovalStatus.EXPIRED, "deadline did not expire request")
            try:
                service.grant(expiring.approval_id, original, actor="user@example.com")
            except ApprovalExpiredError:
                pass
            else:
                raise RuntimeError("expired approval was granted")
            _require(
                tuple(
                    event.event_type
                    for event in store.read_stream(f"approval:{expiring.approval_id}")
                )
                == ("approval.requested", "approval.expired"),
                "expiry was not audited",
            )
            checks += 3

            rejected_request = service.request(original, decision)
            rejected = service.reject(rejected_request.approval_id, actor="user@example.com")
            _require(rejected.status == ApprovalStatus.REJECTED, "rejection did not persist")
            try:
                service.grant(rejected.approval_id, original, actor="user@example.com")
            except ApprovalStateError:
                pass
            else:
                raise RuntimeError("rejected request was later granted")
            checks += 2

            revoked_request = service.request(original, decision)
            service.grant(revoked_request.approval_id, original, actor="approver@example.com")
            revoked = service.invalidate(
                revoked_request.approval_id,
                original,
                actor="revoker@example.com",
            )
            _require(revoked.status == ApprovalStatus.INVALIDATED, "revocation did not persist")
            try:
                service.consume(revoked.approval_id, original)
            except ApprovalStateError:
                pass
            else:
                raise RuntimeError("invalidated approval was consumed")
            checks += 2

            edit_request = service.request(original, decision)
            revised = _reply(body="User-edited response", version=2)
            revised_decision = _ask_decision(revised)
            old, replacement = service.replace_for_edit(
                edit_request.approval_id,
                original,
                revised,
                revised_decision,
                actor="user@example.com",
            )
            _require(old.status == ApprovalStatus.INVALIDATED, "edit left old request active")
            _require(
                old.superseded_by_approval_id == replacement.approval_id,
                "old request does not identify replacement",
            )
            _require(replacement.proposal_version == 2, "replacement version is wrong")
            try:
                service.grant(replacement.approval_id, original, actor="user@example.com")
            except ApprovalBindingError:
                pass
            else:
                raise RuntimeError("replacement approved the pre-edit payload")
            service.grant(replacement.approval_id, revised, actor="user@example.com")
            checks += 4

            stale_request = service.request(original, decision)
            service.grant(stale_request.approval_id, original, actor="user@example.com")
            with SQLiteStore(db_path) as second_store:
                stale_record = second_store.get_approval(stale_request.approval_id)
                service.consume(stale_request.approval_id, original)
                stale_consumed = stale_record.model_copy(
                    update={
                        "status": ApprovalStatus.CONSUMED,
                        "consumed_at": clock.current,
                        "updated_at": clock.current,
                    }
                )
                try:
                    second_store.transition_approval(
                        stale_consumed,
                        expected_statuses=(ApprovalStatus.GRANTED,),
                        event_type="approval.consumed",
                        event_payload={"approval_id": stale_request.approval_id},
                    )
                except ApprovalStateConflictError:
                    pass
                else:
                    raise RuntimeError("stale worker consumed an approval twice")
            checks += 1

            try:
                service.request(original, decision, ttl=timedelta(0))
            except ApprovalError:
                pass
            else:
                raise RuntimeError("non-positive approval TTL was accepted")

            wrong_decision = decision.model_copy(update={"tier": AutonomyTier.NOTIFY})
            try:
                service.request(original, wrong_decision)
            except ApprovalDecisionError:
                pass
            else:
                raise RuntimeError("non-ASK decision created an approval")

            inconsistent_decision = decision.model_copy(
                update={"capability_floor": AutonomyTier.SILENT}
            )
            try:
                service.request(original, inconsistent_decision)
            except ApprovalDecisionError:
                pass
            else:
                raise RuntimeError("inconsistent safety floors created an approval")

            try:
                service.grant(
                    service.request(original, decision).approval_id,
                    original,
                    actor="   ",
                )
            except ApprovalError:
                pass
            else:
                raise RuntimeError("blank approval actor was accepted")
            checks += 4

        with SQLiteStore(db_path) as reopened:
            _require(
                reopened.get_approval(request.approval_id).status == ApprovalStatus.CONSUMED,
                "approval state was not durable across restart",
            )
            checks += 1

        legacy_connection = sqlite3.connect(legacy_path)
        try:
            legacy_connection.executescript(MIGRATION_V1)
        finally:
            legacy_connection.close()
        with SQLiteStore(legacy_path) as migrated:
            _require(migrated.schema_version == 2, "version-1 database did not migrate")
            migrated_request = ApprovalService(migrated, clock=clock).request(original, decision)
            _require(
                migrated.get_approval(migrated_request.approval_id) == migrated_request,
                "migrated database cannot store approvals",
            )
            checks += 2

    print(f"Approval checks passed: {checks}")


if __name__ == "__main__":
    main()
