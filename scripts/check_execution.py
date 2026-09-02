"""Behavioral checks for safe, durable, at-most-once mailbox execution."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from wajo_agent.approvals import ApprovalService
from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    ApprovalStatus,
    AutonomyTier,
    ExecutionCommand,
    ExecutionResult,
    ExecutionState,
    InjectionSignal,
    Intent,
    MessagePayload,
    ReplyPayload,
    RiskAssessment,
)
from wajo_agent.execution import (
    ExecutionAuthorizationError,
    ExecutionInProgressError,
    ExecutionService,
    MockMailboxExecutor,
    MockOutcome,
    effect_idempotency_key,
)
from wajo_agent.policy import PolicyEngine
from wajo_agent.storage import SQLiteStore


class FixedClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 2, 20, 9, 30, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current


class WrongResultExecutor:
    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        return ExecutionResult(
            execution_id=command.execution_id,
            command_id="wrong-command",
            idempotency_key=command.idempotency_key,
            proposal_id=command.proposal_id,
            state=ExecutionState.SUCCEEDED,
            detail="Malformed mock result",
        )


class CrashAfterClaimExecutor:
    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        del command
        raise KeyboardInterrupt("simulated process crash")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _archive(suffix: str, *, version: int = 1) -> ActionProposal:
    return ActionProposal(
        proposal_id=f"proposal_archive_{suffix}",
        version=version,
        email_id=f"email_archive_{suffix}",
        action_type=ActionType.ARCHIVE,
        intent=Intent.NEWSLETTER,
        summary="Archive this newsletter",
        payload=MessagePayload(
            kind=ActionType.ARCHIVE,
            message_id=f"provider_archive_{suffix}",
        ),
    )


def _reply(suffix: str) -> ActionProposal:
    return ActionProposal(
        proposal_id=f"proposal_reply_{suffix}",
        email_id=f"email_reply_{suffix}",
        action_type=ActionType.SEND_REPLY,
        intent=Intent.REQUEST,
        summary="Reply to the sender",
        payload=ReplyPayload(
            kind=ActionType.SEND_REPLY,
            recipients=("sender@example.com",),
            body="Yes, that works for me.",
        ),
    )


def _decision(proposal: ActionProposal):
    return PolicyEngine().decide(
        proposal,
        risk=RiskAssessment(),
        preference_tier=AutonomyTier.SILENT,
    )


@contextmanager
def _workspace_database_file() -> Generator[Path, None, None]:
    database = Path.cwd() / "execution_check.sqlite3"
    artifacts = tuple(Path(f"{database}{suffix}") for suffix in ("", "-wal", "-shm"))
    if any(path.exists() for path in artifacts):
        raise RuntimeError("execution-check artifact already exists; refusing to overwrite it")
    try:
        yield database
    finally:
        for path in artifacts:
            path.unlink(missing_ok=True)


def main() -> None:
    checks = 0
    clock = FixedClock()
    archive = _archive("success")
    archive_decision = _decision(archive)
    archive_key = effect_idempotency_key(
        "mailbox_1",
        "provider_archive_success",
        archive,
    )
    _require(len(archive_key) == 64, "effect key is not SHA-256")
    _require(
        archive_key == effect_idempotency_key(" mailbox_1 ", " provider_archive_success ", archive),
        "identity normalization changed the effect key",
    )
    _require(
        archive_key != effect_idempotency_key("mailbox_2", "provider_archive_success", archive),
        "effect key leaked between mailboxes",
    )
    changed_summary = archive.model_copy(update={"summary": "Different display text"})
    _require(
        archive_key
        == effect_idempotency_key(
            "mailbox_1",
            "provider_archive_success",
            changed_summary,
        ),
        "display-only text changed effect identity",
    )
    checks += 4

    with _workspace_database_file() as db_path:
        with SQLiteStore(db_path) as store:
            success_executor = MockMailboxExecutor()
            success_service = ExecutionService(
                store,
                success_executor,
                clock=clock,
            )
            first = success_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_success",
                proposal=archive,
                decision=archive_decision,
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(first.state == ExecutionState.SUCCEEDED, "safe effect did not succeed")
            _require(success_executor.call_count == 1, "adapter was not called exactly once")
            checks += 2

            duplicate = success_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_success",
                proposal=archive,
                decision=archive_decision,
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(duplicate.execution_id == first.execution_id, "retry returned another effect")
            _require(success_executor.call_count == 1, "successful effect executed twice")
            checks += 2

            reply = _reply("approved")
            reply_decision = _decision(reply)
            approvals = ApprovalService(store, clock=clock)
            approval = approvals.request(reply, reply_decision)
            approvals.grant(approval.approval_id, reply, actor="user@example.com")
            reply_executor = MockMailboxExecutor()
            reply_service = ExecutionService(
                store,
                reply_executor,
                approval_service=approvals,
                clock=clock,
            )
            sent = reply_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_reply_approved",
                proposal=reply,
                decision=reply_decision,
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
                approval_id=approval.approval_id,
            )
            _require(sent.state == ExecutionState.SUCCEEDED, "approved reply did not execute")
            _require(
                store.get_approval(approval.approval_id).status == ApprovalStatus.CONSUMED,
                "execution claim did not consume approval",
            )
            approval_events = store.read_stream(f"approval:{approval.approval_id}")
            _require(
                tuple(event.event_type for event in approval_events)
                == ("approval.requested", "approval.granted", "approval.consumed"),
                "approval consumption was not audited",
            )
            checks += 3

            sent_again = reply_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_reply_approved",
                proposal=reply,
                decision=reply_decision,
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
                approval_id=approval.approval_id,
            )
            _require(sent_again.execution_id == sent.execution_id, "reply retry changed result")
            _require(reply_executor.call_count == 1, "approved reply was sent twice")
            checks += 2

            no_approval_reply = _reply("missing_approval")
            try:
                reply_service.execute(
                    mailbox_id="mailbox_1",
                    provider_message_id="provider_reply_missing",
                    proposal=no_approval_reply,
                    decision=_decision(no_approval_reply),
                    risk=RiskAssessment(),
                    preference_tier=AutonomyTier.SILENT,
                )
            except ExecutionAuthorizationError:
                pass
            else:
                raise RuntimeError("ASK action executed without approval")
            checks += 1

            changed_risk_reply = _reply("risk_changed")
            changed_risk_decision = _decision(changed_risk_reply)
            risk_approval = approvals.request(changed_risk_reply, changed_risk_decision)
            approvals.grant(
                risk_approval.approval_id,
                changed_risk_reply,
                actor="user@example.com",
            )
            injection_risk = RiskAssessment(
                injection_signals=frozenset({InjectionSignal.INSTRUCTION_OVERRIDE})
            )
            before_risk_calls = reply_executor.call_count
            try:
                reply_service.execute(
                    mailbox_id="mailbox_1",
                    provider_message_id="provider_reply_risk_changed",
                    proposal=changed_risk_reply,
                    decision=changed_risk_decision,
                    risk=injection_risk,
                    preference_tier=AutonomyTier.SILENT,
                    approval_id=risk_approval.approval_id,
                )
            except ExecutionAuthorizationError:
                pass
            else:
                raise RuntimeError("stale decision bypassed the fresh policy check")
            _require(reply_executor.call_count == before_risk_calls, "unsafe adapter call occurred")
            _require(
                store.get_approval(risk_approval.approval_id).status == ApprovalStatus.GRANTED,
                "policy rejection consumed approval",
            )
            checks += 3

            safe_failure = _archive("failed_safe")
            safe_failure_executor = MockMailboxExecutor((MockOutcome.RAISE_UNAVAILABLE,))
            safe_failure_service = ExecutionService(
                store,
                safe_failure_executor,
                clock=clock,
            )
            failed = safe_failure_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_failed_safe",
                proposal=safe_failure,
                decision=_decision(safe_failure),
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(failed.state == ExecutionState.FAILED_SAFE, "safe failure became unknown")
            safe_failure_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_failed_safe",
                proposal=safe_failure,
                decision=_decision(safe_failure),
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(safe_failure_executor.call_count == 1, "safe failure retried automatically")
            checks += 2

            unknown = _archive("unknown")
            unknown_executor = MockMailboxExecutor((MockOutcome.RAISE_UNKNOWN,))
            unknown_service = ExecutionService(store, unknown_executor, clock=clock)
            unknown_result = unknown_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_unknown",
                proposal=unknown,
                decision=_decision(unknown),
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(unknown_result.state == ExecutionState.UNKNOWN, "unknown was not preserved")
            unknown_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_unknown",
                proposal=unknown,
                decision=_decision(unknown),
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(unknown_executor.call_count == 1, "unknown effect was retried")
            checks += 2

            malformed = _archive("malformed_result")
            malformed_service = ExecutionService(store, WrongResultExecutor(), clock=clock)
            malformed_result = malformed_service.execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_malformed",
                proposal=malformed,
                decision=_decision(malformed),
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(
                malformed_result.state == ExecutionState.UNKNOWN,
                "malformed adapter result was trusted",
            )
            checks += 1

            crash = _archive("crash")
            crash_service = ExecutionService(store, CrashAfterClaimExecutor(), clock=clock)
            try:
                crash_service.execute(
                    mailbox_id="mailbox_1",
                    provider_message_id="provider_archive_crash",
                    proposal=crash,
                    decision=_decision(crash),
                    risk=RiskAssessment(),
                    preference_tier=AutonomyTier.SILENT,
                )
            except KeyboardInterrupt:
                pass
            else:
                raise RuntimeError("simulated process crash did not escape")
            crash_key = effect_idempotency_key(
                "mailbox_1",
                "provider_archive_crash",
                crash,
            )
            crash_record = store.get_execution_by_idempotency_key(crash_key)
            _require(
                crash_record is not None and crash_record.state == ExecutionState.EXECUTING,
                "pre-effect execution claim was not durable",
            )
            recovery_executor = MockMailboxExecutor()
            try:
                ExecutionService(store, recovery_executor, clock=clock).execute(
                    mailbox_id="mailbox_1",
                    provider_message_id="provider_archive_crash",
                    proposal=crash,
                    decision=_decision(crash),
                    risk=RiskAssessment(),
                    preference_tier=AutonomyTier.SILENT,
                )
            except ExecutionInProgressError:
                pass
            else:
                raise RuntimeError("crash-gap effect was repeated")
            _require(recovery_executor.call_count == 0, "recovery repeated unresolved effect")
            checks += 3

            first_execution_id = first.execution_id

        with SQLiteStore(db_path) as reopened:
            restart_executor = MockMailboxExecutor()
            restart_result = ExecutionService(reopened, restart_executor, clock=clock).execute(
                mailbox_id="mailbox_1",
                provider_message_id="provider_archive_success",
                proposal=archive,
                decision=archive_decision,
                risk=RiskAssessment(),
                preference_tier=AutonomyTier.SILENT,
            )
            _require(
                restart_result.execution_id == first_execution_id,
                "restart forgot the completed execution",
            )
            _require(restart_executor.call_count == 0, "restart repeated completed effect")
            checks += 2

    print(f"Execution checks passed: {checks}")


if __name__ == "__main__":
    main()
