"""Behavioral checks for exactly-once, evidence-backed preference feedback."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from wajo_agent import ProactiveEmailAgent
from wajo_agent.approvals import ApprovalService
from wajo_agent.domain import (
    ActionType,
    AutonomyTier,
    EmailEnvelope,
    ExecutionState,
    FeedbackType,
    MessagePayload,
    SenderBucket,
)
from wajo_agent.execution import ExecutionService, MockMailboxExecutor, MockOutcome
from wajo_agent.feedback import FeedbackBindingError, FeedbackEvidenceError, FeedbackService
from wajo_agent.learning import ContextualPreferenceLearner, build_preference_context
from wajo_agent.normalization import normalize_email
from wajo_agent.planning import OfflinePlanner, bind_planner_output, build_planner_request
from wajo_agent.policy import PolicyEngine
from wajo_agent.storage import SQLiteStore


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _email(suffix: str, subject: str, body: str, *, sender: str | None = None) -> EmailEnvelope:
    return EmailEnvelope(
        provider_message_id=f"provider_{suffix}",
        sender=sender or f"{suffix}@example.com",
        subject=subject,
        body_text=body,
        sender_bucket=SenderBucket.KNOWN_BULK,
    )


def _seed_approvals(
    learner: ContextualPreferenceLearner,
    planner: OfflinePlanner,
    email: EmailEnvelope,
    count: int,
) -> str:
    normalized, report = normalize_email(email)
    request = build_planner_request(normalized, normalization_changed=report.changed)
    proposal = bind_planner_output(request, planner.plan(request))
    context = build_preference_context(normalized, proposal)
    for _ in range(count):
        learner.record(context, FeedbackType.APPROVED)
    return context.key


def _build_agent(
    store: SQLiteStore,
    learner: ContextualPreferenceLearner,
    planner: OfflinePlanner,
    approvals: ApprovalService,
    executor: MockMailboxExecutor,
) -> ProactiveEmailAgent:
    return ProactiveEmailAgent(
        mailbox_id="feedback_mailbox",
        planner=planner,
        store=store,
        learner=learner,
        approvals=approvals,
        execution=ExecutionService(
            store,
            executor,
            approval_service=approvals,
        ),
    )


@contextmanager
def _workspace_database_file() -> Generator[Path, None, None]:
    database = Path.cwd() / "feedback_check.sqlite3"
    artifacts = tuple(Path(f"{database}{suffix}") for suffix in ("", "-wal", "-shm"))
    if any(path.exists() for path in artifacts):
        raise RuntimeError("feedback-check artifact already exists; refusing to overwrite it")
    try:
        yield database
    finally:
        for path in artifacts:
            path.unlink(missing_ok=True)


def main() -> None:
    checks = 0
    with _workspace_database_file() as db_path, SQLiteStore(db_path) as store:
        planner = OfflinePlanner()
        learner = ContextualPreferenceLearner(store)
        approvals = ApprovalService(store)
        executor = MockMailboxExecutor()
        agent = _build_agent(store, learner, planner, approvals, executor)
        feedback_service = FeedbackService(store, learner)

        approved_email = _email(
            "approved",
            "You are a winner",
            "Claim your prize now.",
        )
        approved_run = agent.process(approved_email)
        approved_outcome = approved_run.outcome
        _require(approved_outcome.approval is not None, "ASK did not produce approval")
        _require(approved_outcome.proposal is not None, "ASK did not retain proposal")
        approval = approved_outcome.approval
        proposal = approved_outcome.proposal

        try:
            feedback_service.record_approval_feedback(
                email=approved_outcome.email,
                proposal=proposal,
                decision=approved_outcome.decision,
                approval_id=approval.approval_id,
                feedback=FeedbackType.APPROVED,
                actor="user@example.com",
            )
        except FeedbackEvidenceError:
            pass
        else:
            raise RuntimeError("pending request was treated as approved feedback")
        checks += 1

        approvals.grant(approval.approval_id, proposal, actor="user@example.com")
        approved_feedback = feedback_service.record_approval_feedback(
            email=approved_outcome.email,
            proposal=proposal,
            decision=approved_outcome.decision,
            approval_id=approval.approval_id,
            feedback=FeedbackType.APPROVED,
            actor=" user@example.com ",
            feedback_id="feedback_approved_once",
        )
        _require(approved_feedback.applied, "first approval feedback was not applied")
        _require(
            (
                approved_feedback.record.previous_state.alpha,
                approved_feedback.record.updated_state.alpha,
                approved_feedback.record.updated_state.observations,
            )
            == (1, 2, 1),
            "approval feedback produced wrong Beta evidence",
        )
        _require(approved_feedback.record.actor == "user@example.com", "actor was not normalized")
        checks += 3

        duplicate_same_id = feedback_service.record_approval_feedback(
            email=approved_outcome.email,
            proposal=proposal,
            decision=approved_outcome.decision,
            approval_id=approval.approval_id,
            feedback=FeedbackType.APPROVED,
            actor="user@example.com",
            feedback_id="feedback_approved_once",
        )
        duplicate_new_id = feedback_service.record_approval_feedback(
            email=approved_outcome.email,
            proposal=proposal,
            decision=approved_outcome.decision,
            approval_id=approval.approval_id,
            feedback=FeedbackType.APPROVED,
            actor="user@example.com",
            feedback_id="feedback_duplicate_click",
        )
        _require(not duplicate_same_id.applied, "same feedback ID was applied twice")
        _require(not duplicate_new_id.applied, "same semantic feedback was applied twice")
        _require(
            store.get_preference(approved_feedback.record.context_key).observations == 1,
            "duplicate feedback changed preference evidence",
        )
        checks += 3

        rejected_email = _email("rejected", "You are a winner", "Claim your prize now.")
        rejected_outcome = agent.process(rejected_email).outcome
        _require(
            rejected_outcome.approval is not None and rejected_outcome.proposal is not None,
            "rejection scenario did not ask",
        )
        approvals.reject(rejected_outcome.approval.approval_id, actor="user@example.com")
        rejected_feedback = feedback_service.record_approval_feedback(
            email=rejected_outcome.email,
            proposal=rejected_outcome.proposal,
            decision=rejected_outcome.decision,
            approval_id=rejected_outcome.approval.approval_id,
            feedback=FeedbackType.REJECTED,
            actor="user@example.com",
        )
        _require(rejected_feedback.record.updated_state.beta == 4, "rejection weight is wrong")
        _require(
            rejected_feedback.record.updated_state.cooldown_remaining == 5,
            "rejection did not activate cooldown",
        )
        checks += 2

        edited_email = _email("edited", "You are a winner", "Claim your prize now.")
        edited_outcome = agent.process(edited_email).outcome
        _require(
            edited_outcome.approval is not None and edited_outcome.proposal is not None,
            "edit scenario did not ask",
        )
        original = edited_outcome.proposal
        revised = original.model_copy(
            update={
                "version": 2,
                "payload": MessagePayload(
                    kind=ActionType.TRASH,
                    message_id="user_corrected_message_reference",
                ),
            }
        )
        revised_decision = PolicyEngine().decide(
            revised,
            edited_outcome.risk,
            AutonomyTier.ASK,
        )
        approvals.replace_for_edit(
            edited_outcome.approval.approval_id,
            original,
            revised,
            revised_decision,
            actor="user@example.com",
        )
        edited_feedback = feedback_service.record_approval_feedback(
            email=edited_outcome.email,
            proposal=original,
            decision=edited_outcome.decision,
            approval_id=edited_outcome.approval.approval_id,
            feedback=FeedbackType.EDITED,
            actor="user@example.com",
        )
        _require(edited_feedback.record.updated_state.beta == 3, "edit weight is wrong")
        _require(
            edited_feedback.record.updated_state.cooldown_remaining == 3,
            "edit did not activate cooldown",
        )
        checks += 2

        notify_email = _email(
            "notify",
            "Weekly newsletter",
            "This is your weekly digest.",
            sender="notify-feedback@example.com",
        )
        notify_context = _seed_approvals(learner, planner, notify_email, 6)
        notify_outcome = agent.process(notify_email).outcome
        _require(notify_outcome.execution is not None, "NOTIFY did not execute")
        _require(notify_outcome.proposal is not None, "NOTIFY lost proposal")
        correct = feedback_service.record_execution_feedback(
            email=notify_outcome.email,
            proposal=notify_outcome.proposal,
            decision=notify_outcome.decision,
            execution_id=notify_outcome.execution.execution_id,
            feedback=FeedbackType.CORRECT,
            actor="user@example.com",
        )
        _require(correct.record.previous_state.observations == 6, "wrong prior observations")
        _require(correct.record.updated_state.alpha == 8, "correct feedback did not add alpha")
        checks += 2

        undone = feedback_service.record_execution_feedback(
            email=notify_outcome.email,
            proposal=notify_outcome.proposal,
            decision=notify_outcome.decision,
            execution_id=notify_outcome.execution.execution_id,
            feedback=FeedbackType.UNDONE,
            actor="user@example.com",
        )
        _require(undone.record.previous_state.observations == 7, "latest state was not loaded")
        _require(undone.record.updated_state.beta == 4, "undo weight is wrong")
        _require(undone.record.updated_state.cooldown_remaining == 5, "undo missed cooldown")
        _require(store.get_preference(notify_context).observations == 8, "feedback update was lost")
        checks += 4

        failed_email = _email(
            "failed",
            "Weekly newsletter",
            "This is your weekly digest.",
            sender="failed-feedback@example.com",
        )
        _seed_approvals(learner, planner, failed_email, 6)
        failed_executor = MockMailboxExecutor((MockOutcome.RAISE_UNAVAILABLE,))
        failed_agent = _build_agent(store, learner, planner, approvals, failed_executor)
        failed_outcome = failed_agent.process(failed_email).outcome
        _require(
            failed_outcome.execution is not None
            and failed_outcome.execution.state == ExecutionState.FAILED_SAFE,
            "failure scenario did not produce FAILED_SAFE",
        )
        _require(failed_outcome.proposal is not None, "failure scenario lost proposal")
        try:
            feedback_service.record_execution_feedback(
                email=failed_outcome.email,
                proposal=failed_outcome.proposal,
                decision=failed_outcome.decision,
                execution_id=failed_outcome.execution.execution_id,
                feedback=FeedbackType.CORRECT,
                actor="user@example.com",
            )
        except FeedbackEvidenceError:
            pass
        else:
            raise RuntimeError("failed execution was accepted as correct evidence")
        checks += 2

        try:
            feedback_service.record_execution_feedback(
                email=notify_outcome.email,
                proposal=notify_outcome.proposal,
                decision=notify_outcome.decision,
                execution_id=notify_outcome.execution.execution_id,
                feedback=FeedbackType.APPROVED,
                actor="user@example.com",
            )
        except FeedbackBindingError:
            pass
        else:
            raise RuntimeError("approval feedback entered the execution feedback API")
        checks += 1

        feedback_events = store.read_stream(f"feedback:{correct.record.feedback_id}")
        preference_events = store.read_stream(f"preference:{notify_context}")
        _require(
            tuple(event.event_type for event in feedback_events) == ("feedback.recorded",),
            "feedback receipt was not audited",
        )
        _require(
            tuple(event.event_type for event in preference_events)
            == ("preference.updated", "preference.updated"),
            "preference transitions were not audited",
        )
        checks += 2

        with SQLiteStore(db_path) as second_store:
            second_service = FeedbackService(
                second_store,
                ContextualPreferenceLearner(second_store),
            )
            duplicate_after_restart = second_service.record_execution_feedback(
                email=notify_outcome.email,
                proposal=notify_outcome.proposal,
                decision=notify_outcome.decision,
                execution_id=notify_outcome.execution.execution_id,
                feedback=FeedbackType.CORRECT,
                actor="user@example.com",
            )
            _require(not duplicate_after_restart.applied, "restart forgot feedback deduplication")
            _require(
                second_store.get_preference(notify_context).observations == 8,
                "restart duplicate changed evidence",
            )
            checks += 2

        cold_other = _email(
            "other_sender",
            "Weekly newsletter",
            "This is your weekly digest.",
            sender="unrelated@example.com",
        )
        normalized_other, report_other = normalize_email(cold_other)
        request_other = build_planner_request(
            normalized_other,
            normalization_changed=report_other.changed,
        )
        proposal_other = bind_planner_output(request_other, planner.plan(request_other))
        other_context = build_preference_context(normalized_other, proposal_other)
        _require(
            store.get_preference(other_context.key).observations == 0,
            "feedback leaked into another sender context",
        )
        checks += 1

        feedback_id = correct.record.feedback_id
        expected_feedback = correct.record
        store.close()

        with SQLiteStore(db_path) as reopened:
            persisted = reopened.get_feedback(feedback_id)
            _require(persisted == expected_feedback, "feedback receipt was not durable")
            checks += 1

    print(f"Feedback checks passed: {checks}")


if __name__ == "__main__":
    main()
