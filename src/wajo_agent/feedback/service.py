"""Evidence-backed explicit feedback with atomic contextual learning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hmac import compare_digest

from wajo_agent.approvals import approval_payload_hash
from wajo_agent.domain import (
    ActionProposal,
    ApprovalStatus,
    AutonomyTier,
    Decision,
    EmailEnvelope,
    ExecutionState,
    FeedbackRecord,
    FeedbackSubmission,
    FeedbackType,
    PreferenceState,
)
from wajo_agent.domain.models import new_id, utc_now
from wajo_agent.feedback.dedupe import feedback_dedupe_key
from wajo_agent.learning import ContextualPreferenceLearner, build_preference_context
from wajo_agent.policy import get_capability
from wajo_agent.storage import ApprovalNotFoundError, SQLiteStore

APPROVAL_FEEDBACK = frozenset({FeedbackType.APPROVED, FeedbackType.REJECTED, FeedbackType.EDITED})
EXECUTION_FEEDBACK = frozenset({FeedbackType.CORRECT, FeedbackType.UNDONE})


class FeedbackError(RuntimeError):
    """Base class for rejected or unavailable preference feedback."""


class FeedbackBindingError(FeedbackError):
    """The supplied objects do not describe one exact decision and proposal."""


class FeedbackEvidenceError(FeedbackError):
    """Durable approval or execution evidence does not support the feedback."""


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    """A durable receipt and whether this call applied new learning evidence."""

    record: FeedbackRecord
    applied: bool


class FeedbackService:
    """Accept only explicit feedback supported by a completed workflow fact."""

    def __init__(
        self,
        store: SQLiteStore,
        learner: ContextualPreferenceLearner,
        *,
        internal_domains: Iterable[str] = (),
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._store = store
        self._learner = learner
        self._internal_domains = tuple(internal_domains)
        self._clock = clock

    def record_approval_feedback(
        self,
        *,
        email: EmailEnvelope,
        proposal: ActionProposal,
        decision: Decision,
        approval_id: str,
        feedback: FeedbackType,
        actor: str,
        feedback_id: str | None = None,
    ) -> FeedbackResult:
        """Record approve, reject, or edit only when the approval state proves it."""
        self._validate_common(email, proposal, decision)
        if decision.tier != AutonomyTier.ASK:
            raise FeedbackBindingError("approval feedback requires an ASK decision")
        if feedback not in APPROVAL_FEEDBACK:
            raise FeedbackBindingError("feedback type does not belong to approval workflow")

        try:
            approval = self._store.get_approval(approval_id)
        except ApprovalNotFoundError as exc:
            raise FeedbackEvidenceError("approval evidence does not exist") from exc
        if (
            approval.proposal_id != proposal.proposal_id
            or approval.proposal_version != proposal.version
            or not compare_digest(approval.payload_hash, approval_payload_hash(proposal))
        ):
            raise FeedbackEvidenceError("approval evidence belongs to another proposal")

        allowed_statuses = {
            FeedbackType.APPROVED: {ApprovalStatus.GRANTED, ApprovalStatus.CONSUMED},
            FeedbackType.REJECTED: {ApprovalStatus.REJECTED},
            FeedbackType.EDITED: {ApprovalStatus.INVALIDATED},
        }[feedback]
        if approval.status not in allowed_statuses:
            raise FeedbackEvidenceError(
                f"approval status {approval.status.value} does not prove {feedback.value}"
            )
        return self._commit(
            email=email,
            proposal=proposal,
            decision=decision,
            feedback=feedback,
            actor=actor,
            source_reference=approval.approval_id,
            feedback_id=feedback_id,
        )

    def record_execution_feedback(
        self,
        *,
        email: EmailEnvelope,
        proposal: ActionProposal,
        decision: Decision,
        execution_id: str,
        feedback: FeedbackType,
        actor: str,
        feedback_id: str | None = None,
    ) -> FeedbackResult:
        """Record correct or undone only for a proven successful autonomous effect."""
        self._validate_common(email, proposal, decision)
        if decision.tier not in {AutonomyTier.SILENT, AutonomyTier.NOTIFY}:
            raise FeedbackBindingError("execution feedback requires SILENT or NOTIFY")
        if feedback not in EXECUTION_FEEDBACK:
            raise FeedbackBindingError("feedback type does not belong to execution workflow")

        execution = self._store.get_execution(execution_id)
        if execution is None or execution.state != ExecutionState.SUCCEEDED:
            raise FeedbackEvidenceError("successful execution evidence does not exist")
        command = execution.command
        if (
            command.email_id != email.email_id
            or command.proposal_id != proposal.proposal_id
            or command.proposal_version != proposal.version
            or command.action_type != proposal.action_type
            or command.payload != proposal.payload
        ):
            raise FeedbackEvidenceError("execution evidence belongs to another proposal")
        if feedback == FeedbackType.UNDONE and not get_capability(proposal.action_type).reversible:
            raise FeedbackEvidenceError("irreversible action cannot provide undo feedback")

        return self._commit(
            email=email,
            proposal=proposal,
            decision=decision,
            feedback=feedback,
            actor=actor,
            source_reference=execution.execution_id,
            feedback_id=feedback_id,
        )

    def _commit(
        self,
        *,
        email: EmailEnvelope,
        proposal: ActionProposal,
        decision: Decision,
        feedback: FeedbackType,
        actor: str,
        source_reference: str,
        feedback_id: str | None,
    ) -> FeedbackResult:
        context = build_preference_context(
            email,
            proposal,
            internal_domains=self._internal_domains,
        )
        submission = FeedbackSubmission(
            feedback_id=feedback_id or new_id("feedback"),
            dedupe_key=feedback_dedupe_key(decision, proposal, feedback),
            decision_id=decision.decision_id,
            proposal_id=proposal.proposal_id,
            proposal_version=proposal.version,
            context_key=context.key,
            feedback_type=feedback,
            actor=_clean_actor(actor),
            source_reference=source_reference,
            created_at=self._now(),
        )
        record, applied = self._store.commit_feedback(
            submission,
            update=lambda state: self._updated_state(state, feedback),
        )
        return FeedbackResult(record=record, applied=applied)

    def _updated_state(
        self,
        state: PreferenceState,
        feedback: FeedbackType,
    ) -> PreferenceState:
        return self._learner.updated_state(state, feedback)

    @staticmethod
    def _validate_common(
        email: EmailEnvelope,
        proposal: ActionProposal,
        decision: Decision,
    ) -> None:
        if proposal.email_id != email.email_id:
            raise FeedbackBindingError("proposal belongs to another email")
        if (
            decision.proposal_id != proposal.proposal_id
            or decision.proposal_version != proposal.version
        ):
            raise FeedbackBindingError("decision belongs to another proposal version")
        if decision.tier == AutonomyTier.ESCALATE:
            raise FeedbackBindingError("escalation is not preference-action feedback")

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise FeedbackError("feedback clock must return a timezone-aware time")
        return value.astimezone(UTC)


def _clean_actor(actor: str) -> str:
    cleaned = actor.strip()
    if not cleaned:
        raise FeedbackError("feedback actor cannot be blank")
    if len(cleaned) > 200:
        raise FeedbackError("feedback actor cannot exceed 200 characters")
    return cleaned
