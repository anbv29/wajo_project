"""Policy-checked, approval-aware, idempotent execution orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from wajo_agent.approvals import ApprovalError, ApprovalService
from wajo_agent.domain import (
    ActionProposal,
    ApprovalRecord,
    AutonomyTier,
    Decision,
    ExecutionCommand,
    ExecutionRecord,
    ExecutionResult,
    ExecutionState,
    RiskAssessment,
)
from wajo_agent.domain.models import utc_now
from wajo_agent.execution.contracts import (
    ExecutorContractError,
    ExecutorOutcomeUnknownError,
    ExecutorUnavailableError,
    MailboxExecutor,
    execute_checked,
    validate_execution_command,
)
from wajo_agent.execution.idempotency import effect_idempotency_key
from wajo_agent.policy import PolicyEngine, PolicyViolation
from wajo_agent.storage import (
    ApprovalStateConflictError,
    SQLiteStore,
    StorageError,
)


class ExecutionError(RuntimeError):
    """Base class for safe execution-orchestration failures."""


class ExecutionAuthorizationError(ExecutionError):
    """Current policy or approval evidence does not authorize execution."""


class ExecutionInProgressError(ExecutionError):
    """The same effect already has an unresolved execution claim."""


class ExecutionPersistenceError(ExecutionError):
    """The effect may have occurred but its final state could not be persisted."""


class ExecutionService:
    """The only application service allowed to invoke a mailbox executor."""

    def __init__(
        self,
        store: SQLiteStore,
        executor: MailboxExecutor,
        *,
        approval_service: ApprovalService | None = None,
        policy: PolicyEngine | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._store = store
        self._executor = executor
        self._approval_service = approval_service
        self._policy = policy or PolicyEngine()
        self._clock = clock

    def execute(
        self,
        *,
        mailbox_id: str,
        provider_message_id: str,
        proposal: ActionProposal,
        decision: Decision,
        risk: RiskAssessment,
        preference_tier: AutonomyTier,
        approval_id: str | None = None,
    ) -> ExecutionResult:
        """Execute at most once after revalidating every authority boundary."""
        authorized_tier = self._validate_authority(
            proposal,
            decision,
            risk,
            preference_tier,
            approval_id,
        )
        try:
            command = ExecutionCommand(
                idempotency_key=effect_idempotency_key(
                    mailbox_id,
                    provider_message_id,
                    proposal,
                ),
                email_id=proposal.email_id,
                proposal_id=proposal.proposal_id,
                proposal_version=proposal.version,
                decision_id=decision.decision_id,
                action_type=proposal.action_type,
                payload=proposal.payload,
                authorized_tier=authorized_tier,
                approval_id=approval_id,
            )
            validate_execution_command(command)
        except (ValueError, ExecutorContractError) as exc:
            raise ExecutionAuthorizationError("execution command is invalid") from exc

        try:
            existing = self._store.get_execution_by_idempotency_key(command.idempotency_key)
        except StorageError as exc:
            raise ExecutionPersistenceError("execution state could not be read") from exc
        if existing is not None:
            return _existing_result(existing)

        consumed_approval = self._prepare_approval(command, proposal)
        now = self._now()
        claim = ExecutionRecord(
            execution_id=command.execution_id,
            command=command,
            state=ExecutionState.EXECUTING,
            detail="Execution claimed before mailbox adapter call",
            created_at=now,
            updated_at=now,
            started_at=now,
        )
        try:
            claimed, created = self._store.claim_execution(
                claim,
                consumed_approval=consumed_approval,
            )
        except (ApprovalStateConflictError, ApprovalError) as exc:
            raise ExecutionAuthorizationError(
                "approval changed before execution could claim it"
            ) from exc
        except StorageError as exc:
            raise ExecutionPersistenceError("execution claim could not be persisted") from exc

        if not created:
            return _existing_result(claimed)

        result = self._call_adapter(command)
        completed_at = self._now()
        terminal = claimed.model_copy(
            update={
                "state": result.state,
                "detail": result.detail,
                "provider_operation_id": result.provider_operation_id,
                "updated_at": completed_at,
                "completed_at": completed_at,
            }
        )
        try:
            completed = self._store.complete_execution(terminal)
        except StorageError as exc:
            raise ExecutionPersistenceError(
                "mailbox returned but terminal execution state was not committed"
            ) from exc
        return completed.as_result()

    def _validate_authority(
        self,
        proposal: ActionProposal,
        decision: Decision,
        risk: RiskAssessment,
        preference_tier: AutonomyTier,
        approval_id: str | None,
    ) -> Literal[AutonomyTier.SILENT, AutonomyTier.NOTIFY, AutonomyTier.ASK]:
        try:
            self._policy.validate(decision, proposal, risk, preference_tier)
        except PolicyViolation as exc:
            raise ExecutionAuthorizationError("current policy rejected execution") from exc
        if decision.tier == AutonomyTier.ESCALATE:
            raise ExecutionAuthorizationError("ESCALATE decisions cannot execute")
        if decision.tier == AutonomyTier.ASK and approval_id is None:
            raise ExecutionAuthorizationError("ASK execution requires approval")
        if decision.tier != AutonomyTier.ASK and approval_id is not None:
            raise ExecutionAuthorizationError("only ASK execution may consume approval")
        return decision.tier

    def _prepare_approval(
        self,
        command: ExecutionCommand,
        proposal: ActionProposal,
    ) -> ApprovalRecord | None:
        if command.approval_id is None:
            return None
        if self._approval_service is None:
            raise ExecutionAuthorizationError("approval service is not configured")
        try:
            return self._approval_service.prepare_consumption(
                command.approval_id,
                proposal,
            )
        except ApprovalError as exc:
            raise ExecutionAuthorizationError("approval is not executable") from exc

    def _call_adapter(self, command: ExecutionCommand) -> ExecutionResult:
        try:
            return execute_checked(self._executor, command)
        except ExecutorUnavailableError:
            return _failure_result(
                command,
                state=ExecutionState.FAILED_SAFE,
                detail="Mailbox adapter was unavailable before an effect began",
            )
        except ExecutorOutcomeUnknownError:
            return _failure_result(
                command,
                state=ExecutionState.UNKNOWN,
                detail="Mailbox adapter could not prove whether the effect occurred",
            )
        except ExecutorContractError:
            return _failure_result(
                command,
                state=ExecutionState.UNKNOWN,
                detail="Mailbox returned an invalid result; effect status is unknown",
            )
        except Exception as exc:
            return _failure_result(
                command,
                state=ExecutionState.UNKNOWN,
                detail=f"Unexpected {type(exc).__name__}; effect status is unknown",
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ExecutionError("execution clock must return a timezone-aware time")
        return value.astimezone(UTC)


def _existing_result(record: ExecutionRecord) -> ExecutionResult:
    if record.state == ExecutionState.EXECUTING:
        raise ExecutionInProgressError(
            f"effect already has an unresolved claim: {record.execution_id}"
        )
    return record.as_result()


def _failure_result(
    command: ExecutionCommand,
    *,
    state: ExecutionState,
    detail: str,
) -> ExecutionResult:
    return ExecutionResult(
        execution_id=command.execution_id,
        command_id=command.command_id,
        idempotency_key=command.idempotency_key,
        proposal_id=command.proposal_id,
        state=state,
        detail=detail,
    )
