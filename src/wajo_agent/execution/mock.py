"""Deterministic mailbox adapter used by the offline demo and failure tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from enum import StrEnum

from wajo_agent.domain import ExecutionCommand, ExecutionResult, ExecutionState
from wajo_agent.execution.contracts import (
    ExecutorOutcomeUnknownError,
    ExecutorUnavailableError,
)


class MockOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED_SAFE = "failed_safe"
    UNKNOWN = "unknown"
    RAISE_UNAVAILABLE = "raise_unavailable"
    RAISE_UNKNOWN = "raise_unknown"
    RAISE_UNEXPECTED = "raise_unexpected"


class MockMailboxExecutor:
    """Return scripted outcomes while counting how many effects were attempted."""

    def __init__(self, outcomes: Iterable[MockOutcome] = ()) -> None:
        self._outcomes = deque(outcomes)
        self.commands: list[ExecutionCommand] = []

    @property
    def call_count(self) -> int:
        return len(self.commands)

    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        self.commands.append(command)
        outcome = self._outcomes.popleft() if self._outcomes else MockOutcome.SUCCEEDED
        if outcome == MockOutcome.RAISE_UNAVAILABLE:
            raise ExecutorUnavailableError("mock provider unavailable before effect")
        if outcome == MockOutcome.RAISE_UNKNOWN:
            raise ExecutorOutcomeUnknownError("mock provider outcome is unknown")
        if outcome == MockOutcome.RAISE_UNEXPECTED:
            raise RuntimeError("mock unexpected adapter failure")

        state = ExecutionState(outcome.value)
        return ExecutionResult(
            execution_id=command.execution_id,
            command_id=command.command_id,
            idempotency_key=command.idempotency_key,
            proposal_id=command.proposal_id,
            state=state,
            detail=f"Mock adapter returned {state.value}",
            provider_operation_id=(
                f"mock-operation:{command.execution_id}"
                if state == ExecutionState.SUCCEEDED
                else None
            ),
        )
