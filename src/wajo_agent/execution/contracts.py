"""Narrow contract for code that may perform mailbox side effects."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wajo_agent.domain import ExecutionCommand, ExecutionResult, is_at_least
from wajo_agent.policy import get_capability


class ExecutorError(RuntimeError):
    """Base class for executor failures."""


class ExecutorUnavailableError(ExecutorError):
    """The executor was unavailable before an effect could safely begin."""


class ExecutorContractError(ExecutorError):
    """A command or result contradicted the executor boundary."""


@runtime_checkable
class MailboxExecutor(Protocol):
    """Perform one pre-authorized command; never plan or choose autonomy."""

    def execute(self, command: ExecutionCommand) -> ExecutionResult: ...


def validate_execution_command(command: ExecutionCommand) -> None:
    """Enforce capability-level invariants before an adapter sees a command."""
    capability = get_capability(command.action_type)
    if not capability.enabled:
        raise ExecutorContractError(f"disabled action cannot execute: {command.action_type.value}")
    if not is_at_least(command.authorized_tier, capability.minimum_tier):
        raise ExecutorContractError("command is below the action's capability floor")
    if capability.external and command.approval_id is None:
        raise ExecutorContractError("external command requires a bound approval")


def validate_execution_result(command: ExecutionCommand, result: ExecutionResult) -> None:
    """Ensure an adapter result belongs to the exact command that was sent."""
    if result.command_id != command.command_id:
        raise ExecutorContractError("executor result belongs to a different command")
    if result.idempotency_key != command.idempotency_key:
        raise ExecutorContractError("executor result has a different idempotency key")
    if result.proposal_id != command.proposal_id:
        raise ExecutorContractError("executor result belongs to a different proposal")


def execute_checked(executor: MailboxExecutor, command: ExecutionCommand) -> ExecutionResult:
    """Apply both contract checks around a mailbox adapter call."""
    validate_execution_command(command)
    result = executor.execute(command)
    validate_execution_result(command, result)
    return result
