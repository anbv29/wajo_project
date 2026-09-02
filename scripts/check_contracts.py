"""Fast checks for planner/executor separation and contract validation."""

from __future__ import annotations

from pydantic import ValidationError

from wajo_agent.domain import (
    ActionType,
    AutonomyTier,
    DraftPayload,
    EmailEnvelope,
    ExecutionCommand,
    ExecutionResult,
    ExecutionState,
    Intent,
    MessagePayload,
    NoActionPayload,
    PlannerOutput,
    PlannerRequest,
    ReplyPayload,
)
from wajo_agent.execution import (
    ExecutorContractError,
    MailboxExecutor,
    execute_checked,
    validate_execution_command,
)
from wajo_agent.planning import (
    Planner,
    PlannerContractError,
    bind_planner_output,
    build_planner_request,
    validate_planner_output,
)


class FakePlanner:
    def plan(self, request: PlannerRequest) -> PlannerOutput:
        del request
        return PlannerOutput(
            action_type=ActionType.NO_ACTION,
            intent=Intent.INFORMATIONAL,
            summary="No mailbox change is required",
            payload=NoActionPayload(),
        )


class FakeExecutor:
    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        return ExecutionResult(
            execution_id=command.execution_id,
            command_id=command.command_id,
            idempotency_key=command.idempotency_key,
            proposal_id=command.proposal_id,
            state=ExecutionState.SUCCEEDED,
            detail="Fixture effect completed",
        )


class WrongResultExecutor:
    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        return ExecutionResult(
            execution_id=command.execution_id,
            command_id="wrong-command",
            idempotency_key=command.idempotency_key,
            proposal_id=command.proposal_id,
            state=ExecutionState.SUCCEEDED,
            detail="This result should be rejected",
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _archive_command() -> ExecutionCommand:
    return ExecutionCommand(
        idempotency_key="archive-once",
        email_id="email-1",
        proposal_id="proposal-1",
        proposal_version=1,
        decision_id="decision-1",
        action_type=ActionType.ARCHIVE,
        payload=MessagePayload(kind=ActionType.ARCHIVE, message_id="provider-message-1"),
        authorized_tier=AutonomyTier.SILENT,
    )


def main() -> None:
    checks = 0
    email = EmailEnvelope(
        provider_message_id="provider-message-1",
        sender="sender@example.com",
        subject="Contract check",
    )
    request = build_planner_request(email, normalization_changed=False)
    planner = FakePlanner()

    _require(isinstance(planner, Planner), "planner does not satisfy the protocol")
    checks += 1
    _require(ActionType.PAYMENT not in request.allowed_actions, "disabled action reached planner")
    checks += 1

    output = planner.plan(request)
    proposal = bind_planner_output(request, output)
    _require(proposal.email_id == email.email_id, "output was bound to the wrong email")
    checks += 1

    disallowed = PlannerOutput(
        action_type=ActionType.PAYMENT,
        intent=Intent.FINANCIAL,
        summary="Attempt a disallowed payment",
        payload=ReplyPayload(kind=ActionType.PAYMENT, body="Do not execute"),
    )
    try:
        validate_planner_output(request, disallowed)
    except PlannerContractError:
        checks += 1
    else:
        raise RuntimeError("planner contract accepted a disallowed action")

    executor = FakeExecutor()
    _require(isinstance(executor, MailboxExecutor), "executor does not satisfy the protocol")
    checks += 1
    command = _archive_command()
    result = execute_checked(executor, command)
    _require(result.state == ExecutionState.SUCCEEDED, "valid executor result was rejected")
    checks += 1

    try:
        execute_checked(WrongResultExecutor(), command)
    except ExecutorContractError:
        checks += 1
    else:
        raise RuntimeError("executor contract accepted a result for another command")

    try:
        ExecutionCommand(
            **command.model_dump(exclude={"authorized_tier"}),
            authorized_tier=AutonomyTier.ASK,
        )
    except ValidationError:
        checks += 1
    else:
        raise RuntimeError("ASK command was created without approval")

    under_floor = command.model_copy(
        update={
            "action_type": ActionType.CREATE_DRAFT,
            "payload": DraftPayload(body="Draft"),
        }
    )
    try:
        validate_execution_command(under_floor)
    except ExecutorContractError:
        checks += 1
    else:
        raise RuntimeError("command below capability floor was accepted")

    payment = ExecutionCommand(
        idempotency_key="payment-once",
        email_id="email-1",
        proposal_id="proposal-payment",
        proposal_version=1,
        decision_id="decision-payment",
        action_type=ActionType.PAYMENT,
        payload=ReplyPayload(kind=ActionType.PAYMENT, body="Do not execute"),
        authorized_tier=AutonomyTier.ASK,
        approval_id="approval-payment",
    )
    try:
        validate_execution_command(payment)
    except ExecutorContractError:
        checks += 1
    else:
        raise RuntimeError("disabled financial command was accepted")

    print(f"Contract checks passed: {checks}")


if __name__ == "__main__":
    main()
