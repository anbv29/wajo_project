"""Side-effect-free contract shared by every planner implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wajo_agent.domain import (
    ActionProposal,
    EmailEnvelope,
    LabelPayload,
    MessagePayload,
    PlannerOutput,
    PlannerRequest,
)
from wajo_agent.policy import CAPABILITIES


class PlannerError(RuntimeError):
    """Base class for planner failures that the agent must handle safely."""


class PlannerUnavailableError(PlannerError):
    """The planner could not produce an answer due to an operational failure."""


class PlannerContractError(PlannerError):
    """A planner answer violated its typed request contract."""


@runtime_checkable
class Planner(Protocol):
    """Interpret normalized email data without receiving execution tools."""

    def plan(self, request: PlannerRequest) -> PlannerOutput: ...


def build_planner_request(
    email: EmailEnvelope,
    *,
    normalization_changed: bool,
) -> PlannerRequest:
    """Create a request containing only actions enabled by application policy."""
    enabled_actions = tuple(
        action for action, capability in CAPABILITIES.items() if capability.enabled
    )
    return PlannerRequest(
        email=email,
        allowed_actions=enabled_actions,
        normalization_changed=normalization_changed,
    )


def validate_planner_output(request: PlannerRequest, output: PlannerOutput) -> None:
    """Reject an otherwise valid proposal if the request did not allow its action."""
    if output.action_type not in request.allowed_actions:
        raise PlannerContractError(
            f"planner proposed disallowed action: {output.action_type.value}"
        )
    if isinstance(output.payload, (MessagePayload, LabelPayload)) and (
        output.payload.message_id != request.email.provider_message_id
    ):
        raise PlannerContractError("message action targets a different provider message")


def bind_planner_output(request: PlannerRequest, output: PlannerOutput) -> ActionProposal:
    """Validate model-owned data, then attach application-owned email identity."""
    validate_planner_output(request, output)
    return output.bind_to_email(request.email.email_id)
