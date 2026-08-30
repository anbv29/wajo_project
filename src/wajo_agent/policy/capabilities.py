from __future__ import annotations

from dataclasses import dataclass

from wajo_agent.domain import ActionType, AutonomyTier


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    action_type: ActionType
    reversible: bool
    external: bool
    destructive: bool
    financial: bool
    enabled: bool
    minimum_tier: AutonomyTier


def _cap(
    action: ActionType,
    *,
    reversible: bool,
    external: bool = False,
    destructive: bool = False,
    financial: bool = False,
    enabled: bool = True,
    minimum: AutonomyTier = AutonomyTier.SILENT,
) -> CapabilitySpec:
    return CapabilitySpec(
        action_type=action,
        reversible=reversible,
        external=external,
        destructive=destructive,
        financial=financial,
        enabled=enabled,
        minimum_tier=minimum,
    )


CAPABILITIES: dict[ActionType, CapabilitySpec] = {
    ActionType.NO_ACTION: _cap(ActionType.NO_ACTION, reversible=True),
    ActionType.MARK_READ: _cap(ActionType.MARK_READ, reversible=True),
    ActionType.MARK_UNREAD: _cap(ActionType.MARK_UNREAD, reversible=True),
    ActionType.ADD_LABEL: _cap(ActionType.ADD_LABEL, reversible=True),
    ActionType.ARCHIVE: _cap(ActionType.ARCHIVE, reversible=True),
    ActionType.CREATE_DRAFT: _cap(
        ActionType.CREATE_DRAFT,
        reversible=True,
        minimum=AutonomyTier.NOTIFY,
    ),
    ActionType.TRASH: _cap(
        ActionType.TRASH,
        reversible=True,
        destructive=True,
        minimum=AutonomyTier.ASK,
    ),
    ActionType.SEND_REPLY: _cap(
        ActionType.SEND_REPLY,
        reversible=False,
        external=True,
        minimum=AutonomyTier.ASK,
    ),
    ActionType.FORWARD: _cap(
        ActionType.FORWARD,
        reversible=False,
        external=True,
        minimum=AutonomyTier.ASK,
    ),
    ActionType.UNSUBSCRIBE: _cap(
        ActionType.UNSUBSCRIBE,
        reversible=False,
        external=True,
        minimum=AutonomyTier.ASK,
    ),
    ActionType.PERMANENT_DELETE: _cap(
        ActionType.PERMANENT_DELETE,
        reversible=False,
        destructive=True,
        enabled=False,
        minimum=AutonomyTier.ESCALATE,
    ),
    ActionType.PAYMENT: _cap(
        ActionType.PAYMENT,
        reversible=False,
        external=True,
        financial=True,
        enabled=False,
        minimum=AutonomyTier.ESCALATE,
    ),
    ActionType.ACCOUNT_CHANGE: _cap(
        ActionType.ACCOUNT_CHANGE,
        reversible=False,
        external=True,
        enabled=False,
        minimum=AutonomyTier.ESCALATE,
    ),
}


if set(CAPABILITIES) != set(ActionType):
    raise RuntimeError("every action type must have exactly one capability specification")
