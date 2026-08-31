from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from wajo_agent.domain import ActionType, AutonomyTier, is_at_least


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    action_type: ActionType
    reversible: bool
    external: bool
    destructive: bool
    financial: bool
    enabled: bool
    minimum_tier: AutonomyTier

    def __post_init__(self) -> None:
        if self.external and not is_at_least(self.minimum_tier, AutonomyTier.ASK):
            raise ValueError("external actions must have a minimum tier of ASK or ESCALATE")
        if self.destructive and not is_at_least(self.minimum_tier, AutonomyTier.ASK):
            raise ValueError("destructive actions must have a minimum tier of ASK or ESCALATE")
        if self.financial and self.minimum_tier != AutonomyTier.ESCALATE:
            raise ValueError("financial actions must always escalate")
        if not self.enabled and self.minimum_tier != AutonomyTier.ESCALATE:
            raise ValueError("disabled actions must always escalate")

    @property
    def silent_eligible(self) -> bool:
        """Whether preference learning may ever recommend silent execution."""
        return (
            self.enabled
            and self.reversible
            and not self.external
            and not self.destructive
            and not self.financial
            and self.minimum_tier == AutonomyTier.SILENT
        )


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


CAPABILITIES: Mapping[ActionType, CapabilitySpec] = MappingProxyType(
    {
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
)


if set(CAPABILITIES) != set(ActionType):
    raise RuntimeError("every action type must have exactly one capability specification")

if any(spec.action_type != action for action, spec in CAPABILITIES.items()):
    raise RuntimeError("capability keys must match their action specifications")


def get_capability(action: ActionType) -> CapabilitySpec:
    """Return the immutable authority limits for an action."""
    return CAPABILITIES[action]
