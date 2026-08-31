"""The four autonomy outcomes and their exact operational meaning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AutonomyTier(StrEnum):
    """How much freedom the agent has for one proposed action."""

    SILENT = "silent"
    NOTIFY = "notify"
    ASK = "ask"
    ESCALATE = "escalate"


TIER_RANK: dict[AutonomyTier, int] = {
    AutonomyTier.SILENT: 0,
    AutonomyTier.NOTIFY: 1,
    AutonomyTier.ASK: 2,
    AutonomyTier.ESCALATE: 3,
}


@dataclass(frozen=True, slots=True)
class AutonomyBehavior:
    """The side-effect and user-interaction rules for one autonomy tier."""

    execute_automatically: bool
    notify_after_execution: bool
    requires_user_approval: bool
    requires_human_review: bool


AUTONOMY_BEHAVIORS: dict[AutonomyTier, AutonomyBehavior] = {
    AutonomyTier.SILENT: AutonomyBehavior(
        execute_automatically=True,
        notify_after_execution=False,
        requires_user_approval=False,
        requires_human_review=False,
    ),
    AutonomyTier.NOTIFY: AutonomyBehavior(
        execute_automatically=True,
        notify_after_execution=True,
        requires_user_approval=False,
        requires_human_review=False,
    ),
    AutonomyTier.ASK: AutonomyBehavior(
        execute_automatically=False,
        notify_after_execution=False,
        requires_user_approval=True,
        requires_human_review=False,
    ),
    AutonomyTier.ESCALATE: AutonomyBehavior(
        execute_automatically=False,
        notify_after_execution=False,
        requires_user_approval=False,
        requires_human_review=True,
    ),
}


if set(TIER_RANK) != set(AutonomyTier):
    raise RuntimeError("every autonomy tier must have a safety rank")

if set(AUTONOMY_BEHAVIORS) != set(AutonomyTier):
    raise RuntimeError("every autonomy tier must have an operational behavior")

for _tier, _behavior in AUTONOMY_BEHAVIORS.items():
    if _behavior.execute_automatically and (
        _behavior.requires_user_approval or _behavior.requires_human_review
    ):
        raise RuntimeError(f"{_tier} cannot execute automatically while waiting for a human")
    if _behavior.notify_after_execution and not _behavior.execute_automatically:
        raise RuntimeError(f"{_tier} cannot notify after an execution that did not occur")
    if _behavior.requires_user_approval and _behavior.requires_human_review:
        raise RuntimeError(f"{_tier} cannot be both an approval request and an escalation")


def behavior_for(tier: AutonomyTier) -> AutonomyBehavior:
    """Return the routing instructions for a final autonomy decision."""
    return AUTONOMY_BEHAVIORS[tier]


def is_at_least(actual: AutonomyTier, required_floor: AutonomyTier) -> bool:
    """Return whether actual is as restrictive as the required safety floor."""
    return TIER_RANK[actual] >= TIER_RANK[required_floor]


def most_restrictive(*tiers: AutonomyTier) -> AutonomyTier:
    """Combine independent opinions without allowing a less-safe result to win."""
    if not tiers:
        raise ValueError("at least one autonomy tier is required")
    return max(tiers, key=TIER_RANK.__getitem__)
