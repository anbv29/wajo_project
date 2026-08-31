from __future__ import annotations

from wajo_agent.domain import (
    ActionProposal,
    AutonomyTier,
    Decision,
    PreferenceState,
    RiskAssessment,
)
from wajo_agent.domain.autonomy import TIER_RANK, most_restrictive
from wajo_agent.policy.capabilities import CAPABILITIES, CapabilitySpec


class PolicyViolation(RuntimeError):
    """Raised when a decision contradicts an immutable authority rule."""


HARD_SENSITIVE_CATEGORIES = frozenset(
    {"banking", "credentials", "account_recovery", "legal_commitment", "payment"}
)


class PolicyEngine:
    def decide(
        self,
        proposal: ActionProposal,
        risk: RiskAssessment,
        preference_tier: AutonomyTier,
        _preference_state: PreferenceState,
    ) -> Decision:
        capability = CAPABILITIES[proposal.action_type]
        reasons: list[str] = []
        content_floor = AutonomyTier.SILENT

        if not capability.enabled:
            content_floor = AutonomyTier.ESCALATE
            reasons.append(f"{proposal.action_type} is disabled")

        if risk.injection_detected:
            content_floor = AutonomyTier.ESCALATE
            reasons.append("untrusted email contains prompt-injection indicators")

        hard_categories = risk.sensitive_categories & HARD_SENSITIVE_CATEGORIES
        if hard_categories:
            content_floor = AutonomyTier.ESCALATE
            reasons.append("hard-sensitive content: " + ", ".join(sorted(hard_categories)))

        if risk.missing_information and content_floor != AutonomyTier.ESCALATE:
            content_floor = AutonomyTier.ASK
            reasons.append("required information is missing or ambiguous")

        final_tier = most_restrictive(
            capability.minimum_tier,
            content_floor,
            preference_tier,
        )

        if final_tier == capability.minimum_tier:
            reasons.append(f"capability floor is {capability.minimum_tier}")
        if final_tier == preference_tier:
            reasons.append(f"preference evidence recommends {preference_tier}")
        if not reasons:
            reasons.append("all safety and preference checks passed")

        decision = Decision(
            proposal_id=proposal.proposal_id,
            proposal_version=proposal.version,
            tier=final_tier,
            capability_floor=capability.minimum_tier,
            preference_tier=preference_tier,
            content_floor=content_floor,
            reasons=tuple(dict.fromkeys(reasons)),
        )
        self.validate(decision, capability, risk)
        return decision

    def validate(
        self,
        decision: Decision,
        capability: CapabilitySpec,
        risk: RiskAssessment,
    ) -> None:
        if TIER_RANK[decision.tier] < TIER_RANK[capability.minimum_tier]:
            raise PolicyViolation("decision is below capability safety floor")
        if capability.external and TIER_RANK[decision.tier] < TIER_RANK[AutonomyTier.ASK]:
            raise PolicyViolation("external action cannot be below ASK")
        if (capability.financial or not capability.enabled) and (
            decision.tier != AutonomyTier.ESCALATE
        ):
            raise PolicyViolation("disabled or financial action must escalate")
        if risk.injection_detected and decision.tier != AutonomyTier.ESCALATE:
            raise PolicyViolation("injection-affected decision must escalate")
