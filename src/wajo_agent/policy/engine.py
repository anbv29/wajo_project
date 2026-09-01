from __future__ import annotations

from wajo_agent.domain import (
    ActionProposal,
    AutonomyTier,
    Decision,
    NormalizationFlag,
    RiskAssessment,
    SensitiveCategory,
)
from wajo_agent.domain.autonomy import is_at_least, most_restrictive
from wajo_agent.policy.capabilities import get_capability


class PolicyViolation(RuntimeError):
    """Raised when a decision contradicts an immutable authority rule."""


HARD_SENSITIVE_CATEGORIES: frozenset[SensitiveCategory] = frozenset(
    {
        SensitiveCategory.BANKING,
        SensitiveCategory.CREDENTIALS,
        SensitiveCategory.ACCOUNT_RECOVERY,
        SensitiveCategory.LEGAL_COMMITMENT,
        SensitiveCategory.PAYMENT,
        SensitiveCategory.PERSONAL_DATA,
    }
)

CAUTION_NORMALIZATION_FLAGS: frozenset[NormalizationFlag] = frozenset(
    {
        NormalizationFlag.TRUNCATED_CONTENT,
        NormalizationFlag.INVISIBLE_CHARACTERS,
        NormalizationFlag.CONTROL_CHARACTERS,
        NormalizationFlag.NO_VISIBLE_CONTENT,
    }
)


class PolicyEngine:
    """Choose autonomy with ordinary code that AI output cannot modify."""

    def decide(
        self,
        proposal: ActionProposal,
        risk: RiskAssessment,
        preference_tier: AutonomyTier,
    ) -> Decision:
        """Combine independent floors and return their most restrictive tier."""
        capability = get_capability(proposal.action_type)
        content_floor, reasons = self._content_floor(risk)

        if not capability.enabled:
            reasons.append(f"action {proposal.action_type.value} is disabled")

        final_tier = most_restrictive(
            capability.minimum_tier,
            content_floor,
            preference_tier,
        )

        reasons.append(
            f"capability floor for {proposal.action_type.value} is {capability.minimum_tier.value}"
        )
        reasons.append(f"preference recommendation is {preference_tier.value}")

        decision = Decision(
            proposal_id=proposal.proposal_id,
            proposal_version=proposal.version,
            tier=final_tier,
            capability_floor=capability.minimum_tier,
            preference_tier=preference_tier,
            content_floor=content_floor,
            reasons=tuple(dict.fromkeys(reasons)),
        )
        self.validate(decision, proposal, risk, preference_tier)
        return decision

    @staticmethod
    def _content_floor(risk: RiskAssessment) -> tuple[AutonomyTier, list[str]]:
        """Convert observed content risk into a deterministic minimum tier."""
        floors = [AutonomyTier.SILENT]
        reasons: list[str] = []

        if risk.injection_detected:
            floors.append(AutonomyTier.ESCALATE)
            signals = ", ".join(sorted(signal.value for signal in risk.injection_signals))
            reasons.append(f"prompt-injection indicators: {signals}")

        hard_categories = risk.sensitive_categories & HARD_SENSITIVE_CATEGORIES
        if hard_categories:
            floors.append(AutonomyTier.ESCALATE)
            categories = ", ".join(sorted(category.value for category in hard_categories))
            reasons.append(f"hard-sensitive content: {categories}")

        caution_flags = risk.normalization_flags & CAUTION_NORMALIZATION_FLAGS
        if caution_flags:
            floors.append(AutonomyTier.ASK)
            flags = ", ".join(sorted(flag.value for flag in caution_flags))
            reasons.append(f"normalization requires caution: {flags}")

        if risk.missing_information:
            floors.append(AutonomyTier.ASK)
            reasons.append("required information is missing or ambiguous")

        return most_restrictive(*floors), reasons

    def validate(
        self,
        decision: Decision,
        proposal: ActionProposal,
        risk: RiskAssessment,
        expected_preference_tier: AutonomyTier,
    ) -> None:
        """Recompute the expected result and reject any inconsistent decision."""
        capability = get_capability(proposal.action_type)
        expected_content_floor, _ = self._content_floor(risk)
        expected_tier = most_restrictive(
            capability.minimum_tier,
            expected_content_floor,
            expected_preference_tier,
        )

        if decision.proposal_id != proposal.proposal_id:
            raise PolicyViolation("decision is bound to the wrong proposal")
        if decision.proposal_version != proposal.version:
            raise PolicyViolation("decision is bound to the wrong proposal version")
        if decision.capability_floor != capability.minimum_tier:
            raise PolicyViolation("decision contains an incorrect capability floor")
        if decision.content_floor != expected_content_floor:
            raise PolicyViolation("decision contains an incorrect content-risk floor")
        if decision.preference_tier != expected_preference_tier:
            raise PolicyViolation("decision contains an incorrect preference recommendation")
        if decision.tier != expected_tier:
            raise PolicyViolation("decision is not the most restrictive required tier")
        if not is_at_least(decision.tier, capability.minimum_tier):
            raise PolicyViolation("decision is below capability safety floor")
        if capability.external and not is_at_least(decision.tier, AutonomyTier.ASK):
            raise PolicyViolation("external action cannot be below ASK")
        if capability.destructive and not is_at_least(decision.tier, AutonomyTier.ASK):
            raise PolicyViolation("destructive action cannot be below ASK")
        if (capability.financial or not capability.enabled) and decision.tier != (
            AutonomyTier.ESCALATE
        ):
            raise PolicyViolation("disabled or financial action must escalate")
        if risk.injection_detected and decision.tier != AutonomyTier.ESCALATE:
            raise PolicyViolation("injection-affected decision must escalate")
        if (
            risk.sensitive_categories & HARD_SENSITIVE_CATEGORIES
            and decision.tier != AutonomyTier.ESCALATE
        ):
            raise PolicyViolation("hard-sensitive decision must escalate")
        if risk.missing_information and not is_at_least(decision.tier, AutonomyTier.ASK):
            raise PolicyViolation("incomplete decision cannot be below ASK")
        if risk.normalization_flags & CAUTION_NORMALIZATION_FLAGS and not is_at_least(
            decision.tier, AutonomyTier.ASK
        ):
            raise PolicyViolation("normalization-affected decision cannot be below ASK")
