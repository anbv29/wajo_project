"""Fast deterministic smoke checks for the immutable policy boundary."""

from __future__ import annotations

from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    AutonomyTier,
    DraftPayload,
    InjectionSignal,
    Intent,
    LabelPayload,
    MessagePayload,
    NoActionPayload,
    NormalizationFlag,
    ReplyPayload,
    RiskAssessment,
    SensitiveCategory,
    most_restrictive,
)
from wajo_agent.policy import CAPABILITIES, PolicyEngine, PolicyViolation


def _proposal(action: ActionType) -> ActionProposal:
    if action == ActionType.NO_ACTION:
        payload = NoActionPayload()
    elif action in {
        ActionType.MARK_READ,
        ActionType.MARK_UNREAD,
        ActionType.ARCHIVE,
        ActionType.TRASH,
        ActionType.PERMANENT_DELETE,
    }:
        payload = MessagePayload(kind=action, message_id="provider-message-1")
    elif action == ActionType.ADD_LABEL:
        payload = LabelPayload(message_id="provider-message-1", label="Agent/Test")
    elif action == ActionType.CREATE_DRAFT:
        payload = DraftPayload(subject="Draft", body="A safe draft")
    else:
        payload = ReplyPayload(
            kind=action,
            recipients=("recipient@example.com",),
            subject="Proposed action",
            body="This payload is used only by the policy smoke check.",
        )

    return ActionProposal(
        email_id="email-policy-check",
        action_type=action,
        intent=Intent.UNKNOWN,
        summary=f"Policy check for {action.value}",
        payload=payload,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    engine = PolicyEngine()
    clean_risk = RiskAssessment()
    checks = 0

    for action, capability in CAPABILITIES.items():
        proposal = _proposal(action)
        for preference_tier in AutonomyTier:
            decision = engine.decide(proposal, clean_risk, preference_tier)
            expected = most_restrictive(capability.minimum_tier, preference_tier)
            _require(
                decision.tier == expected,
                f"{action.value} with {preference_tier.value} produced the wrong tier",
            )
            checks += 1

    archive = _proposal(ActionType.ARCHIVE)
    risk_scenarios = (
        (
            RiskAssessment(injection_signals=frozenset({InjectionSignal.INSTRUCTION_OVERRIDE})),
            AutonomyTier.ESCALATE,
            "prompt injection",
        ),
        (
            RiskAssessment(sensitive_categories=frozenset({SensitiveCategory.PAYMENT})),
            AutonomyTier.ESCALATE,
            "money-related content",
        ),
        (
            RiskAssessment(sensitive_categories=frozenset({SensitiveCategory.PERSONAL_DATA})),
            AutonomyTier.ESCALATE,
            "personal data",
        ),
        (
            RiskAssessment(normalization_flags=frozenset({NormalizationFlag.TRUNCATED_CONTENT})),
            AutonomyTier.ASK,
            "truncated content",
        ),
        (
            RiskAssessment(missing_information=("recipient intent",)),
            AutonomyTier.ASK,
            "missing information",
        ),
    )
    for risk, expected, label in risk_scenarios:
        decision = engine.decide(archive, risk, AutonomyTier.SILENT)
        _require(decision.tier == expected, f"{label} did not fail closed")
        checks += 1

    valid = engine.decide(archive, clean_risk, AutonomyTier.ASK)
    tampered = valid.model_copy(update={"tier": AutonomyTier.SILENT})
    try:
        engine.validate(tampered, archive, clean_risk, AutonomyTier.ASK)
    except PolicyViolation:
        checks += 1
    else:
        raise RuntimeError("the independent validator accepted a downgraded decision")

    fully_tampered = valid.model_copy(
        update={
            "preference_tier": AutonomyTier.SILENT,
            "tier": AutonomyTier.SILENT,
        }
    )
    try:
        engine.validate(fully_tampered, archive, clean_risk, AutonomyTier.ASK)
    except PolicyViolation:
        checks += 1
    else:
        raise RuntimeError("the validator accepted a forged preference recommendation")

    print(f"Policy checks passed: {checks}")


if __name__ == "__main__":
    main()
