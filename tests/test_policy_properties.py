from __future__ import annotations

from types import MappingProxyType

from hypothesis import given, settings
from hypothesis import strategies as st

from wajo_agent.domain import (
    TIER_RANK,
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
    behavior_for,
)
from wajo_agent.policy import CAPABILITIES, PolicyEngine


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
        payload = MessagePayload(kind=action, message_id="property-message")
    elif action == ActionType.ADD_LABEL:
        payload = LabelPayload(message_id="property-message", label="Evaluation")
    elif action == ActionType.CREATE_DRAFT:
        payload = DraftPayload(subject="Draft", body="Synthetic evaluation draft")
    else:
        payload = ReplyPayload(
            kind=action,
            recipients=("outside@synthetic.example",),
            body="Synthetic evaluation payload",
        )
    return ActionProposal(
        email_id="property-email",
        action_type=action,
        intent=Intent.UNKNOWN,
        summary="Property-generated policy proposal",
        payload=payload,
    )


@settings(max_examples=1_000, derandomize=True, deadline=None)
@given(
    action=st.sampled_from(tuple(ActionType)),
    preference=st.sampled_from(tuple(AutonomyTier)),
    injection=st.sets(st.sampled_from(tuple(InjectionSignal)), max_size=3),
    sensitive=st.sets(st.sampled_from(tuple(SensitiveCategory)), max_size=3),
    normalization=st.sets(st.sampled_from(tuple(NormalizationFlag)), max_size=2),
    missing=st.booleans(),
)
def test_policy_never_drops_below_any_authority_floor(
    action: ActionType,
    preference: AutonomyTier,
    injection: set[InjectionSignal],
    sensitive: set[SensitiveCategory],
    normalization: set[NormalizationFlag],
    missing: bool,
) -> None:
    capability = CAPABILITIES[action]
    risk = RiskAssessment(
        injection_signals=frozenset(injection),
        sensitive_categories=frozenset(sensitive),
        normalization_flags=frozenset(normalization),
        missing_information=("synthetic missing field",) if missing else (),
    )
    decision = PolicyEngine().decide(_proposal(action), risk, preference)

    assert TIER_RANK[decision.tier] >= TIER_RANK[capability.minimum_tier]
    assert TIER_RANK[decision.tier] >= TIER_RANK[preference]
    if capability.external or capability.destructive:
        assert TIER_RANK[decision.tier] >= TIER_RANK[AutonomyTier.ASK]
    if not capability.enabled or capability.financial or injection or sensitive:
        assert decision.tier == AutonomyTier.ESCALATE
        assert not behavior_for(decision.tier).execute_automatically


def test_learning_cannot_mutate_the_capability_catalogue() -> None:
    assert isinstance(CAPABILITIES, MappingProxyType)
    snapshot = tuple(CAPABILITIES.items())
    PolicyEngine().decide(
        _proposal(ActionType.ARCHIVE),
        RiskAssessment(),
        AutonomyTier.SILENT,
    )
    assert tuple(CAPABILITIES.items()) == snapshot
