from __future__ import annotations

from math import comb
from typing import Protocol

from wajo_agent.domain import (
    AutonomyTier,
    FeedbackType,
    PreferenceContext,
    PreferenceState,
)
from wajo_agent.policy.capabilities import CapabilitySpec


class PreferenceRepository(Protocol):
    def get_preference(self, context_key: str) -> PreferenceState: ...

    def save_preference(self, state: PreferenceState) -> None: ...


def beta_tail_probability(alpha: int, beta: int, threshold: float) -> float:
    """Return P(theta > threshold) for integer Beta(alpha, beta)."""
    if alpha < 1 or beta < 1:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    n = alpha + beta - 1
    return sum(
        comb(n, j) * threshold**j * (1.0 - threshold) ** (n - j)
        for j in range(alpha)
    )


class ContextualPreferenceLearner:
    def __init__(self, repository: PreferenceRepository) -> None:
        self.repository = repository

    def state_for(self, context: PreferenceContext) -> PreferenceState:
        return self.repository.get_preference(context.key)

    def recommend(
        self,
        context: PreferenceContext,
        capability: CapabilitySpec,
    ) -> tuple[AutonomyTier, PreferenceState]:
        state = self.state_for(context)

        if context.action_type.value == "no_action":
            return AutonomyTier.SILENT, state
        if state.cooldown_remaining > 0 or state.observations < 4:
            return AutonomyTier.ASK, state

        can_be_silent = (
            capability.enabled
            and capability.reversible
            and not capability.external
            and not capability.destructive
            and not capability.financial
        )
        recent_negative = any(
            item in {FeedbackType.REJECTED, FeedbackType.UNDONE, FeedbackType.EDITED}
            for item in state.recent_feedback[-5:]
        )

        if (
            can_be_silent
            and state.observations >= 12
            and not recent_negative
            and beta_tail_probability(state.alpha, state.beta, 0.90) >= 0.80
        ):
            return AutonomyTier.SILENT, state

        if beta_tail_probability(state.alpha, state.beta, 0.70) >= 0.90:
            return AutonomyTier.NOTIFY, state
        return AutonomyTier.ASK, state

    def record(self, context: PreferenceContext, feedback: FeedbackType) -> PreferenceState:
        state = self.state_for(context)
        alpha = state.alpha
        beta = state.beta
        cooldown = max(0, state.cooldown_remaining - 1)

        if feedback in {FeedbackType.APPROVED, FeedbackType.CORRECT}:
            alpha += 1
        elif feedback == FeedbackType.EDITED:
            beta += 2
            cooldown = max(cooldown, 3)
        elif feedback in {FeedbackType.REJECTED, FeedbackType.UNDONE}:
            beta += 3
            cooldown = max(cooldown, 5)

        updated = PreferenceState(
            context_key=context.key,
            alpha=alpha,
            beta=beta,
            observations=state.observations + 1,
            recent_feedback=(*state.recent_feedback[-4:], feedback),
            cooldown_remaining=cooldown,
        )
        self.repository.save_preference(updated)
        return updated

