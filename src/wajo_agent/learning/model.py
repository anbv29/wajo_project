"""Cautious contextual learning from explicit user feedback."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, fsum
from typing import Protocol

from wajo_agent.domain import (
    ActionType,
    AutonomyTier,
    FeedbackType,
    PreferenceContext,
    PreferenceRecommendation,
    PreferenceState,
)
from wajo_agent.policy.capabilities import CapabilitySpec


class PreferenceDataError(RuntimeError):
    """Raised when stored learning state belongs to the wrong context."""


class PreferenceRepository(Protocol):
    def get_preference(self, context_key: str) -> PreferenceState: ...

    def save_preference(self, state: PreferenceState) -> None: ...


class InMemoryPreferenceRepository:
    """Small deterministic repository used until the SQLite adapter is added."""

    def __init__(self) -> None:
        self._states: dict[str, PreferenceState] = {}

    def get_preference(self, context_key: str) -> PreferenceState:
        return self._states.get(context_key, PreferenceState(context_key=context_key))

    def save_preference(self, state: PreferenceState) -> None:
        self._states[state.context_key] = state


@dataclass(frozen=True, slots=True)
class LearningThresholds:
    """Code-owned promotion and negative-feedback settings."""

    notify_min_observations: int = 4
    notify_acceptance_threshold: float = 0.70
    notify_required_probability: float = 0.90
    silent_min_observations: int = 12
    silent_acceptance_threshold: float = 0.90
    silent_required_probability: float = 0.80
    edited_beta_weight: int = 2
    negative_beta_weight: int = 3
    edited_cooldown: int = 3
    negative_cooldown: int = 5

    def __post_init__(self) -> None:
        if self.notify_min_observations < 1:
            raise ValueError("notify_min_observations must be positive")
        if self.silent_min_observations < self.notify_min_observations:
            raise ValueError("silent promotion must require at least as much evidence as notify")
        probability_values = (
            self.notify_acceptance_threshold,
            self.notify_required_probability,
            self.silent_acceptance_threshold,
            self.silent_required_probability,
        )
        if any(not 0.0 < value < 1.0 for value in probability_values):
            raise ValueError("learning probability settings must be between zero and one")
        if self.silent_acceptance_threshold < self.notify_acceptance_threshold:
            raise ValueError("silent acceptance threshold cannot be weaker than notify")
        if (
            min(
                self.edited_beta_weight,
                self.negative_beta_weight,
                self.edited_cooldown,
                self.negative_cooldown,
            )
            < 1
        ):
            raise ValueError("feedback weights and cooldowns must be positive")


def beta_tail_probability(alpha: int, beta: int, threshold: float) -> float:
    """Return P(theta > threshold) for integer Beta(alpha, beta)."""
    if alpha < 1 or beta < 1:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if threshold == 0.0:
        return 1.0
    if threshold == 1.0:
        return 0.0

    n = alpha + beta - 1
    probability = fsum(
        comb(n, successes) * threshold**successes * (1.0 - threshold) ** (n - successes)
        for successes in range(alpha)
    )
    return min(1.0, max(0.0, probability))


class ContextualPreferenceLearner:
    """Learn narrow user preferences without granting action authority."""

    def __init__(
        self,
        repository: PreferenceRepository,
        *,
        thresholds: LearningThresholds | None = None,
    ) -> None:
        self.repository = repository
        self.thresholds = thresholds or LearningThresholds()

    def state_for(self, context: PreferenceContext) -> PreferenceState:
        state = self.repository.get_preference(context.key)
        if state.context_key != context.key:
            raise PreferenceDataError("repository returned state for the wrong context")
        return state

    def recommend(
        self,
        context: PreferenceContext,
        capability: CapabilitySpec,
    ) -> PreferenceRecommendation:
        state = self.state_for(context)
        thresholds = self.thresholds
        posterior_mean = state.alpha / (state.alpha + state.beta)
        notify_probability = beta_tail_probability(
            state.alpha,
            state.beta,
            thresholds.notify_acceptance_threshold,
        )
        silent_probability = beta_tail_probability(
            state.alpha,
            state.beta,
            thresholds.silent_acceptance_threshold,
        )
        reasons: list[str] = []

        if context.action_type == ActionType.NO_ACTION:
            tier = AutonomyTier.SILENT
            reasons.append("no_action has no side effect")
        elif state.cooldown_remaining > 0:
            tier = AutonomyTier.ASK
            reasons.append(f"negative-feedback cooldown: {state.cooldown_remaining} observations")
        elif state.observations < thresholds.notify_min_observations:
            tier = AutonomyTier.ASK
            reasons.append(
                f"cold start: {state.observations}/{thresholds.notify_min_observations} "
                "observations"
            )
        else:
            recent_negative = any(
                feedback in {FeedbackType.EDITED, FeedbackType.REJECTED, FeedbackType.UNDONE}
                for feedback in state.recent_feedback
            )
            silent_ready = (
                capability.silent_eligible
                and state.observations >= thresholds.silent_min_observations
                and not recent_negative
                and silent_probability >= thresholds.silent_required_probability
            )

            if silent_ready:
                tier = AutonomyTier.SILENT
                reasons.append("strong evidence supports silent handling in this exact context")
            elif notify_probability >= thresholds.notify_required_probability:
                tier = AutonomyTier.NOTIFY
                reasons.append("evidence supports acting with a user notification")
                if not capability.silent_eligible:
                    reasons.append("this capability is never eligible for learned silent handling")
                elif recent_negative:
                    reasons.append("recent negative feedback blocks silent promotion")
                elif state.observations < thresholds.silent_min_observations:
                    reasons.append("more observations are required before silent promotion")
                else:
                    reasons.append("silent-confidence threshold has not been reached")
            else:
                tier = AutonomyTier.ASK
                reasons.append("preference confidence is not high enough for autonomous handling")

        return PreferenceRecommendation(
            context_key=context.key,
            tier=tier,
            alpha=state.alpha,
            beta=state.beta,
            observations=state.observations,
            posterior_mean=posterior_mean,
            notify_probability=notify_probability,
            silent_probability=silent_probability,
            reasons=tuple(reasons),
        )

    def record(self, context: PreferenceContext, feedback: FeedbackType) -> PreferenceState:
        """Apply one explicit feedback observation to this exact context."""
        state = self.state_for(context)
        thresholds = self.thresholds
        alpha = state.alpha
        beta = state.beta
        cooldown = max(0, state.cooldown_remaining - 1)

        if feedback in {FeedbackType.APPROVED, FeedbackType.CORRECT}:
            alpha += 1
        elif feedback == FeedbackType.EDITED:
            beta += thresholds.edited_beta_weight
            cooldown = max(cooldown, thresholds.edited_cooldown)
        elif feedback in {FeedbackType.REJECTED, FeedbackType.UNDONE}:
            beta += thresholds.negative_beta_weight
            cooldown = max(cooldown, thresholds.negative_cooldown)

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
