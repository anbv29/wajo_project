"""Strict schemas for frozen, synthetic evaluation datasets."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from wajo_agent.domain import (
    ActionType,
    AutonomyTier,
    EmailEnvelope,
    FeedbackType,
    InjectionSignal,
    Intent,
    SensitiveCategory,
)
from wajo_agent.domain.models import StrictModel


class DatasetSplit(StrEnum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class SemanticCase(StrictModel):
    """One semantic-planning and cold-policy reference example."""

    case_id: str = Field(pattern=r"^sem_[a-z0-9_]+$")
    split: DatasetSplit
    email: EmailEnvelope
    expected_intent: Intent
    acceptable_actions: tuple[ActionType, ...] = Field(min_length=1)
    required_sensitive_categories: frozenset[SensitiveCategory] = frozenset()
    required_injection_signals: frozenset[InjectionSignal] = frozenset()
    minimum_tier: AutonomyTier
    annotation_notes: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_semantic_case(self) -> SemanticCase:
        if len(self.acceptable_actions) != len(set(self.acceptable_actions)):
            raise ValueError("acceptable semantic actions must be unique")
        if self.email.source != "fixture":
            raise ValueError("evaluation emails must be synthetic fixtures")
        if self.required_injection_signals and self.minimum_tier != AutonomyTier.ESCALATE:
            raise ValueError("injection cases must require escalation")
        return self


class InjectionCase(StrictModel):
    """One attack or its matched benign control."""

    case_id: str = Field(pattern=r"^inj_[a-z0-9_]+$")
    pair_id: str = Field(pattern=r"^pair_[a-z0-9_]+$")
    split: DatasetSplit
    is_attack: bool
    email: EmailEnvelope
    expected_signals: frozenset[InjectionSignal] = frozenset()
    minimum_tier: AutonomyTier
    annotation_notes: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_attack_label(self) -> InjectionCase:
        if self.email.source != "fixture":
            raise ValueError("injection data must be synthetic")
        if self.is_attack:
            if not self.expected_signals:
                raise ValueError("an attack must name at least one expected signal")
            if self.minimum_tier != AutonomyTier.ESCALATE:
                raise ValueError("an attack must require escalation")
        elif self.expected_signals:
            raise ValueError("a benign control cannot require injection signals")
        return self


class PersonaStep(StrictModel):
    """One chronological interaction with tier-appropriate explicit feedback."""

    sequence: int = Field(ge=1)
    email: EmailEnvelope
    expected_action: ActionType
    feedback_if_ask: FeedbackType
    feedback_if_autonomous: FeedbackType
    annotation_notes: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_feedback_families(self) -> PersonaStep:
        if self.feedback_if_ask not in {
            FeedbackType.APPROVED,
            FeedbackType.EDITED,
            FeedbackType.REJECTED,
        }:
            raise ValueError("ASK feedback must come from the approval workflow")
        if self.feedback_if_autonomous not in {
            FeedbackType.CORRECT,
            FeedbackType.UNDONE,
        }:
            raise ValueError("autonomous feedback must come from execution evidence")
        if self.email.source != "fixture":
            raise ValueError("persona emails must be synthetic fixtures")
        return self


class LearningPersona(StrictModel):
    """A frozen user history evaluated strictly in sequence order."""

    persona_id: str = Field(pattern=r"^persona_[a-z0-9_]+$")
    split: DatasetSplit
    description: str = Field(min_length=1, max_length=1_000)
    most_autonomous_allowed: AutonomyTier
    expected_first_notify_step: int | None = Field(default=None, ge=1)
    expected_first_silent_step: int | None = Field(default=None, ge=1)
    steps: tuple[PersonaStep, ...] = Field(min_length=20, max_length=30)

    @model_validator(mode="after")
    def validate_chronology(self) -> LearningPersona:
        sequences = tuple(step.sequence for step in self.steps)
        if sequences != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("persona steps must be contiguous and chronological")
        provider_ids = tuple(step.email.provider_message_id for step in self.steps)
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("persona provider message IDs must be unique")
        if self.expected_first_notify_step is not None and (
            self.expected_first_notify_step > len(self.steps)
        ):
            raise ValueError("notify milestone is outside the persona")
        if self.expected_first_silent_step is not None and (
            self.expected_first_silent_step > len(self.steps)
        ):
            raise ValueError("silent milestone is outside the persona")
        if (
            self.expected_first_silent_step is not None
            and self.most_autonomous_allowed != AutonomyTier.SILENT
        ):
            raise ValueError("silent milestone requires a silent-eligible persona")
        return self


class FailureScenario(StrictModel):
    """Expected safety behavior for one injected operational failure."""

    scenario_id: str = Field(pattern=r"^fail_[a-z0-9_]+$")
    component: str = Field(min_length=1, max_length=100)
    trigger: str = Field(min_length=1, max_length=1_000)
    expected_outcome: str = Field(min_length=1, max_length=500)
    required_invariants: tuple[str, ...] = Field(min_length=1, max_length=20)
    provider_call_expected: bool
    automatic_retry_allowed: bool = False

    @model_validator(mode="after")
    def reject_unsafe_retry_expectation(self) -> FailureScenario:
        if self.automatic_retry_allowed:
            raise ValueError("frozen safety failures cannot authorize automatic retry")
        return self


class DatasetFileManifest(StrictModel):
    path: str = Field(pattern=r"^[a-z0-9_]+\.jsonl$")
    record_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(StrictModel):
    dataset_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    frozen_on: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    synthetic_only: bool
    files: tuple[DatasetFileManifest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        if not self.synthetic_only:
            raise ValueError("take-home datasets must contain only synthetic data")
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("manifest paths must be unique")
        return self
