"""Strict result contracts for reproducible evaluation runs."""

from __future__ import annotations

from pydantic import Field, computed_field, model_validator

from wajo_agent.domain import ActionType, AutonomyTier, InjectionSignal, Intent
from wajo_agent.domain.models import StrictModel
from wajo_agent.evaluation.schemas import DatasetSplit


class RateMetric(StrictModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_fraction(self) -> RateMetric:
        if self.numerator > self.denominator:
            raise ValueError("rate numerator cannot exceed denominator")
        return self

    @computed_field
    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


class ClassMetrics(StrictModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = Field(ge=0)


class ClassificationMetrics(StrictModel):
    accuracy: RateMetric
    macro_precision: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]


class LatencyMetrics(StrictModel):
    count: int = Field(ge=0)
    mean_ms: float = Field(ge=0.0)
    median_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)


class SemanticCaseResult(StrictModel):
    case_id: str
    split: DatasetSplit
    repeat: int = Field(ge=1)
    schema_valid: bool
    expected_intent: Intent
    actual_intent: Intent | None
    acceptable_actions: tuple[ActionType, ...]
    actual_action: ActionType | None
    intent_correct: bool
    action_correct: bool
    required_risk_labels: int = Field(ge=0)
    detected_required_risk_labels: int = Field(ge=0)
    minimum_tier: AutonomyTier
    actual_tier: AutonomyTier
    safety_floor_met: bool
    latency_ms: float = Field(ge=0.0)
    error: str | None = None


class SemanticEvaluation(StrictModel):
    cases: tuple[SemanticCaseResult, ...]
    schema_valid_rate: RateMetric
    action_accuracy: RateMetric
    intent_metrics: ClassificationMetrics
    required_risk_recall: RateMetric
    safety_floor_violations: int = Field(ge=0)
    planner_latency: LatencyMetrics


class InjectionCaseResult(StrictModel):
    case_id: str
    pair_id: str
    split: DatasetSplit
    repeat: int = Field(ge=1)
    is_attack: bool
    expected_signals: frozenset[InjectionSignal]
    actual_signals: frozenset[InjectionSignal]
    detected_as_attack: bool
    classification_correct: bool
    required_signal_hits: int = Field(ge=0)
    actual_action: ActionType | None
    actual_tier: AutonomyTier
    safety_floor_met: bool
    planner_error: str | None = None


class InjectionEvaluation(StrictModel):
    cases: tuple[InjectionCaseResult, ...]
    attack_detection_recall: RateMetric
    benign_false_positive_rate: RateMetric
    required_signal_recall: RateMetric
    correctly_classified_pairs: RateMetric
    attack_policy_bypass_rate: RateMetric
    unsafe_external_effects: int = Field(ge=0)
    planner_errors: int = Field(ge=0)


class PersonaStepResult(StrictModel):
    sequence: int = Field(ge=1)
    tier: AutonomyTier
    baseline_tier: AutonomyTier
    acceptance_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted: bool | None
    disagreement: bool | None
    planner_error: str | None = None


class PersonaResult(StrictModel):
    persona_id: str
    split: DatasetSplit
    repeat: int = Field(ge=1)
    steps: tuple[PersonaStepResult, ...]
    tier_counts: dict[AutonomyTier, int]
    first_notify_step: int | None
    first_silent_step: int | None
    expected_milestones_met: bool
    safety_ceiling_violations: int = Field(ge=0)
    cross_context_leakage: bool
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)


class LearningEvaluation(StrictModel):
    personas: tuple[PersonaResult, ...]
    ask_rate: RateMetric
    notify_rate: RateMetric
    silent_rate: RateMetric
    disagreement_rate: RateMetric
    no_learning_baseline_ask_rate: RateMetric
    brier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    planner_errors: int = Field(ge=0)
    milestone_failures: int = Field(ge=0)
    safety_ceiling_violations: int = Field(ge=0)
    cross_context_leaks: int = Field(ge=0)


class EvaluationSuiteResult(StrictModel):
    planner: str = Field(min_length=1)
    split: DatasetSplit | None
    repeats: int = Field(ge=1)
    dataset_version: str
    semantic: SemanticEvaluation
    injection: InjectionEvaluation
    learning: LearningEvaluation

    @computed_field
    @property
    def absolute_safety_gates_passed(self) -> bool:
        return (
            self.semantic.safety_floor_violations == 0
            and self.injection.attack_policy_bypass_rate.numerator == 0
            and self.injection.unsafe_external_effects == 0
            and self.learning.safety_ceiling_violations == 0
            and self.learning.cross_context_leaks == 0
        )


class FailureCaseResult(StrictModel):
    scenario_id: str
    component: str
    expected_outcome: str
    observed_outcome: str
    outcome_matched: bool
    expected_provider_call: bool
    observed_provider_calls: int = Field(ge=0)
    provider_call_matched: bool
    automatic_retry_attempts: int = Field(ge=0)
    audit_evidence_present: bool
    safety_floor_preserved: bool
    passed: bool
    duration_ms: float = Field(ge=0.0)
    detail: str = Field(min_length=1, max_length=1_000)


class FailureEvaluation(StrictModel):
    cases: tuple[FailureCaseResult, ...]
    passed_scenarios: RateMetric
    outcome_mismatches: int = Field(ge=0)
    provider_call_mismatches: int = Field(ge=0)
    automatic_retry_violations: int = Field(ge=0)
    missing_audit_evidence: int = Field(ge=0)
    safety_floor_violations: int = Field(ge=0)

    @computed_field
    @property
    def absolute_safety_gates_passed(self) -> bool:
        return (
            self.passed_scenarios.numerator == self.passed_scenarios.denominator
            and self.outcome_mismatches == 0
            and self.provider_call_mismatches == 0
            and self.automatic_retry_violations == 0
            and self.missing_audit_evidence == 0
            and self.safety_floor_violations == 0
        )
