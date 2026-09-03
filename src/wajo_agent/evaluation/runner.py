"""Run frozen datasets through the real planner, scanner, policy, and learner."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

from wajo_agent.domain import (
    TIER_RANK,
    ActionProposal,
    AutonomyTier,
    EmailEnvelope,
    FeedbackType,
    Intent,
    behavior_for,
)
from wajo_agent.evaluation.datasets import load_jsonl, load_manifest, verify_manifest
from wajo_agent.evaluation.metrics import (
    brier_score,
    classification_metrics,
    latency_metrics,
    rate,
)
from wajo_agent.evaluation.results import (
    EvaluationSuiteResult,
    InjectionCaseResult,
    InjectionEvaluation,
    LearningEvaluation,
    PersonaResult,
    PersonaStepResult,
    SemanticCaseResult,
    SemanticEvaluation,
)
from wajo_agent.evaluation.schemas import (
    DatasetSplit,
    InjectionCase,
    LearningPersona,
    SemanticCase,
)
from wajo_agent.learning import (
    ContextualPreferenceLearner,
    InMemoryPreferenceRepository,
    build_preference_context,
)
from wajo_agent.normalization import normalize_email
from wajo_agent.perception import RiskScanner
from wajo_agent.planning import Planner, bind_planner_output, build_planner_request
from wajo_agent.policy import PolicyEngine, get_capability


class SplitRecord(Protocol):
    split: DatasetSplit


class EvaluationRunner:
    """Evaluate components without granting the harness mailbox execution tools."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        planner: Planner,
        planner_name: str,
        split: DatasetSplit | None = None,
        repeats: int = 1,
    ) -> None:
        if not planner_name.strip():
            raise ValueError("planner_name cannot be blank")
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self.dataset_root = dataset_root
        self.planner = planner
        self.planner_name = planner_name.strip()
        self.split = split
        self.repeats = repeats
        self.scanner = RiskScanner()
        self.policy = PolicyEngine()

    def run(self) -> EvaluationSuiteResult:
        manifest = load_manifest(self.dataset_root / "manifest.json")
        verify_manifest(self.dataset_root, manifest)
        semantic = self._select(load_jsonl(self.dataset_root / "semantic.jsonl", SemanticCase))
        injection = self._select(load_jsonl(self.dataset_root / "injection.jsonl", InjectionCase))
        personas = self._select(load_jsonl(self.dataset_root / "personas.jsonl", LearningPersona))
        if not semantic or not injection or not personas:
            raise ValueError("selected split must contain every evaluation dataset family")
        return EvaluationSuiteResult(
            planner=self.planner_name,
            split=self.split,
            repeats=self.repeats,
            dataset_version=manifest.dataset_version,
            semantic=self._semantic(semantic),
            injection=self._injection(injection),
            learning=self._learning(personas),
        )

    def _select[RecordT: SplitRecord](self, records: tuple[RecordT, ...]) -> tuple[RecordT, ...]:
        if self.split is None:
            return records
        return tuple(record for record in records if record.split == self.split)

    def _plan(self, email: EmailEnvelope) -> tuple[ActionProposal | None, float, str | None]:
        normalized, report = normalize_email(email)
        request = build_planner_request(normalized, normalization_changed=report.changed)
        started = perf_counter_ns()
        try:
            proposal = bind_planner_output(request, self.planner.plan(request))
        except Exception as exc:
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000
            return None, elapsed_ms, type(exc).__name__
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        return proposal, elapsed_ms, None

    def _semantic(self, records: tuple[SemanticCase, ...]) -> SemanticEvaluation:
        results: list[SemanticCaseResult] = []
        for repeat in range(1, self.repeats + 1):
            for record in records:
                normalized, report = normalize_email(record.email)
                risk = self.scanner.scan(normalized, normalization_report=report)
                proposal, elapsed_ms, error = self._plan(normalized)
                if proposal is None:
                    tier = AutonomyTier.ESCALATE
                else:
                    context = build_preference_context(normalized, proposal)
                    recommendation = ContextualPreferenceLearner(
                        InMemoryPreferenceRepository()
                    ).recommend(context, get_capability(proposal.action_type))
                    tier = self.policy.decide(proposal, risk, recommendation.tier).tier

                required_labels = len(record.required_sensitive_categories) + len(
                    record.required_injection_signals
                )
                label_hits = len(
                    record.required_sensitive_categories & risk.sensitive_categories
                ) + len(record.required_injection_signals & risk.injection_signals)
                results.append(
                    SemanticCaseResult(
                        case_id=record.case_id,
                        split=record.split,
                        repeat=repeat,
                        schema_valid=proposal is not None,
                        expected_intent=record.expected_intent,
                        actual_intent=proposal.intent if proposal is not None else None,
                        acceptable_actions=record.acceptable_actions,
                        actual_action=proposal.action_type if proposal is not None else None,
                        intent_correct=(
                            proposal is not None and proposal.intent == record.expected_intent
                        ),
                        action_correct=(
                            proposal is not None
                            and proposal.action_type in record.acceptable_actions
                        ),
                        required_risk_labels=required_labels,
                        detected_required_risk_labels=label_hits,
                        minimum_tier=record.minimum_tier,
                        actual_tier=tier,
                        safety_floor_met=TIER_RANK[tier] >= TIER_RANK[record.minimum_tier],
                        latency_ms=elapsed_ms,
                        error=error,
                    )
                )

        result_tuple = tuple(results)
        expected = tuple(result.expected_intent.value for result in result_tuple)
        predicted = tuple(
            result.actual_intent.value if result.actual_intent is not None else "planner_error"
            for result in result_tuple
        )
        return SemanticEvaluation(
            cases=result_tuple,
            schema_valid_rate=rate(
                sum(result.schema_valid for result in result_tuple), len(results)
            ),
            action_accuracy=rate(
                sum(result.action_correct for result in result_tuple), len(results)
            ),
            intent_metrics=classification_metrics(
                expected,
                predicted,
                labels=tuple(intent.value for intent in Intent),
            ),
            required_risk_recall=rate(
                sum(result.detected_required_risk_labels for result in result_tuple),
                sum(result.required_risk_labels for result in result_tuple),
            ),
            safety_floor_violations=sum(not result.safety_floor_met for result in result_tuple),
            planner_latency=latency_metrics(tuple(result.latency_ms for result in result_tuple)),
        )

    def _injection(self, records: tuple[InjectionCase, ...]) -> InjectionEvaluation:
        results: list[InjectionCaseResult] = []
        for repeat in range(1, self.repeats + 1):
            for record in records:
                normalized, report = normalize_email(record.email)
                risk = self.scanner.scan(normalized, normalization_report=report)
                proposal, _, planner_error = self._plan(normalized)
                tier = (
                    AutonomyTier.ESCALATE
                    if proposal is None
                    else self.policy.decide(proposal, risk, AutonomyTier.SILENT).tier
                )
                detected = bool(risk.injection_signals)
                results.append(
                    InjectionCaseResult(
                        case_id=record.case_id,
                        pair_id=record.pair_id,
                        split=record.split,
                        repeat=repeat,
                        is_attack=record.is_attack,
                        expected_signals=record.expected_signals,
                        actual_signals=risk.injection_signals,
                        detected_as_attack=detected,
                        classification_correct=detected == record.is_attack,
                        required_signal_hits=len(record.expected_signals & risk.injection_signals),
                        actual_action=proposal.action_type if proposal is not None else None,
                        actual_tier=tier,
                        safety_floor_met=TIER_RANK[tier] >= TIER_RANK[record.minimum_tier],
                        planner_error=planner_error,
                    )
                )

        result_tuple = tuple(results)
        attacks = tuple(result for result in result_tuple if result.is_attack)
        controls = tuple(result for result in result_tuple if not result.is_attack)
        pairs: dict[tuple[int, str], list[InjectionCaseResult]] = defaultdict(list)
        for result in result_tuple:
            pairs[(result.repeat, result.pair_id)].append(result)
        correct_pairs = sum(
            len(pair) == 2 and all(result.classification_correct for result in pair)
            for pair in pairs.values()
        )
        unsafe_external_effects = sum(
            result.actual_action is not None
            and get_capability(result.actual_action).external
            and behavior_for(result.actual_tier).execute_automatically
            for result in attacks
        )

        return InjectionEvaluation(
            cases=result_tuple,
            attack_detection_recall=rate(
                sum(result.detected_as_attack for result in attacks), len(attacks)
            ),
            benign_false_positive_rate=rate(
                sum(result.detected_as_attack for result in controls), len(controls)
            ),
            required_signal_recall=rate(
                sum(result.required_signal_hits for result in attacks),
                sum(len(result.expected_signals) for result in attacks),
            ),
            correctly_classified_pairs=rate(correct_pairs, len(pairs)),
            attack_policy_bypass_rate=rate(
                sum(result.actual_tier != AutonomyTier.ESCALATE for result in attacks),
                len(attacks),
            ),
            unsafe_external_effects=unsafe_external_effects,
            planner_errors=sum(result.planner_error is not None for result in result_tuple),
        )

    def _learning(self, records: tuple[LearningPersona, ...]) -> LearningEvaluation:
        persona_results = tuple(
            self._persona(record, repeat)
            for repeat in range(1, self.repeats + 1)
            for record in records
        )
        steps = tuple(step for persona in persona_results for step in persona.steps)
        scored_steps = tuple(
            step
            for step in steps
            if step.acceptance_probability is not None and step.accepted is not None
        )
        probabilities = tuple(
            step.acceptance_probability
            for step in scored_steps
            if step.acceptance_probability is not None
        )
        outcomes = tuple(step.accepted for step in scored_steps if step.accepted is not None)
        disagreements = tuple(step.disagreement for step in steps if step.disagreement is not None)
        return LearningEvaluation(
            personas=persona_results,
            ask_rate=rate(sum(step.tier == AutonomyTier.ASK for step in steps), len(steps)),
            notify_rate=rate(sum(step.tier == AutonomyTier.NOTIFY for step in steps), len(steps)),
            silent_rate=rate(sum(step.tier == AutonomyTier.SILENT for step in steps), len(steps)),
            disagreement_rate=rate(sum(disagreements), len(disagreements)),
            no_learning_baseline_ask_rate=rate(
                sum(step.baseline_tier == AutonomyTier.ASK for step in steps), len(steps)
            ),
            brier_score=brier_score(probabilities, outcomes) if probabilities else None,
            planner_errors=sum(step.planner_error is not None for step in steps),
            milestone_failures=sum(
                not persona.expected_milestones_met for persona in persona_results
            ),
            safety_ceiling_violations=sum(
                persona.safety_ceiling_violations for persona in persona_results
            ),
            cross_context_leaks=sum(persona.cross_context_leakage for persona in persona_results),
        )

    def _persona(self, persona: LearningPersona, repeat: int) -> PersonaResult:
        learner = ContextualPreferenceLearner(InMemoryPreferenceRepository())
        steps: list[PersonaStepResult] = []
        last_proposal: ActionProposal | None = None
        for step in persona.steps:
            normalized, report = normalize_email(step.email)
            risk = self.scanner.scan(normalized, normalization_report=report)
            proposal, _, error = self._plan(normalized)
            if proposal is None:
                steps.append(
                    PersonaStepResult(
                        sequence=step.sequence,
                        tier=AutonomyTier.ESCALATE,
                        baseline_tier=AutonomyTier.ESCALATE,
                        acceptance_probability=None,
                        accepted=None,
                        disagreement=None,
                        planner_error=error,
                    )
                )
                continue
            last_proposal = proposal
            context = build_preference_context(normalized, proposal)
            recommendation = learner.recommend(context, get_capability(proposal.action_type))
            tier = self.policy.decide(proposal, risk, recommendation.tier).tier
            cold_recommendation = ContextualPreferenceLearner(
                InMemoryPreferenceRepository()
            ).recommend(context, get_capability(proposal.action_type))
            baseline_tier = self.policy.decide(proposal, risk, cold_recommendation.tier).tier
            feedback = (
                step.feedback_if_ask if tier == AutonomyTier.ASK else step.feedback_if_autonomous
            )
            accepted = feedback in {FeedbackType.APPROVED, FeedbackType.CORRECT}
            steps.append(
                PersonaStepResult(
                    sequence=step.sequence,
                    tier=tier,
                    baseline_tier=baseline_tier,
                    acceptance_probability=recommendation.posterior_mean,
                    accepted=accepted,
                    disagreement=not accepted,
                )
            )
            learner.record(context, feedback)

        step_tuple = tuple(steps)
        first_notify = _first_tier(step_tuple, AutonomyTier.NOTIFY)
        first_silent = _first_tier(step_tuple, AutonomyTier.SILENT)
        ceiling_violations = sum(
            TIER_RANK[step.tier] < TIER_RANK[persona.most_autonomous_allowed] for step in step_tuple
        )
        leakage = self._cross_context_leakage(persona, learner, last_proposal)
        counts = Counter(step.tier for step in step_tuple)
        return PersonaResult(
            persona_id=persona.persona_id,
            split=persona.split,
            repeat=repeat,
            steps=step_tuple,
            tier_counts={tier: counts[tier] for tier in AutonomyTier},
            first_notify_step=first_notify,
            first_silent_step=first_silent,
            expected_milestones_met=(
                first_notify == persona.expected_first_notify_step
                and first_silent == persona.expected_first_silent_step
            ),
            safety_ceiling_violations=ceiling_violations,
            cross_context_leakage=leakage,
            brier_score=(
                brier_score(
                    tuple(
                        step.acceptance_probability
                        for step in step_tuple
                        if step.acceptance_probability is not None
                    ),
                    tuple(step.accepted for step in step_tuple if step.accepted is not None),
                )
                if any(step.acceptance_probability is not None for step in step_tuple)
                else None
            ),
        )

    def _cross_context_leakage(
        self,
        persona: LearningPersona,
        learner: ContextualPreferenceLearner,
        last_proposal: ActionProposal | None,
    ) -> bool:
        if last_proposal is None:
            return False
        original = persona.steps[-1].email
        unseen = original.model_copy(
            update={
                "email_id": f"{original.email_id}_unseen",
                "provider_message_id": f"{original.provider_message_id}-unseen",
                "sender": f"unseen-{persona.persona_id}@synthetic.example",
            }
        )
        proposal, _, _ = self._plan(unseen)
        if proposal is None:
            return False
        context = build_preference_context(unseen, proposal)
        recommendation = learner.recommend(context, get_capability(proposal.action_type))
        return TIER_RANK[recommendation.tier] < TIER_RANK[AutonomyTier.ASK]


def _first_tier(steps: tuple[PersonaStepResult, ...], tier: AutonomyTier) -> int | None:
    return next((step.sequence for step in steps if step.tier == tier), None)
