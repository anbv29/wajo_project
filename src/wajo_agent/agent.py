"""End-to-end proactive email agent orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import JsonValue, TypeAdapter

from wajo_agent.approvals import ApprovalError, ApprovalService, approval_payload_hash
from wajo_agent.domain import (
    ActionProposal,
    AgentOutcome,
    AutonomyTier,
    Decision,
    EmailEnvelope,
    ExecutionState,
    OutcomeRoute,
    PreferenceRecommendation,
    RiskAssessment,
)
from wajo_agent.execution import ExecutionError, ExecutionService
from wajo_agent.learning import ContextualPreferenceLearner, build_preference_context
from wajo_agent.lifecycle import AgentLifecycle, AgentStage
from wajo_agent.normalization import EmailNormalizationReport, normalize_email
from wajo_agent.perception import RiskScanner
from wajo_agent.planning import Planner, PlannerError, bind_planner_output, build_planner_request
from wajo_agent.policy import PolicyEngine, get_capability
from wajo_agent.storage import SQLiteStore

JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class AgentRun:
    """One finished outcome plus the human-readable lifecycle that produced it."""

    outcome: AgentOutcome
    lifecycle: AgentLifecycle
    normalization: EmailNormalizationReport


class DuplicateEmailError(RuntimeError):
    """A provider redelivered an email already claimed by another agent run."""

    def __init__(self, existing_run_id: str) -> None:
        self.existing_run_id = existing_run_id
        super().__init__(f"email was already claimed by run: {existing_run_id}")


class ProactiveEmailAgent:
    """Coordinates perception, planning, policy, routing, action, and future learning."""

    def __init__(
        self,
        *,
        mailbox_id: str,
        planner: Planner,
        store: SQLiteStore,
        learner: ContextualPreferenceLearner,
        approvals: ApprovalService,
        execution: ExecutionService,
        scanner: RiskScanner | None = None,
        policy: PolicyEngine | None = None,
        internal_domains: Iterable[str] = (),
    ) -> None:
        cleaned_mailbox_id = mailbox_id.strip()
        if not cleaned_mailbox_id:
            raise ValueError("mailbox_id cannot be blank")
        self._mailbox_id = cleaned_mailbox_id
        self._planner = planner
        self._store = store
        self._learner = learner
        self._approvals = approvals
        self._execution = execution
        self._scanner = scanner or RiskScanner()
        self._policy = policy or PolicyEngine()
        self._internal_domains = tuple(internal_domains)

    def process(self, email: EmailEnvelope) -> AgentRun:
        """Run one untrusted email through every mandatory agent stage."""
        lifecycle = AgentLifecycle.start(email.email_id)
        existing_run_id, created = self._store.claim_agent_run(
            run_id=lifecycle.run_id,
            mailbox_id=self._mailbox_id,
            provider_message_id=email.provider_message_id,
            email_id=email.email_id,
            source=email.source,
            occurred_at=lifecycle.entries[-1].recorded_at,
        )
        if not created:
            raise DuplicateEmailError(existing_run_id)

        normalized, report = normalize_email(email)
        lifecycle = lifecycle.advance(AgentStage.NORMALIZE, _normalization_note(report))
        self._event(
            lifecycle.run_id,
            "email.normalized",
            {
                "changed_fields": list(report.changed_fields),
                "truncated_fields": list(report.truncated_fields),
                "zero_width_removed": report.zero_width_removed,
                "control_characters_removed": report.control_characters_removed,
                "visible_content_removed": report.visible_content_removed,
            },
        )

        risk = self._scanner.scan(normalized, normalization_report=report)
        lifecycle = lifecycle.advance(AgentStage.ASSESS_RISK, _risk_note(risk))
        self._event(
            lifecycle.run_id,
            "risk.assessed",
            {
                "injection_signals": sorted(signal.value for signal in risk.injection_signals),
                "sensitive_categories": sorted(
                    category.value for category in risk.sensitive_categories
                ),
                "normalization_flags": sorted(flag.value for flag in risk.normalization_flags),
                "missing_information": list(risk.missing_information),
            },
        )

        request = build_planner_request(normalized, normalization_changed=report.changed)
        try:
            output = self._planner.plan(request)
            proposal = bind_planner_output(request, output)
        except Exception as exc:
            failure_name = (
                type(exc).__name__
                if isinstance(exc, PlannerError)
                else f"unexpected {type(exc).__name__}"
            )
            return self._planner_failure(
                lifecycle=lifecycle,
                normalized=normalized,
                report=report,
                risk=risk,
                failure_name=failure_name,
            )

        lifecycle = lifecycle.advance(
            AgentStage.INTERPRET,
            f"Planner proposed {proposal.action_type.value} as {proposal.intent.value}.",
        )
        self._event(
            lifecycle.run_id,
            "proposal.created",
            {
                "proposal_id": proposal.proposal_id,
                "proposal_version": proposal.version,
                "action_type": proposal.action_type.value,
                "intent": proposal.intent.value,
                "payload_hash": approval_payload_hash(proposal),
            },
        )

        context = build_preference_context(
            normalized,
            proposal,
            internal_domains=self._internal_domains,
        )
        preference = self._learner.recommend(context, get_capability(proposal.action_type))
        decision = self._policy.decide(proposal, risk, preference.tier)
        lifecycle = lifecycle.advance(
            AgentStage.DECIDE_AUTONOMY,
            f"Policy selected {decision.tier.value}; learning supplied only a recommendation.",
        )
        self._record_decision(lifecycle.run_id, decision, preference)

        try:
            outcome = self._route(
                lifecycle.run_id,
                normalized,
                proposal,
                risk,
                preference,
                decision,
            )
        except (ApprovalError, ExecutionError) as exc:
            decision = _runtime_escalation(decision, type(exc).__name__)
            outcome = AgentOutcome(
                run_id=lifecycle.run_id,
                email=normalized,
                proposal=proposal,
                risk=risk,
                preference=preference,
                decision=decision,
                route=OutcomeRoute.ESCALATED,
                user_message="The agent stopped because authorization or execution state changed.",
            )
            self._event(
                lifecycle.run_id,
                "run.runtime_escalated",
                {"reason": type(exc).__name__},
            )

        lifecycle = lifecycle.advance(AgentStage.ACT_OR_WAIT, _route_note(outcome))
        lifecycle = lifecycle.advance(
            AgentStage.LEARN,
            "No implicit learning occurred; explicit user feedback is required.",
        )
        self._complete_run(lifecycle, outcome)
        return AgentRun(outcome=outcome, lifecycle=lifecycle, normalization=report)

    def _route(
        self,
        run_id: str,
        email: EmailEnvelope,
        proposal: ActionProposal,
        risk: RiskAssessment,
        preference: PreferenceRecommendation,
        decision: Decision,
    ) -> AgentOutcome:
        if decision.tier == AutonomyTier.ESCALATE:
            return AgentOutcome(
                run_id=run_id,
                email=email,
                proposal=proposal,
                risk=risk,
                preference=preference,
                decision=decision,
                route=OutcomeRoute.ESCALATED,
                user_message="The agent stopped and escalated this email for manual review.",
            )

        if decision.tier == AutonomyTier.ASK:
            approval = self._approvals.request(proposal, decision)
            return AgentOutcome(
                run_id=run_id,
                email=email,
                proposal=proposal,
                risk=risk,
                preference=preference,
                decision=decision,
                route=OutcomeRoute.AWAITING_APPROVAL,
                approval=approval,
                user_message=f"Approval is required before {proposal.action_type.value}.",
            )

        execution = self._execution.execute(
            mailbox_id=self._mailbox_id,
            provider_message_id=email.provider_message_id,
            proposal=proposal,
            decision=decision,
            risk=risk,
            preference_tier=preference.tier,
        )
        route, user_message = _execution_route(decision.tier, proposal, execution.state)
        return AgentOutcome(
            run_id=run_id,
            email=email,
            proposal=proposal,
            risk=risk,
            preference=preference,
            decision=decision,
            route=route,
            execution=execution,
            user_message=user_message,
        )

    def _planner_failure(
        self,
        *,
        lifecycle: AgentLifecycle,
        normalized: EmailEnvelope,
        report: EmailNormalizationReport,
        risk: RiskAssessment,
        failure_name: str,
    ) -> AgentRun:
        lifecycle = lifecycle.advance(
            AgentStage.INTERPRET,
            f"Planner failed safely: {failure_name}.",
        )
        self._event(lifecycle.run_id, "planner.failed", {"failure": failure_name})
        decision = Decision(
            proposal_id=None,
            proposal_version=None,
            tier=AutonomyTier.ESCALATE,
            capability_floor=AutonomyTier.ESCALATE,
            preference_tier=AutonomyTier.ASK,
            content_floor=AutonomyTier.SILENT,
            reasons=(f"planner did not produce a valid proposal: {failure_name}",),
        )
        lifecycle = lifecycle.advance(
            AgentStage.DECIDE_AUTONOMY,
            "Missing valid planner output forced ESCALATE.",
        )
        self._record_decision(lifecycle.run_id, decision, None)
        outcome = AgentOutcome(
            run_id=lifecycle.run_id,
            email=normalized,
            proposal=None,
            risk=risk,
            preference=None,
            decision=decision,
            route=OutcomeRoute.ESCALATED,
            user_message=(
                "The planner could not safely interpret this email; manual review is required."
            ),
        )
        lifecycle = lifecycle.advance(
            AgentStage.ACT_OR_WAIT,
            "No action was attempted; the email was escalated.",
        )
        lifecycle = lifecycle.advance(
            AgentStage.LEARN,
            "Planner failure is not user preference feedback, so nothing was learned.",
        )
        self._complete_run(lifecycle, outcome)
        return AgentRun(outcome=outcome, lifecycle=lifecycle, normalization=report)

    def _record_decision(
        self,
        run_id: str,
        decision: Decision,
        preference: PreferenceRecommendation | None,
    ) -> None:
        self._event(
            run_id,
            "decision.created",
            {
                "decision_id": decision.decision_id,
                "tier": decision.tier.value,
                "capability_floor": decision.capability_floor.value,
                "content_floor": decision.content_floor.value,
                "preference_tier": decision.preference_tier.value,
                "preference_context": (preference.context_key if preference is not None else None),
                "reasons": list(decision.reasons),
            },
        )

    def _event(self, run_id: str, event_type: str, payload: object) -> None:
        self._store.append_event(
            stream_id=f"run:{run_id}",
            event_type=event_type,
            payload=JSON_OBJECT_ADAPTER.validate_python(payload),
        )

    def _complete_run(self, lifecycle: AgentLifecycle, outcome: AgentOutcome) -> None:
        self._store.complete_agent_run_with_outcome(
            outcome,
            occurred_at=lifecycle.entries[-1].recorded_at,
        )


def _execution_route(
    tier: AutonomyTier,
    proposal: ActionProposal,
    state: ExecutionState,
) -> tuple[OutcomeRoute, str | None]:
    if state == ExecutionState.FAILED_SAFE:
        return (
            OutcomeRoute.EXECUTION_FAILED,
            f"The proposed {proposal.action_type.value} did not run; no effect occurred.",
        )
    if state == ExecutionState.UNKNOWN:
        return (
            OutcomeRoute.EXECUTION_UNKNOWN,
            f"The result of {proposal.action_type.value} is unknown and needs review.",
        )
    if tier == AutonomyTier.NOTIFY:
        return (
            OutcomeRoute.EXECUTED_AND_NOTIFY,
            f"The agent completed {proposal.action_type.value}.",
        )
    return OutcomeRoute.EXECUTED_SILENTLY, None


def _runtime_escalation(decision: Decision, failure_name: str) -> Decision:
    return decision.model_copy(
        update={
            "tier": AutonomyTier.ESCALATE,
            "reasons": (*decision.reasons, f"runtime safety stop: {failure_name}"),
        }
    )


def _normalization_note(report: EmailNormalizationReport) -> str:
    if report.changed:
        fields = ", ".join(report.changed_fields)
        return f"Email was converted to inert normalized text; changed: {fields}."
    return "Email was already in normalized inert form."


def _risk_note(risk: RiskAssessment) -> str:
    signal_count = len(risk.injection_signals)
    sensitive_count = len(risk.sensitive_categories)
    return (
        f"Risk scan found {signal_count} injection signal(s) and "
        f"{sensitive_count} sensitive category finding(s)."
    )


def _route_note(outcome: AgentOutcome) -> str:
    return f"Run routed to {outcome.route.value}."
