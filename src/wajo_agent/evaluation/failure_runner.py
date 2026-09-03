"""Active, deterministic fault injection against the real safety boundaries."""

# ruff: noqa: SIM117

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

from pydantic import JsonValue

from wajo_agent import DuplicateEmailError, ProactiveEmailAgent
from wajo_agent.approvals import (
    ApprovalBindingError,
    ApprovalExpiredError,
    ApprovalService,
    ApprovalStateError,
)
from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    ApprovalStatus,
    AuditEvent,
    AutonomyTier,
    Decision,
    EmailEnvelope,
    ExecutionCommand,
    ExecutionResult,
    ExecutionState,
    FeedbackType,
    Intent,
    LabelPayload,
    MessagePayload,
    PlannerOutput,
    PlannerRequest,
    ReplyPayload,
    RiskAssessment,
)
from wajo_agent.evaluation.datasets import load_jsonl, load_manifest, verify_manifest
from wajo_agent.evaluation.metrics import rate
from wajo_agent.evaluation.results import FailureCaseResult, FailureEvaluation
from wajo_agent.evaluation.schemas import FailureScenario
from wajo_agent.execution import (
    ExecutionInProgressError,
    ExecutionService,
    MailboxExecutor,
    MockMailboxExecutor,
    MockOutcome,
    effect_idempotency_key,
)
from wajo_agent.feedback import FeedbackEvidenceError, FeedbackService
from wajo_agent.gmail import (
    EnvironmentAccessTokenProvider,
    GmailAdapterConfig,
    GmailAmbiguousError,
    GmailConfigurationError,
    GmailMailboxExecutor,
    GmailResponse,
)
from wajo_agent.gmail.contracts import GmailMethod
from wajo_agent.learning import ContextualPreferenceLearner, build_preference_context
from wajo_agent.planning import (
    OfflinePlanner,
    Planner,
    PlannerContractError,
    PlannerUnavailableError,
)
from wajo_agent.policy import PolicyEngine
from wajo_agent.storage import (
    SCHEMA_VERSION,
    SchemaVersionError,
    SQLiteStore,
    StorageError,
)


@dataclass(frozen=True, slots=True)
class ScenarioObservation:
    observed_outcome: str
    observed_provider_calls: int
    automatic_retry_attempts: int
    audit_evidence_present: bool
    safety_floor_preserved: bool
    detail: str


class FixedClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


class TimeoutPlanner:
    def plan(self, request: PlannerRequest) -> PlannerOutput:
        del request
        raise PlannerUnavailableError("injected planner timeout")


class InvalidSchemaPlanner:
    def plan(self, request: PlannerRequest) -> PlannerOutput:
        del request
        raise PlannerContractError("injected malformed structured output")


class DisallowedActionPlanner:
    def plan(self, request: PlannerRequest) -> PlannerOutput:
        del request
        return PlannerOutput(
            action_type=ActionType.PAYMENT,
            intent=Intent.FINANCIAL,
            summary="Injected disabled payment proposal",
            payload=ReplyPayload(kind=ActionType.PAYMENT, body="Do not execute"),
        )


class CountingPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self._planner = OfflinePlanner()

    def plan(self, request: PlannerRequest) -> PlannerOutput:
        self.calls += 1
        return self._planner.plan(request)


class ClaimFailingStore(SQLiteStore):
    def claim_agent_run(
        self,
        *,
        run_id: str,
        mailbox_id: str,
        provider_message_id: str,
        email_id: str,
        source: str,
        occurred_at: datetime,
    ) -> tuple[str, bool]:
        del run_id, mailbox_id, provider_message_id, email_id, source, occurred_at
        raise StorageError("injected run-claim failure")


class FeedbackAuditFailingStore(SQLiteStore):
    fail_feedback_audit = False

    def __enter__(self) -> FeedbackAuditFailingStore:
        return self

    def _insert_event(
        self,
        *,
        stream_id: str,
        event_type: str,
        payload: dict[str, JsonValue],
        event_version: int,
        event_id: str | None,
        occurred_at: datetime | None,
    ) -> AuditEvent:
        if self.fail_feedback_audit and event_type == "feedback.recorded":
            raise StorageError("injected feedback audit failure")
        return super()._insert_event(
            stream_id=stream_id,
            event_type=event_type,
            payload=payload,
            event_version=event_version,
            event_id=event_id,
            occurred_at=occurred_at,
        )


class WrongIdentityExecutor:
    def __init__(self) -> None:
        self.call_count = 0

    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        self.call_count += 1
        return ExecutionResult(
            execution_id=command.execution_id,
            command_id="wrong-command",
            idempotency_key=command.idempotency_key,
            proposal_id=command.proposal_id,
            state=ExecutionState.SUCCEEDED,
            detail="Injected mismatched command identity",
        )


class CrashAfterClaimExecutor:
    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        del command
        raise KeyboardInterrupt("injected crash after durable claim")


class FakeGmailTransport:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def request(
        self,
        method: GmailMethod,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, JsonValue] | None = None,
    ) -> GmailResponse:
        del method, path, query, body
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return GmailResponse(status_code=200, body={"id": "gmail-failure-message"})


class FailureInjectionRunner:
    """Execute every frozen failure case and retain its direct observations."""

    def __init__(self, *, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self._handlers: dict[str, Callable[[], ScenarioObservation]] = {
            "fail_planner_timeout": lambda: self._planner_failure(TimeoutPlanner()),
            "fail_planner_invalid_schema": lambda: self._planner_failure(InvalidSchemaPlanner()),
            "fail_planner_disallowed_action": lambda: self._planner_failure(
                DisallowedActionPlanner()
            ),
            "fail_duplicate_webhook": self._duplicate_webhook,
            "fail_storage_claim_failure": self._storage_claim_failure,
            "fail_approval_expired": self._approval_expired,
            "fail_approval_replay": self._approval_replay,
            "fail_approval_payload_tamper": self._approval_payload_tamper,
            "fail_approval_wrong_version": self._approval_wrong_version,
            "fail_approval_stale_edit": self._approval_stale_edit,
            "fail_executor_unavailable": self._executor_unavailable,
            "fail_executor_timeout": self._executor_timeout,
            "fail_executor_bad_identity": self._executor_bad_identity,
            "fail_crash_after_claim": self._crash_after_claim,
            "fail_duplicate_execution": self._duplicate_execution,
            "fail_feedback_duplicate": self._feedback_duplicate,
            "fail_feedback_missing_evidence": self._feedback_missing_evidence,
            "fail_feedback_write_failure": self._feedback_write_failure,
            "fail_newer_schema": self._newer_schema,
            "fail_audit_update": self._audit_update,
            "fail_gmail_missing_token": self._gmail_missing_token,
            "fail_gmail_post_timeout": self._gmail_post_timeout,
            "fail_gmail_recipient_blocked": self._gmail_recipient_blocked,
            "fail_gmail_label_missing": self._gmail_label_missing,
        }

    def run(self) -> FailureEvaluation:
        manifest = load_manifest(self.dataset_root / "manifest.json")
        verify_manifest(self.dataset_root, manifest)
        scenarios = load_jsonl(self.dataset_root / "failures.jsonl", FailureScenario)
        if {scenario.scenario_id for scenario in scenarios} != set(self._handlers):
            raise ValueError("failure scenario registry does not match the frozen dataset")
        results = tuple(self._run_scenario(scenario) for scenario in scenarios)
        return FailureEvaluation(
            cases=results,
            passed_scenarios=rate(sum(result.passed for result in results), len(results)),
            outcome_mismatches=sum(not result.outcome_matched for result in results),
            provider_call_mismatches=sum(not result.provider_call_matched for result in results),
            automatic_retry_violations=sum(result.automatic_retry_attempts for result in results),
            missing_audit_evidence=sum(not result.audit_evidence_present for result in results),
            safety_floor_violations=sum(not result.safety_floor_preserved for result in results),
        )

    def _run_scenario(self, scenario: FailureScenario) -> FailureCaseResult:
        started = perf_counter_ns()
        try:
            observation = self._handlers[scenario.scenario_id]()
        except Exception as exc:
            observation = ScenarioObservation(
                observed_outcome=f"UNEXPECTED_{type(exc).__name__}",
                observed_provider_calls=0,
                automatic_retry_attempts=0,
                audit_evidence_present=False,
                safety_floor_preserved=False,
                detail="Failure injector raised an unexpected exception.",
            )
        duration_ms = (perf_counter_ns() - started) / 1_000_000
        outcome_matched = observation.observed_outcome == scenario.expected_outcome
        provider_matched = (observation.observed_provider_calls > 0) == (
            scenario.provider_call_expected
        )
        passed = (
            outcome_matched
            and provider_matched
            and observation.automatic_retry_attempts == 0
            and observation.audit_evidence_present
            and observation.safety_floor_preserved
        )
        return FailureCaseResult(
            scenario_id=scenario.scenario_id,
            component=scenario.component,
            expected_outcome=scenario.expected_outcome,
            observed_outcome=observation.observed_outcome,
            outcome_matched=outcome_matched,
            expected_provider_call=scenario.provider_call_expected,
            observed_provider_calls=observation.observed_provider_calls,
            provider_call_matched=provider_matched,
            automatic_retry_attempts=observation.automatic_retry_attempts,
            audit_evidence_present=observation.audit_evidence_present,
            safety_floor_preserved=observation.safety_floor_preserved,
            passed=passed,
            duration_ms=duration_ms,
            detail=observation.detail,
        )

    def _planner_failure(self, planner: Planner) -> ScenarioObservation:
        with _database_path("planner") as path:
            with SQLiteStore(path) as store:
                executor = MockMailboxExecutor()
                agent = _agent(store, planner, executor)
                run = agent.process(_newsletter("planner"))
                events = store.read_stream(f"run:{run.outcome.run_id}")
                safe = (
                    run.outcome.decision.tier == AutonomyTier.ESCALATE
                    and run.outcome.execution is None
                )
                return _observation(
                    "ESCALATE",
                    audit=any(event.event_type == "planner.failed" for event in events),
                    safety=safe,
                    detail="Planner fault produced an audited escalation with no execution.",
                )

    def _duplicate_webhook(self) -> ScenarioObservation:
        with _database_path("duplicate") as path:
            with SQLiteStore(path) as store:
                executor = MockMailboxExecutor()
                agent = _agent(store, OfflinePlanner(), executor)
                email = _newsletter("duplicate")
                first = agent.process(email)
                try:
                    agent.process(email)
                except DuplicateEmailError as exc:
                    matched = exc.existing_run_id == first.outcome.run_id
                else:
                    matched = False
                return _observation(
                    "Return original run identity" if matched else "Duplicate processed",
                    audit=bool(store.read_stream(f"run:{first.outcome.run_id}")),
                    safety=matched and executor.call_count == 0,
                    detail="The unique mailbox/message claim rejected the second delivery.",
                )

    def _storage_claim_failure(self) -> ScenarioObservation:
        with _database_path("claim") as path:
            with ClaimFailingStore(path) as store:
                planner = CountingPlanner()
                executor = MockMailboxExecutor()
                agent = _agent(store, planner, executor)
                try:
                    agent.process(_newsletter("claim"))
                except StorageError:
                    blocked = planner.calls == 0 and executor.call_count == 0
                else:
                    blocked = False
                return _observation(
                    "No planning or execution" if blocked else "Work continued after claim failure",
                    audit=True,
                    safety=blocked,
                    detail=(
                        "The injected claim failure stopped the pipeline before AI or adapter use."
                    ),
                )

    def _approval_expired(self) -> ScenarioObservation:
        clock = FixedClock()
        with _database_path("expired") as path:
            with SQLiteStore(path) as store:
                proposal = _reply("expired")
                service = ApprovalService(store, clock=clock)
                record = service.request(proposal, _decision(proposal), ttl=timedelta(minutes=1))
                clock.advance(timedelta(minutes=2))
                try:
                    service.grant(record.approval_id, proposal, actor="user@synthetic.example")
                except ApprovalExpiredError:
                    rejected = True
                else:
                    rejected = False
                events = store.read_stream(f"approval:{record.approval_id}")
                return _observation(
                    "Reject grant" if rejected else "Grant accepted",
                    audit=any(event.event_type == "approval.expired" for event in events),
                    safety=rejected
                    and store.get_approval(record.approval_id).status == ApprovalStatus.EXPIRED,
                    detail="The clock advanced beyond TTL and the grant was rejected durably.",
                )

    def _approval_replay(self) -> ScenarioObservation:
        with _database_path("replay") as path:
            with SQLiteStore(path) as store:
                proposal = _reply("replay")
                service = ApprovalService(store)
                record = service.request(proposal, _decision(proposal))
                service.grant(record.approval_id, proposal, actor="user@synthetic.example")
                service.consume(record.approval_id, proposal)
                try:
                    service.consume(record.approval_id, proposal)
                except ApprovalStateError:
                    rejected = True
                else:
                    rejected = False
                events = store.read_stream(f"approval:{record.approval_id}")
                return _observation(
                    "Reject replay" if rejected else "Replay accepted",
                    audit=len(events) == 3,
                    safety=rejected
                    and store.get_approval(record.approval_id).status == ApprovalStatus.CONSUMED,
                    detail="A consumed approval could not authorize a second transition.",
                )

    def _approval_payload_tamper(self) -> ScenarioObservation:
        with _database_path("tamper") as path:
            with SQLiteStore(path) as store:
                proposal = _reply("tamper")
                service = ApprovalService(store)
                record = service.request(proposal, _decision(proposal))
                tampered = proposal.model_copy(
                    update={"payload": proposal.payload.model_copy(update={"body": "Changed body"})}
                )
                try:
                    service.grant(record.approval_id, tampered, actor="user@synthetic.example")
                except ApprovalBindingError:
                    rejected = True
                else:
                    rejected = False
                return _observation(
                    "Reject binding" if rejected else "Tampered payload accepted",
                    audit=bool(store.read_stream(f"approval:{record.approval_id}")),
                    safety=rejected
                    and store.get_approval(record.approval_id).status == ApprovalStatus.PENDING,
                    detail="Changing approved body bytes invalidated the approval hash binding.",
                )

    def _approval_wrong_version(self) -> ScenarioObservation:
        with _database_path("version") as path:
            with SQLiteStore(path) as store:
                proposal = _reply("version")
                service = ApprovalService(store)
                record = service.request(proposal, _decision(proposal))
                revised = proposal.model_copy(update={"version": 2})
                try:
                    service.grant(record.approval_id, revised, actor="user@synthetic.example")
                except ApprovalBindingError:
                    rejected = True
                else:
                    rejected = False
                return _observation(
                    "Reject binding" if rejected else "Wrong version accepted",
                    audit=bool(store.read_stream(f"approval:{record.approval_id}")),
                    safety=rejected,
                    detail="Approval version 1 could not authorize proposal version 2.",
                )

    def _approval_stale_edit(self) -> ScenarioObservation:
        with _database_path("stale") as path:
            with SQLiteStore(path) as first_store, SQLiteStore(path) as second_store:
                proposal = _reply("stale")
                first = ApprovalService(first_store)
                second = ApprovalService(second_store)
                record = first.request(proposal, _decision(proposal))
                revised_a = _revised_reply(proposal, "First edit")
                revised_b = _revised_reply(proposal, "Second edit")
                first.replace_for_edit(
                    record.approval_id,
                    proposal,
                    revised_a,
                    _decision(revised_a),
                    actor="first@synthetic.example",
                )
                try:
                    second.replace_for_edit(
                        record.approval_id,
                        proposal,
                        revised_b,
                        _decision(revised_b),
                        actor="second@synthetic.example",
                    )
                except ApprovalStateError:
                    one_winner = True
                else:
                    one_winner = False
                events = first_store.read_stream(f"approval:{record.approval_id}")
                return _observation(
                    "One atomic winner" if one_winner else "Two edits committed",
                    audit=any(event.event_type == "approval.invalidated" for event in events),
                    safety=one_winner,
                    detail="The compare-and-set edit transition permitted one winner.",
                )

    def _executor_unavailable(self) -> ScenarioObservation:
        executor = MockMailboxExecutor((MockOutcome.RAISE_UNAVAILABLE,))
        result, audited = _execute_archive(executor, "unavailable")
        return _observation(
            result.state.value.upper(),
            provider_calls=0,
            audit=audited,
            safety=result.state == ExecutionState.FAILED_SAFE and executor.call_count == 1,
            detail="A certified pre-effect adapter failure became FAILED_SAFE.",
        )

    def _executor_timeout(self) -> ScenarioObservation:
        executor = MockMailboxExecutor((MockOutcome.RAISE_UNKNOWN,))
        with _database_path("timeout") as path:
            with SQLiteStore(path) as store:
                proposal = _archive("timeout")
                service = ExecutionService(store, executor)
                result = _execute(service, proposal)
                before = executor.call_count
                duplicate = _execute(service, proposal)
                retry_attempts = executor.call_count - before
                audited = bool(store.read_stream(f"execution:{result.execution_id}"))
                return _observation(
                    result.state.value.upper(),
                    provider_calls=1,
                    retries=retry_attempts,
                    audit=audited,
                    safety=(
                        result.state == ExecutionState.UNKNOWN
                        and duplicate.execution_id == result.execution_id
                    ),
                    detail="An ambiguous timeout stayed UNKNOWN and was not retried.",
                )

    def _executor_bad_identity(self) -> ScenarioObservation:
        executor = WrongIdentityExecutor()
        result, audited = _execute_archive(executor, "bad_identity")
        return _observation(
            result.state.value.upper(),
            provider_calls=executor.call_count,
            audit=audited,
            safety=result.state == ExecutionState.UNKNOWN,
            detail="A mismatched provider result identity was treated as UNKNOWN.",
        )

    def _crash_after_claim(self) -> ScenarioObservation:
        with _database_path("crash") as path:
            with SQLiteStore(path) as store:
                proposal = _archive("crash")
                try:
                    _execute(ExecutionService(store, CrashAfterClaimExecutor()), proposal)
                except KeyboardInterrupt:
                    crashed = True
                else:
                    crashed = False
                key = effect_idempotency_key(
                    "failure-mailbox", _provider_message_id(proposal), proposal
                )
                record = store.get_execution_by_idempotency_key(key)
                recovery = MockMailboxExecutor()
                try:
                    _execute(ExecutionService(store, recovery), proposal)
                except ExecutionInProgressError:
                    blocked = True
                else:
                    blocked = False
                durable = record is not None and record.state == ExecutionState.EXECUTING
                return _observation(
                    "Require reconciliation"
                    if crashed and durable and blocked
                    else "Crash repeated",
                    audit=(
                        record is not None
                        and bool(store.read_stream(f"execution:{record.execution_id}"))
                    ),
                    safety=crashed and durable and blocked and recovery.call_count == 0,
                    detail="The durable EXECUTING claim blocked replay after the crash gap.",
                )

    def _duplicate_execution(self) -> ScenarioObservation:
        with _database_path("execution_duplicate") as path:
            with SQLiteStore(path) as store:
                proposal = _archive("duplicate_execution")
                executor = MockMailboxExecutor()
                service = ExecutionService(store, executor)
                first = _execute(service, proposal)
                before = executor.call_count
                second = _execute(service, proposal)
                calls_during_duplicate = executor.call_count - before
                matched = first.execution_id == second.execution_id and calls_during_duplicate == 0
                return _observation(
                    "Do not call adapter twice" if matched else "Adapter called twice",
                    provider_calls=calls_during_duplicate,
                    audit=bool(store.read_stream(f"execution:{first.execution_id}")),
                    safety=matched,
                    detail="The stable effect key returned the first terminal result.",
                )

    def _feedback_duplicate(self) -> ScenarioObservation:
        with _database_path("feedback_duplicate") as path:
            with SQLiteStore(path) as store:
                email, proposal, decision, approval = _approved_feedback_setup(store, "duplicate")
                learner = ContextualPreferenceLearner(store)
                service = FeedbackService(store, learner)
                first = service.record_approval_feedback(
                    email=email,
                    proposal=proposal,
                    decision=decision,
                    approval_id=approval,
                    feedback=FeedbackType.APPROVED,
                    actor="user@synthetic.example",
                    feedback_id="feedback-first",
                )
                second = service.record_approval_feedback(
                    email=email,
                    proposal=proposal,
                    decision=decision,
                    approval_id=approval,
                    feedback=FeedbackType.APPROVED,
                    actor="user@synthetic.example",
                    feedback_id="feedback-duplicate-click",
                )
                state = store.get_preference(first.record.context_key)
                matched = (
                    not second.applied and second.record.feedback_id == first.record.feedback_id
                )
                return _observation(
                    "Return original receipt" if matched else "Duplicate feedback applied",
                    audit=bool(store.read_stream(f"feedback:{first.record.feedback_id}")),
                    safety=matched and state.observations == 1,
                    detail="Semantic deduplication returned the first receipt without relearning.",
                )

    def _feedback_missing_evidence(self) -> ScenarioObservation:
        with _database_path("feedback_missing") as path:
            with SQLiteStore(path) as store:
                proposal = _reply("missing_feedback")
                email = _email_for(proposal)
                service = FeedbackService(store, ContextualPreferenceLearner(store))
                context = build_preference_context(email, proposal)
                try:
                    service.record_approval_feedback(
                        email=email,
                        proposal=proposal,
                        decision=_decision(proposal),
                        approval_id="missing-approval",
                        feedback=FeedbackType.APPROVED,
                        actor="user@synthetic.example",
                    )
                except FeedbackEvidenceError:
                    rejected = True
                else:
                    rejected = False
                return _observation(
                    "Reject learning" if rejected else "Unsupported learning applied",
                    audit=True,
                    safety=rejected and store.get_preference(context.key).observations == 0,
                    detail="Feedback without durable workflow evidence could not update learning.",
                )

    def _feedback_write_failure(self) -> ScenarioObservation:
        with _database_path("feedback_write") as path:
            with FeedbackAuditFailingStore(path) as store:
                email, proposal, decision, approval = _approved_feedback_setup(store, "write")
                learner = ContextualPreferenceLearner(store)
                context = build_preference_context(email, proposal)
                store.fail_feedback_audit = True
                try:
                    FeedbackService(store, learner).record_approval_feedback(
                        email=email,
                        proposal=proposal,
                        decision=decision,
                        approval_id=approval,
                        feedback=FeedbackType.APPROVED,
                        actor="user@synthetic.example",
                        feedback_id="feedback-write-failure",
                    )
                except StorageError:
                    rolled_back = True
                else:
                    rolled_back = False
                no_receipt = store.get_feedback("feedback-write-failure") is None
                unchanged = store.get_preference(context.key).observations == 0
                return _observation(
                    "Roll back all feedback writes"
                    if rolled_back and no_receipt and unchanged
                    else "Partial feedback write",
                    audit=True,
                    safety=rolled_back and no_receipt and unchanged,
                    detail=(
                        "Failure during audit insertion rolled back receipt and preference state."
                    ),
                )

    def _newer_schema(self) -> ScenarioObservation:
        with _database_path("future") as path:
            connection = sqlite3.connect(path)
            try:
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
            finally:
                connection.close()
            try:
                unexpected_store = SQLiteStore(path)
            except SchemaVersionError:
                refused = True
            else:
                unexpected_store.close()
                refused = False
            return _observation(
                "Refuse to open" if refused else "Opened future schema",
                audit=True,
                safety=refused,
                detail="The store rejected a database newer than the application schema.",
            )

    def _audit_update(self) -> ScenarioObservation:
        with _database_path("audit") as path:
            with SQLiteStore(path) as store:
                original = store.append_event(
                    stream_id="failure:audit",
                    event_type="failure.original",
                    payload={"safe": True},
                    event_id="failure-audit-event",
                )
            connection = sqlite3.connect(path)
            try:
                try:
                    connection.execute(
                        "UPDATE audit_events SET event_type = 'tampered' WHERE event_id = ?",
                        (original.event_id,),
                    )
                except sqlite3.IntegrityError:
                    rejected = True
                else:
                    rejected = False
            finally:
                connection.close()
            with SQLiteStore(path) as reopened:
                unchanged = reopened.read_stream("failure:audit") == (original,)
            return _observation(
                "Database trigger rejects update" if rejected else "Audit update succeeded",
                audit=unchanged,
                safety=rejected and unchanged,
                detail="The SQLite append-only trigger rejected a direct SQL update.",
            )

    def _gmail_missing_token(self) -> ScenarioObservation:
        variable = "WAJO_FAILURE_TEST_MISSING_TOKEN"
        previous = os.environ.pop(variable, None)
        try:
            try:
                EnvironmentAccessTokenProvider(variable).access_token()
            except GmailConfigurationError:
                rejected = True
            else:
                rejected = False
        finally:
            if previous is not None:
                os.environ[variable] = previous
        return _observation(
            "No Gmail request" if rejected else "Missing token accepted",
            audit=True,
            safety=rejected,
            detail="Missing OAuth material was rejected before HTTP construction.",
        )

    def _gmail_post_timeout(self) -> ScenarioObservation:
        transport = FakeGmailTransport(failure=GmailAmbiguousError("injected POST timeout"))
        executor = GmailMailboxExecutor(transport, _live_gmail_config())
        with _database_path("gmail_timeout") as path:
            with SQLiteStore(path) as store:
                proposal = _archive("gmail-failure-message")
                service = ExecutionService(store, executor)
                result = _execute(service, proposal)
                before = transport.calls
                duplicate = _execute(service, proposal)
                retries = transport.calls - before
                return _observation(
                    result.state.value.upper(),
                    provider_calls=transport.calls,
                    retries=retries,
                    audit=bool(store.read_stream(f"execution:{result.execution_id}")),
                    safety=(
                        result.state == ExecutionState.UNKNOWN
                        and duplicate.execution_id == result.execution_id
                    ),
                    detail="An ambiguous Gmail POST stayed UNKNOWN and was not replayed.",
                )

    def _gmail_recipient_blocked(self) -> ScenarioObservation:
        transport = FakeGmailTransport()
        executor = GmailMailboxExecutor(transport, _live_gmail_config())
        with _database_path("gmail_recipient") as path:
            with SQLiteStore(path) as store:
                proposal = _reply("gmail_recipient", recipient="blocked@outside.example")
                approvals = ApprovalService(store)
                request = approvals.request(proposal, _decision(proposal))
                approvals.grant(request.approval_id, proposal, actor="user@synthetic.example")
                service = ExecutionService(store, executor, approval_service=approvals)
                result = service.execute(
                    mailbox_id="failure-mailbox",
                    provider_message_id="provider_gmail_recipient",
                    proposal=proposal,
                    decision=_decision(proposal),
                    risk=RiskAssessment(),
                    preference_tier=AutonomyTier.SILENT,
                    approval_id=request.approval_id,
                )
                return _observation(
                    result.state.value.upper(),
                    provider_calls=transport.calls,
                    audit=bool(store.read_stream(f"execution:{result.execution_id}")),
                    safety=result.state == ExecutionState.FAILED_SAFE,
                    detail="The recipient allowlist blocked Gmail before the provider call.",
                )

    def _gmail_label_missing(self) -> ScenarioObservation:
        transport = FakeGmailTransport()
        executor = GmailMailboxExecutor(transport, _live_gmail_config())
        with _database_path("gmail_label") as path:
            with SQLiteStore(path) as store:
                proposal = _label("gmail_label")
                result = _execute(ExecutionService(store, executor), proposal)
                return _observation(
                    result.state.value.upper(),
                    provider_calls=transport.calls,
                    audit=bool(store.read_stream(f"execution:{result.execution_id}")),
                    safety=result.state == ExecutionState.FAILED_SAFE,
                    detail="An unmapped Gmail label failed before any provider request.",
                )


def _observation(
    outcome: str,
    *,
    provider_calls: int = 0,
    retries: int = 0,
    audit: bool,
    safety: bool,
    detail: str,
) -> ScenarioObservation:
    return ScenarioObservation(
        observed_outcome=outcome,
        observed_provider_calls=provider_calls,
        automatic_retry_attempts=retries,
        audit_evidence_present=audit,
        safety_floor_preserved=safety,
        detail=detail,
    )


def _newsletter(suffix: str) -> EmailEnvelope:
    return EmailEnvelope(
        email_id=f"email_{suffix}",
        provider_message_id=f"provider_{suffix}",
        sender=f"{suffix}@synthetic.example",
        subject="Weekly newsletter",
        body_text="This is a weekly digest.",
    )


def _archive(suffix: str) -> ActionProposal:
    return ActionProposal(
        proposal_id=f"proposal_archive_{suffix}",
        email_id=f"email_archive_{suffix}",
        action_type=ActionType.ARCHIVE,
        intent=Intent.NEWSLETTER,
        summary="Archive the synthetic newsletter",
        payload=MessagePayload(kind=ActionType.ARCHIVE, message_id=f"provider_{suffix}"),
    )


def _label(suffix: str) -> ActionProposal:
    return ActionProposal(
        proposal_id=f"proposal_label_{suffix}",
        email_id=f"email_label_{suffix}",
        action_type=ActionType.ADD_LABEL,
        intent=Intent.RECEIPT,
        summary="Label the synthetic receipt",
        payload=LabelPayload(message_id=f"provider_{suffix}", label="Unmapped"),
    )


def _reply(suffix: str, *, recipient: str = "sender@synthetic.example") -> ActionProposal:
    return ActionProposal(
        proposal_id=f"proposal_reply_{suffix}",
        email_id=f"email_reply_{suffix}",
        action_type=ActionType.SEND_REPLY,
        intent=Intent.REQUEST,
        summary="Reply to the synthetic request",
        payload=ReplyPayload(
            kind=ActionType.SEND_REPLY,
            recipients=(recipient,),
            body="Synthetic approved response",
        ),
    )


def _revised_reply(proposal: ActionProposal, body: str) -> ActionProposal:
    return proposal.model_copy(
        update={
            "version": proposal.version + 1,
            "payload": proposal.payload.model_copy(update={"body": body}),
        }
    )


def _decision(proposal: ActionProposal) -> Decision:
    return PolicyEngine().decide(proposal, RiskAssessment(), AutonomyTier.SILENT)


def _agent(
    store: SQLiteStore,
    planner: Planner,
    executor: MailboxExecutor,
) -> ProactiveEmailAgent:
    learner = ContextualPreferenceLearner(store)
    approvals = ApprovalService(store)
    return ProactiveEmailAgent(
        mailbox_id="failure-mailbox",
        planner=planner,
        store=store,
        learner=learner,
        approvals=approvals,
        execution=ExecutionService(store, executor, approval_service=approvals),
    )


def _execute(service: ExecutionService, proposal: ActionProposal) -> ExecutionResult:
    return service.execute(
        mailbox_id="failure-mailbox",
        provider_message_id=_provider_message_id(proposal),
        proposal=proposal,
        decision=_decision(proposal),
        risk=RiskAssessment(),
        preference_tier=AutonomyTier.SILENT,
    )


def _execute_archive(executor: MailboxExecutor, suffix: str) -> tuple[ExecutionResult, bool]:
    with _database_path("execution") as path:
        with SQLiteStore(path) as store:
            result = _execute(ExecutionService(store, executor), _archive(suffix))
            audited = bool(store.read_stream(f"execution:{result.execution_id}"))
            return result, audited


def _email_for(proposal: ActionProposal) -> EmailEnvelope:
    return EmailEnvelope(
        email_id=proposal.email_id,
        provider_message_id=f"provider_{proposal.email_id}",
        sender="sender@synthetic.example",
        subject="Please review this request",
        body_text="Could you confirm the response?",
    )


def _approved_feedback_setup(
    store: SQLiteStore, suffix: str
) -> tuple[EmailEnvelope, ActionProposal, Decision, str]:
    proposal = _reply(f"feedback_{suffix}")
    decision = _decision(proposal)
    service = ApprovalService(store)
    record = service.request(proposal, decision)
    service.grant(record.approval_id, proposal, actor="user@synthetic.example")
    return _email_for(proposal), proposal, decision, record.approval_id


def _live_gmail_config() -> GmailAdapterConfig:
    return GmailAdapterConfig(
        enabled=True,
        dry_run=False,
        allowed_recipients=("allowed@synthetic.example",),
    )


def _provider_message_id(proposal: ActionProposal) -> str:
    if isinstance(proposal.payload, (MessagePayload, LabelPayload)):
        return proposal.payload.message_id
    raise ValueError("failure helper requires a message-scoped proposal")


@contextmanager
def _database_path(label: str) -> Generator[Path, None, None]:
    """Create a unique flat workspace database and remove its known SQLite artifacts."""
    path = Path.cwd() / f"failure_eval_{label}_{uuid4().hex}.sqlite3"
    artifacts = tuple(Path(f"{path}{suffix}") for suffix in ("", "-wal", "-shm"))
    if any(artifact.exists() for artifact in artifacts):
        raise RuntimeError("failure-evaluation artifact unexpectedly exists")
    try:
        yield path
    finally:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
