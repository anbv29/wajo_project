from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    field_validator,
    model_validator,
)

from wajo_agent.domain.autonomy import AutonomyTier


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionType(StrEnum):
    NO_ACTION = "no_action"
    MARK_READ = "mark_read"
    MARK_UNREAD = "mark_unread"
    ADD_LABEL = "add_label"
    ARCHIVE = "archive"
    CREATE_DRAFT = "create_draft"
    TRASH = "trash"
    SEND_REPLY = "send_reply"
    FORWARD = "forward"
    UNSUBSCRIBE = "unsubscribe"
    PERMANENT_DELETE = "permanent_delete"
    PAYMENT = "payment"
    ACCOUNT_CHANGE = "account_change"


class Intent(StrEnum):
    NEWSLETTER = "newsletter"
    INFORMATIONAL = "informational"
    REQUEST = "request"
    MEETING = "meeting"
    RECEIPT = "receipt"
    PERSONAL = "personal"
    ACCOUNT_RECOVERY = "account_recovery"
    FINANCIAL = "financial"
    LEGAL = "legal"
    SPAM = "spam"
    UNKNOWN = "unknown"


class SenderBucket(StrEnum):
    KNOWN_PERSON = "known_person"
    KNOWN_BULK = "known_bulk"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class RecipientScope(StrEnum):
    INTERNAL_MAILBOX = "internal_mailbox"
    INTERNAL_RECIPIENTS = "internal_recipients"
    EXTERNAL_SINGLE = "external_single"
    EXTERNAL_MULTIPLE = "external_multiple"
    MIXED_RECIPIENTS = "mixed_recipients"
    NONE = "none"


class FeedbackType(StrEnum):
    APPROVED = "approved"
    CORRECT = "correct"
    EDITED = "edited"
    REJECTED = "rejected"
    UNDONE = "undone"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    GRANTED = "granted"
    REJECTED = "rejected"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


class ExecutionState(StrEnum):
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED_SAFE = "failed_safe"
    UNKNOWN = "unknown"


class OutcomeRoute(StrEnum):
    EXECUTED_SILENTLY = "executed_silently"
    EXECUTED_AND_NOTIFY = "executed_and_notify"
    AWAITING_APPROVAL = "awaiting_approval"
    ESCALATED = "escalated"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_UNKNOWN = "execution_unknown"


class InjectionSignal(StrEnum):
    """Prompt-injection techniques observed in untrusted email text."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    FAKE_PRIVILEGED_ROLE = "fake_privileged_role"
    SECRET_EXFILTRATION = "secret_exfiltration"
    FAKE_APPROVAL = "fake_approval"
    ENCODED_INSTRUCTIONS = "encoded_instructions"
    OBFUSCATED_INSTRUCTIONS = "obfuscated_instructions"
    HIDDEN_ONLY_CONTENT = "hidden_only_content"


class SensitiveCategory(StrEnum):
    """Content categories that later policy code must treat cautiously."""

    CREDENTIALS = "credentials"
    ACCOUNT_RECOVERY = "account_recovery"
    BANKING = "banking"
    PAYMENT = "payment"
    LEGAL_COMMITMENT = "legal_commitment"
    PERSONAL_DATA = "personal_data"


class NormalizationFlag(StrEnum):
    """Lossy or suspicious transformations reported by email normalization."""

    TRUNCATED_CONTENT = "truncated_content"
    INVISIBLE_CHARACTERS = "invisible_characters"
    CONTROL_CHARACTERS = "control_characters"
    NO_VISIBLE_CONTENT = "no_visible_content"


class RiskEvidenceSource(StrEnum):
    SUBJECT = "subject"
    BODY = "body"
    NORMALIZATION = "normalization"


class AttachmentMetadata(StrictModel):
    """Safe facts about an attachment; the attachment bytes are not opened here."""

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="application/octet-stream", max_length=255)
    size_bytes: int = Field(ge=0)
    provider_attachment_id: str | None = None
    is_inline: bool = False

    @field_validator("filename", "content_type")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned


class EmailEnvelope(StrictModel):
    """The typed, immutable observation the agent receives from a mailbox adapter."""

    email_id: str = Field(default_factory=lambda: new_id("email"))
    provider_message_id: str = Field(min_length=1, max_length=512)
    provider_thread_id: str | None = Field(default=None, max_length=512)
    sender: str = Field(min_length=1, max_length=512)
    recipients: tuple[str, ...] = ()
    subject: str = Field(default="", max_length=1_000)
    body_text: str = ""
    body_html: str | None = None
    received_at: datetime = Field(default_factory=utc_now)
    sender_bucket: SenderBucket = SenderBucket.UNKNOWN
    source: Literal["fixture", "gmail"] = "fixture"
    attachments: tuple[AttachmentMetadata, ...] = ()

    @field_validator("provider_message_id", "sender")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned

    @field_validator("provider_thread_id")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("recipients")
    @classmethod
    def clean_recipients(cls, recipients: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(address.strip() for address in recipients if address.strip())
        return tuple(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def contains_observable_content(self) -> EmailEnvelope:
        has_html = self.body_html is not None and bool(self.body_html.strip())
        if not any((self.subject.strip(), self.body_text.strip(), has_html, self.attachments)):
            raise ValueError("email must contain a subject, body, or attachment")
        return self


class NoActionPayload(StrictModel):
    kind: Literal[ActionType.NO_ACTION] = ActionType.NO_ACTION


class MessagePayload(StrictModel):
    kind: Literal[
        ActionType.MARK_READ,
        ActionType.MARK_UNREAD,
        ActionType.ARCHIVE,
        ActionType.TRASH,
        ActionType.PERMANENT_DELETE,
    ]
    message_id: str = Field(min_length=1, max_length=512)

    @field_validator("message_id")
    @classmethod
    def clean_message_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message_id cannot be blank")
        return cleaned


class LabelPayload(StrictModel):
    kind: Literal[ActionType.ADD_LABEL] = ActionType.ADD_LABEL
    message_id: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=100)

    @field_validator("message_id", "label")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned


class DraftPayload(StrictModel):
    kind: Literal[ActionType.CREATE_DRAFT] = ActionType.CREATE_DRAFT
    recipients: tuple[str, ...] = Field(default=(), max_length=100)
    subject: str = Field(default="", max_length=1_000)
    body: str = Field(default="", max_length=50_000)

    @field_validator("recipients")
    @classmethod
    def clean_recipients(cls, recipients: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(address.strip() for address in recipients if address.strip())
        return tuple(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def contains_draft_content(self) -> DraftPayload:
        if not self.subject.strip() and not self.body.strip():
            raise ValueError("draft must contain a subject or body")
        return self


class ReplyPayload(StrictModel):
    kind: Literal[
        ActionType.SEND_REPLY,
        ActionType.FORWARD,
        ActionType.UNSUBSCRIBE,
        ActionType.PAYMENT,
        ActionType.ACCOUNT_CHANGE,
    ]
    recipients: tuple[str, ...] = Field(default=(), max_length=100)
    subject: str = Field(default="", max_length=1_000)
    body: str = Field(default="", max_length=50_000)

    @field_validator("recipients")
    @classmethod
    def clean_recipients(cls, recipients: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(address.strip() for address in recipients if address.strip())
        return tuple(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def validate_action_requirements(self) -> ReplyPayload:
        if self.kind in {ActionType.SEND_REPLY, ActionType.FORWARD} and not self.recipients:
            raise ValueError(f"{self.kind} requires at least one recipient")
        if self.kind == ActionType.SEND_REPLY and not self.body.strip():
            raise ValueError("send_reply requires a non-empty body")
        return self


ActionPayload = Annotated[
    NoActionPayload | MessagePayload | LabelPayload | DraftPayload | ReplyPayload,
    Field(discriminator="kind"),
]


class PlannerRequest(StrictModel):
    """Normalized, explicitly untrusted data passed across the planner boundary."""

    request_id: str = Field(default_factory=lambda: new_id("plan_request"), max_length=512)
    email: EmailEnvelope
    allowed_actions: tuple[ActionType, ...] = Field(min_length=1)
    normalization_changed: bool = False
    content_trust: Literal["untrusted_email"] = "untrusted_email"

    @field_validator("request_id")
    @classmethod
    def clean_request_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("request_id cannot be blank")
        return cleaned

    @field_validator("allowed_actions")
    @classmethod
    def unique_allowed_actions(cls, actions: tuple[ActionType, ...]) -> tuple[ActionType, ...]:
        if len(actions) != len(set(actions)):
            raise ValueError("allowed_actions cannot contain duplicates")
        return actions


class PlannerOutput(StrictModel):
    """The only action-shaped data the AI planner is allowed to return."""

    action_type: ActionType
    intent: Intent
    summary: str = Field(min_length=1, max_length=500)
    payload: ActionPayload
    evidence: tuple[str, ...] = Field(default=(), max_length=10)
    uncertainty_reasons: tuple[str, ...] = Field(default=(), max_length=10)

    @model_validator(mode="after")
    def payload_matches_action(self) -> PlannerOutput:
        if self.payload.kind != self.action_type:
            raise ValueError("payload kind must match action_type")
        return self

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary cannot be blank")
        return cleaned

    @field_validator("evidence", "uncertainty_reasons")
    @classmethod
    def clean_explanations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("explanation entries cannot be blank")
        if any(len(value) > 500 for value in cleaned):
            raise ValueError("explanation entries cannot exceed 500 characters")
        return cleaned

    def bind_to_email(self, email_id: str) -> ActionProposal:
        """Add application-owned identity after the model output has been validated."""
        return ActionProposal(
            email_id=email_id,
            action_type=self.action_type,
            intent=self.intent,
            summary=self.summary,
            payload=self.payload,
            evidence=self.evidence,
            uncertainty_reasons=self.uncertainty_reasons,
        )


class ActionProposal(PlannerOutput):
    """A validated planner suggestion bound to one observed email."""

    proposal_id: str = Field(default_factory=lambda: new_id("proposal"), max_length=512)
    version: int = Field(default=1, ge=1)
    email_id: str = Field(min_length=1, max_length=512)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("proposal_id", "email_id")
    @classmethod
    def clean_identity(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("proposal identity cannot be blank")
        return cleaned


class RiskEvidence(StrictModel):
    """A small, auditable reason why the scanner emitted a signal."""

    signal: str = Field(min_length=1, max_length=100)
    source: RiskEvidenceSource
    matched_text: str = Field(min_length=1, max_length=160)


class RiskAssessment(StrictModel):
    """Evidence from perception only; this model does not choose autonomy."""

    injection_signals: frozenset[InjectionSignal] = frozenset()
    sensitive_categories: frozenset[SensitiveCategory] = frozenset()
    normalization_flags: frozenset[NormalizationFlag] = frozenset()
    evidence: tuple[RiskEvidence, ...] = Field(default=(), max_length=50)
    missing_information: tuple[str, ...] = ()
    normalization_changed: bool = False

    @computed_field
    @property
    def injection_detected(self) -> bool:
        """Derived from evidence so the boolean cannot contradict the signals."""
        return bool(self.injection_signals)


class PreferenceContext(StrictModel):
    """Exact, versioned identity of one narrow preference-learning situation."""

    schema_version: Literal[1] = 1
    action_type: ActionType
    intent: Intent
    sender_bucket: SenderBucket
    sender_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipient_scope: RecipientScope
    action_variant: str = Field(default="default", min_length=1, max_length=100)

    @field_validator("action_variant")
    @classmethod
    def normalize_action_variant(cls, value: str) -> str:
        cleaned = value.strip().casefold()
        if not cleaned:
            raise ValueError("action_variant cannot be blank")
        return cleaned

    @property
    def canonical_dimensions(self) -> str:
        """Unambiguous input used to generate the stable storage key."""
        return "\x1f".join(
            (
                str(self.schema_version),
                self.action_type.value,
                self.intent.value,
                self.sender_bucket.value,
                self.sender_identity_hash,
                self.recipient_scope.value,
                self.action_variant,
            )
        )

    @property
    def key(self) -> str:
        digest = sha256(self.canonical_dimensions.encode("utf-8")).hexdigest()
        return f"ctx_v{self.schema_version}_{digest}"


class PreferenceState(StrictModel):
    """Current Beta evidence for one exact preference context."""

    context_key: str = Field(min_length=1, max_length=100)
    alpha: int = Field(default=1, ge=1)
    beta: int = Field(default=1, ge=1)
    observations: int = Field(default=0, ge=0)
    recent_feedback: tuple[FeedbackType, ...] = Field(default=(), max_length=5)
    cooldown_remaining: int = Field(default=0, ge=0)

    @field_validator("context_key")
    @classmethod
    def clean_context_key(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("context_key cannot be blank")
        return cleaned


class PreferenceRecommendation(StrictModel):
    """Explainable learner opinion; deterministic policy still has final authority."""

    context_key: str = Field(min_length=1, max_length=100)
    tier: AutonomyTier
    alpha: int = Field(ge=1)
    beta: int = Field(ge=1)
    observations: int = Field(ge=0)
    posterior_mean: float = Field(ge=0.0, le=1.0)
    notify_probability: float = Field(ge=0.0, le=1.0)
    silent_probability: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = Field(min_length=1, max_length=10)

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("recommendation reasons cannot be blank")
        return cleaned


class FeedbackSubmission(StrictModel):
    """One normalized explicit-feedback command before preference evidence is updated."""

    feedback_id: str = Field(default_factory=lambda: new_id("feedback"), max_length=512)
    dedupe_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_id: str = Field(min_length=1, max_length=512)
    proposal_id: str = Field(min_length=1, max_length=512)
    proposal_version: int = Field(ge=1)
    context_key: str = Field(min_length=1, max_length=100)
    feedback_type: FeedbackType
    actor: str = Field(min_length=1, max_length=200)
    source_reference: str = Field(min_length=1, max_length=512)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "feedback_id",
        "decision_id",
        "proposal_id",
        "context_key",
        "actor",
        "source_reference",
    )
    @classmethod
    def clean_feedback_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("feedback identity cannot be blank")
        return cleaned

    @field_validator("created_at")
    @classmethod
    def require_aware_feedback_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("feedback timestamp must include a timezone")
        return value.astimezone(UTC)


class FeedbackRecord(FeedbackSubmission):
    """Durable feedback receipt containing the exact evidence transition it caused."""

    previous_state: PreferenceState
    updated_state: PreferenceState

    @model_validator(mode="after")
    def validate_preference_transition(self) -> FeedbackRecord:
        if (
            self.previous_state.context_key != self.context_key
            or self.updated_state.context_key != self.context_key
        ):
            raise ValueError("feedback states belong to another preference context")
        if self.updated_state.observations != self.previous_state.observations + 1:
            raise ValueError("feedback must add exactly one observation")
        if not self.updated_state.recent_feedback:
            raise ValueError("updated preference must retain recent feedback")
        if self.updated_state.recent_feedback[-1] != self.feedback_type:
            raise ValueError("updated preference does not contain this feedback")
        return self


class AuditEvent(StrictModel):
    """One immutable event loaded from the append-only audit stream."""

    event_id: str = Field(min_length=1, max_length=512)
    stream_id: str = Field(min_length=1, max_length=512)
    sequence: int = Field(ge=1)
    event_type: str = Field(min_length=1, max_length=100)
    event_version: int = Field(default=1, ge=1)
    payload: dict[str, JsonValue]
    occurred_at: datetime

    @field_validator("event_id", "stream_id", "event_type")
    @classmethod
    def clean_event_identity(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("event identity cannot be blank")
        return cleaned

    @field_validator("occurred_at")
    @classmethod
    def require_aware_event_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(UTC)


class Decision(StrictModel):
    decision_id: str = Field(default_factory=lambda: new_id("decision"))
    proposal_id: str | None
    proposal_version: int | None
    tier: AutonomyTier
    capability_floor: AutonomyTier
    preference_tier: AutonomyTier
    content_floor: AutonomyTier
    reasons: tuple[str, ...] = Field(min_length=1, max_length=20)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_audit_identity(self) -> Decision:
        if (self.proposal_id is None) != (self.proposal_version is None):
            raise ValueError("proposal_id and proposal_version must be present together")
        if any(not reason.strip() for reason in self.reasons):
            raise ValueError("decision reasons cannot be blank")
        if any(len(reason) > 500 for reason in self.reasons):
            raise ValueError("decision reasons cannot exceed 500 characters")
        return self


class ApprovalRecord(StrictModel):
    approval_id: str = Field(default_factory=lambda: new_id("approval"))
    proposal_id: str = Field(min_length=1, max_length=512)
    proposal_version: int = Field(ge=1)
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ApprovalStatus = ApprovalStatus.PENDING
    actor: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    updated_at: datetime = Field(default_factory=utc_now)
    granted_at: datetime | None = None
    consumed_at: datetime | None = None
    superseded_by_approval_id: str | None = None

    @field_validator("approval_id", "proposal_id")
    @classmethod
    def clean_approval_identity(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("approval identity cannot be blank")
        return cleaned

    @field_validator("actor", "superseded_by_approval_id")
    @classmethod
    def clean_optional_approval_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("optional approval identity cannot be blank")
        return cleaned

    @field_validator("created_at", "expires_at", "updated_at", "granted_at", "consumed_at")
    @classmethod
    def require_aware_approval_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_approval_state(self) -> ApprovalRecord:
        if self.expires_at <= self.created_at:
            raise ValueError("approval expiry must be after creation")
        if self.updated_at < self.created_at:
            raise ValueError("approval update cannot precede creation")
        if self.granted_at is not None and self.granted_at < self.created_at:
            raise ValueError("approval grant cannot precede creation")
        if self.consumed_at is not None and self.consumed_at < self.created_at:
            raise ValueError("approval consumption cannot precede creation")
        if self.consumed_at is not None and self.consumed_at >= self.expires_at:
            raise ValueError("approval cannot be consumed at or after expiry")

        if self.status == ApprovalStatus.PENDING:
            if any(
                value is not None
                for value in (
                    self.actor,
                    self.granted_at,
                    self.consumed_at,
                    self.superseded_by_approval_id,
                )
            ):
                raise ValueError("pending approval cannot contain resolution data")
        elif self.status == ApprovalStatus.GRANTED:
            if self.actor is None or self.granted_at is None or self.consumed_at is not None:
                raise ValueError("granted approval requires actor and grant time only")
        elif self.status == ApprovalStatus.CONSUMED:
            if self.actor is None or self.granted_at is None or self.consumed_at is None:
                raise ValueError("consumed approval requires grant and consumption evidence")
        elif self.status in {ApprovalStatus.REJECTED, ApprovalStatus.INVALIDATED}:
            if self.actor is None or self.consumed_at is not None:
                raise ValueError("user-resolved approval requires an actor and cannot be consumed")
        elif self.status == ApprovalStatus.EXPIRED and self.consumed_at is not None:
            raise ValueError("expired approval cannot be consumed")

        if self.superseded_by_approval_id is not None and self.status != ApprovalStatus.INVALIDATED:
            raise ValueError("only an invalidated approval can reference its replacement")
        if self.superseded_by_approval_id == self.approval_id:
            raise ValueError("approval cannot supersede itself")
        return self


class ExecutionCommand(StrictModel):
    """A policy-authorized instruction presented to a mailbox executor."""

    execution_id: str = Field(
        default_factory=lambda: new_id("execution"), min_length=1, max_length=512
    )
    command_id: str = Field(default_factory=lambda: new_id("command"), min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    email_id: str = Field(min_length=1, max_length=512)
    proposal_id: str = Field(min_length=1, max_length=512)
    proposal_version: int = Field(ge=1)
    decision_id: str = Field(min_length=1, max_length=512)
    action_type: ActionType
    payload: ActionPayload
    authorized_tier: Literal[
        AutonomyTier.SILENT,
        AutonomyTier.NOTIFY,
        AutonomyTier.ASK,
    ]
    approval_id: str | None = Field(default=None, min_length=1, max_length=512)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "command_id",
        "execution_id",
        "idempotency_key",
        "email_id",
        "proposal_id",
        "decision_id",
    )
    @classmethod
    def clean_command_identity(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("execution command identity cannot be blank")
        return cleaned

    @field_validator("approval_id")
    @classmethod
    def clean_optional_approval(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("approval_id cannot be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_execution_authority_shape(self) -> ExecutionCommand:
        if self.payload.kind != self.action_type:
            raise ValueError("execution payload kind must match action_type")
        if self.authorized_tier == AutonomyTier.ASK and self.approval_id is None:
            raise ValueError("an ASK command requires a bound approval")
        if self.authorized_tier != AutonomyTier.ASK and self.approval_id is not None:
            raise ValueError("only an ASK command may carry an approval")
        return self


class ExecutionResult(StrictModel):
    execution_id: str = Field(default_factory=lambda: new_id("execution"))
    command_id: str = Field(min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    proposal_id: str = Field(min_length=1, max_length=512)
    state: ExecutionState
    detail: str = Field(min_length=1, max_length=1_000)
    provider_operation_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("command_id", "idempotency_key", "proposal_id", "detail")
    @classmethod
    def clean_result_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("execution result text cannot be blank")
        return cleaned

    @field_validator("provider_operation_id")
    @classmethod
    def clean_provider_operation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("created_at")
    @classmethod
    def require_aware_result_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("execution result time must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def contains_terminal_executor_state(self) -> ExecutionResult:
        terminal_states = {
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED_SAFE,
            ExecutionState.UNKNOWN,
        }
        if self.state not in terminal_states:
            raise ValueError("an execution result must contain a terminal executor state")
        return self


class ExecutionRecord(StrictModel):
    """Durable execution claim used to prevent a side effect from running twice."""

    execution_id: str = Field(min_length=1, max_length=512)
    command: ExecutionCommand
    state: Literal[
        ExecutionState.EXECUTING,
        ExecutionState.SUCCEEDED,
        ExecutionState.FAILED_SAFE,
        ExecutionState.UNKNOWN,
    ]
    attempt_count: int = Field(default=1, ge=1)
    detail: str = Field(min_length=1, max_length=1_000)
    provider_operation_id: str | None = Field(default=None, max_length=512)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @field_validator("execution_id", "detail")
    @classmethod
    def clean_execution_record_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("execution record text cannot be blank")
        return cleaned

    @field_validator("provider_operation_id")
    @classmethod
    def clean_record_provider_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("created_at", "updated_at", "started_at", "completed_at")
    @classmethod
    def require_aware_execution_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("execution timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_execution_record(self) -> ExecutionRecord:
        if self.execution_id != self.command.execution_id:
            raise ValueError("execution record and command identities differ")
        if self.started_at < self.created_at or self.updated_at < self.created_at:
            raise ValueError("execution timestamps precede creation")
        if self.state == ExecutionState.EXECUTING and self.completed_at is not None:
            raise ValueError("executing record cannot have a completion time")
        if self.state != ExecutionState.EXECUTING and self.completed_at is None:
            raise ValueError("terminal execution record requires a completion time")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("execution completion precedes start")
        return self

    def as_result(self) -> ExecutionResult:
        """Return a terminal public result for an already completed execution."""
        if self.state == ExecutionState.EXECUTING:
            raise ValueError("executing record does not have a terminal result")
        return ExecutionResult(
            execution_id=self.execution_id,
            command_id=self.command.command_id,
            idempotency_key=self.command.idempotency_key,
            proposal_id=self.command.proposal_id,
            state=self.state,
            detail=self.detail,
            provider_operation_id=self.provider_operation_id,
            created_at=self.completed_at or self.updated_at,
        )


class AgentOutcome(StrictModel):
    run_id: str = Field(min_length=1, max_length=512)
    email: EmailEnvelope
    proposal: ActionProposal | None
    risk: RiskAssessment
    preference: PreferenceRecommendation | None = None
    decision: Decision
    route: OutcomeRoute
    approval: ApprovalRecord | None = None
    execution: ExecutionResult | None = None
    user_message: str | None = Field(default=None, max_length=2_000)

    @field_validator("run_id")
    @classmethod
    def clean_run_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_id cannot be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_outcome_route(self) -> AgentOutcome:
        if self.proposal is None:
            if self.decision.tier != AutonomyTier.ESCALATE:
                raise ValueError("missing proposal requires escalation")
            if self.preference is not None:
                raise ValueError("missing proposal cannot have a preference recommendation")
        elif self.proposal.email_id != self.email.email_id:
            raise ValueError("outcome proposal belongs to another email")

        if self.route == OutcomeRoute.AWAITING_APPROVAL:
            if self.decision.tier != AutonomyTier.ASK:
                raise ValueError("approval route requires an ASK decision")
            if self.approval is None or self.execution is not None:
                raise ValueError("approval route requires approval and no execution")
        elif self.route == OutcomeRoute.ESCALATED:
            if self.decision.tier != AutonomyTier.ESCALATE:
                raise ValueError("escalation route requires an ESCALATE decision")
            if self.approval is not None or self.execution is not None:
                raise ValueError("escalation cannot approve or execute")
        else:
            if self.execution is None or self.approval is not None:
                raise ValueError("execution route requires a result and no approval request")
            if self.route == OutcomeRoute.EXECUTION_FAILED and (
                self.execution.state != ExecutionState.FAILED_SAFE
            ):
                raise ValueError("failed route requires FAILED_SAFE result")
            if self.route == OutcomeRoute.EXECUTION_UNKNOWN and (
                self.execution.state != ExecutionState.UNKNOWN
            ):
                raise ValueError("unknown route requires UNKNOWN result")
            if (
                self.route
                in {
                    OutcomeRoute.EXECUTED_SILENTLY,
                    OutcomeRoute.EXECUTED_AND_NOTIFY,
                }
                and self.execution.state != ExecutionState.SUCCEEDED
            ):
                raise ValueError("successful execution route requires SUCCEEDED result")
            if (
                self.route == OutcomeRoute.EXECUTED_SILENTLY
                and self.decision.tier != AutonomyTier.SILENT
            ):
                raise ValueError("silent route requires a SILENT decision")
            if (
                self.route == OutcomeRoute.EXECUTED_AND_NOTIFY
                and self.decision.tier != AutonomyTier.NOTIFY
            ):
                raise ValueError("notify route requires a NOTIFY decision")
            if self.route in {
                OutcomeRoute.EXECUTION_FAILED,
                OutcomeRoute.EXECUTION_UNKNOWN,
            } and self.decision.tier not in {AutonomyTier.SILENT, AutonomyTier.NOTIFY}:
                raise ValueError("autonomous failure requires a SILENT or NOTIFY decision")

        if self.user_message is not None and not self.user_message.strip():
            raise ValueError("user message cannot be blank")
        return self
