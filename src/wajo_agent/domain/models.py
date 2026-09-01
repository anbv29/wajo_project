from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

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
    context_key: str
    alpha: int = 1
    beta: int = 1
    observations: int = 0
    recent_feedback: tuple[FeedbackType, ...] = ()
    cooldown_remaining: int = 0


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
    proposal_id: str
    proposal_version: int
    payload_hash: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    actor: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    consumed_at: datetime | None = None


class ExecutionResult(StrictModel):
    execution_id: str = Field(default_factory=lambda: new_id("execution"))
    idempotency_key: str
    proposal_id: str
    state: ExecutionState
    detail: str
    provider_operation_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class AgentOutcome(StrictModel):
    email: EmailEnvelope
    proposal: ActionProposal | None
    risk: RiskAssessment
    decision: Decision
    approval: ApprovalRecord | None = None
    execution: ExecutionResult | None = None
