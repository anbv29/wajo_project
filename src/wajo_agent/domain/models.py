from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AutonomyTier(StrEnum):
    SILENT = "silent"
    NOTIFY = "notify"
    ASK = "ask"
    ESCALATE = "escalate"


TIER_RANK: dict[AutonomyTier, int] = {
    AutonomyTier.SILENT: 0,
    AutonomyTier.NOTIFY: 1,
    AutonomyTier.ASK: 2,
    AutonomyTier.ESCALATE: 3,
}


def most_restrictive(*tiers: AutonomyTier) -> AutonomyTier:
    return max(tiers, key=TIER_RANK.__getitem__)


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
    EXTERNAL_SINGLE = "external_single"
    EXTERNAL_MULTIPLE = "external_multiple"
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
    message_id: str


class LabelPayload(StrictModel):
    kind: Literal[ActionType.ADD_LABEL] = ActionType.ADD_LABEL
    message_id: str
    label: str = Field(min_length=1, max_length=100)


class DraftPayload(StrictModel):
    kind: Literal[ActionType.CREATE_DRAFT] = ActionType.CREATE_DRAFT
    recipients: tuple[str, ...]
    subject: str
    body: str


class ReplyPayload(StrictModel):
    kind: Literal[
        ActionType.SEND_REPLY,
        ActionType.FORWARD,
        ActionType.UNSUBSCRIBE,
        ActionType.PAYMENT,
        ActionType.ACCOUNT_CHANGE,
    ]
    recipients: tuple[str, ...] = ()
    subject: str = ""
    body: str = ""


ActionPayload = Annotated[
    NoActionPayload | MessagePayload | LabelPayload | DraftPayload | ReplyPayload,
    Field(discriminator="kind"),
]


class ActionProposal(StrictModel):
    proposal_id: str = Field(default_factory=lambda: new_id("proposal"))
    version: int = 1
    email_id: str
    action_type: ActionType
    intent: Intent
    summary: str
    payload: ActionPayload
    evidence: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def payload_matches_action(self) -> ActionProposal:
        if self.payload.kind != self.action_type:
            raise ValueError("payload kind must match action_type")
        return self


class RiskAssessment(StrictModel):
    injection_detected: bool = False
    sensitive_categories: frozenset[str] = frozenset()
    suspicious_patterns: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    normalized_changed: bool = False


class PreferenceContext(StrictModel):
    action_type: ActionType
    intent: Intent
    sender_bucket: SenderBucket
    recipient_scope: RecipientScope

    @property
    def key(self) -> str:
        return "|".join((self.action_type, self.intent, self.sender_bucket, self.recipient_scope))


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
    reasons: tuple[str, ...]
    created_at: datetime = Field(default_factory=utc_now)


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
