"""Single-use, expiring approval workflow bound to exact proposal bytes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hmac import compare_digest

from pydantic import JsonValue

from wajo_agent.approvals.canonicalize import approval_payload_hash
from wajo_agent.domain import (
    ActionProposal,
    ApprovalRecord,
    ApprovalStatus,
    AutonomyTier,
    Decision,
)
from wajo_agent.domain.autonomy import most_restrictive
from wajo_agent.domain.models import utc_now
from wajo_agent.policy import get_capability
from wajo_agent.storage import ApprovalNotFoundError as StorageApprovalNotFoundError
from wajo_agent.storage import ApprovalStateConflictError, SQLiteStore

DEFAULT_APPROVAL_TTL = timedelta(minutes=15)
MAX_APPROVAL_TTL = timedelta(hours=24)


class ApprovalError(RuntimeError):
    """Base class for safe approval failures."""


class ApprovalNotFoundError(ApprovalError):
    """The requested approval does not exist."""


class ApprovalDecisionError(ApprovalError):
    """The supplied decision cannot create an approval request."""


class ApprovalBindingError(ApprovalError):
    """The proposal no longer matches the exact payload the user reviewed."""


class ApprovalStateError(ApprovalError):
    """The requested transition is illegal for the current approval state."""


class ApprovalExpiredError(ApprovalStateError):
    """The approval expired before the requested operation."""


class ApprovalService:
    """Coordinates approval state while SQLite provides atomic persistence."""

    def __init__(
        self,
        store: SQLiteStore,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._store = store
        self._clock = clock

    def request(
        self,
        proposal: ActionProposal,
        decision: Decision,
        *,
        ttl: timedelta = DEFAULT_APPROVAL_TTL,
    ) -> ApprovalRecord:
        """Create a pending request only for an exact ASK decision."""
        _validate_ask_decision(proposal, decision)
        _validate_ttl(ttl)
        now = self._now()
        record = ApprovalRecord(
            proposal_id=proposal.proposal_id,
            proposal_version=proposal.version,
            payload_hash=approval_payload_hash(proposal),
            created_at=now,
            expires_at=now + ttl,
            updated_at=now,
        )
        return self._store.create_approval(
            record,
            event_payload=_request_event_payload(record),
        )

    def get(self, approval_id: str) -> ApprovalRecord:
        """Load an approval and durably mark it expired when its deadline passed."""
        record = self._load(approval_id)
        return self._expire_if_due(record, self._now())

    def grant(
        self,
        approval_id: str,
        proposal: ActionProposal,
        *,
        actor: str,
    ) -> ApprovalRecord:
        """Grant a pending approval after checking its exact proposal binding."""
        now = self._now()
        record = self._active_record(approval_id, now)
        _require_status(record, ApprovalStatus.PENDING)
        _verify_binding(record, proposal)
        cleaned_actor = _clean_actor(actor)
        granted = record.model_copy(
            update={
                "status": ApprovalStatus.GRANTED,
                "actor": cleaned_actor,
                "granted_at": now,
                "updated_at": now,
            }
        )
        return self._transition(
            granted,
            expected_statuses=(ApprovalStatus.PENDING,),
            event_type="approval.granted",
            event_payload={
                "approval_id": granted.approval_id,
                "actor": cleaned_actor,
                "payload_hash": granted.payload_hash,
            },
        )

    def reject(self, approval_id: str, *, actor: str) -> ApprovalRecord:
        """Reject a pending request; no executable authorization is created."""
        now = self._now()
        record = self._active_record(approval_id, now)
        _require_status(record, ApprovalStatus.PENDING)
        cleaned_actor = _clean_actor(actor)
        rejected = record.model_copy(
            update={
                "status": ApprovalStatus.REJECTED,
                "actor": cleaned_actor,
                "updated_at": now,
            }
        )
        return self._transition(
            rejected,
            expected_statuses=(ApprovalStatus.PENDING,),
            event_type="approval.rejected",
            event_payload={
                "approval_id": rejected.approval_id,
                "actor": cleaned_actor,
            },
        )

    def invalidate(
        self,
        approval_id: str,
        proposal: ActionProposal,
        *,
        actor: str,
    ) -> ApprovalRecord:
        """Revoke a pending or granted approval before execution consumes it."""
        now = self._now()
        record = self._active_record(approval_id, now)
        _require_status(record, ApprovalStatus.PENDING, ApprovalStatus.GRANTED)
        _verify_binding(record, proposal)
        cleaned_actor = _clean_actor(actor)
        invalidated = record.model_copy(
            update={
                "status": ApprovalStatus.INVALIDATED,
                "actor": record.actor or cleaned_actor,
                "updated_at": now,
            }
        )
        return self._transition(
            invalidated,
            expected_statuses=(record.status,),
            event_type="approval.invalidated",
            event_payload={
                "approval_id": invalidated.approval_id,
                "invalidated_by": cleaned_actor,
                "reason": "revoked",
            },
        )

    def consume(
        self,
        approval_id: str,
        proposal: ActionProposal,
    ) -> ApprovalRecord:
        """Atomically consume a granted approval exactly once for its bound proposal."""
        consumed = self.prepare_consumption(approval_id, proposal)
        return self._transition(
            consumed,
            expected_statuses=(ApprovalStatus.GRANTED,),
            event_type="approval.consumed",
            event_payload={
                "approval_id": consumed.approval_id,
                "payload_hash": consumed.payload_hash,
            },
        )

    def prepare_consumption(
        self,
        approval_id: str,
        proposal: ActionProposal,
    ) -> ApprovalRecord:
        """Build a checked consumed record for an enclosing atomic execution claim."""
        now = self._now()
        record = self._active_record(approval_id, now)
        _require_status(record, ApprovalStatus.GRANTED)
        _verify_binding(record, proposal)
        return record.model_copy(
            update={
                "status": ApprovalStatus.CONSUMED,
                "consumed_at": now,
                "updated_at": now,
            }
        )

    def replace_for_edit(
        self,
        approval_id: str,
        original: ActionProposal,
        revised: ActionProposal,
        revised_decision: Decision,
        *,
        actor: str,
        ttl: timedelta = DEFAULT_APPROVAL_TTL,
    ) -> tuple[ApprovalRecord, ApprovalRecord]:
        """Atomically invalidate a pending request and create its re-reviewed version."""
        _validate_ttl(ttl)
        _validate_ask_decision(revised, revised_decision)
        now = self._now()
        existing = self._active_record(approval_id, now)
        _require_status(existing, ApprovalStatus.PENDING)
        _verify_binding(existing, original)
        if revised.proposal_id != original.proposal_id:
            raise ApprovalBindingError("edited proposal must retain its proposal identity")
        if revised.version != original.version + 1:
            raise ApprovalBindingError("edited proposal version must increase by exactly one")
        revised_hash = approval_payload_hash(revised)
        if compare_digest(existing.payload_hash, revised_hash):
            raise ApprovalBindingError("edited proposal must change the approved payload")

        cleaned_actor = _clean_actor(actor)
        replacement = ApprovalRecord(
            proposal_id=revised.proposal_id,
            proposal_version=revised.version,
            payload_hash=revised_hash,
            created_at=now,
            expires_at=now + ttl,
            updated_at=now,
        )
        invalidated = existing.model_copy(
            update={
                "status": ApprovalStatus.INVALIDATED,
                "actor": cleaned_actor,
                "updated_at": now,
                "superseded_by_approval_id": replacement.approval_id,
            }
        )
        try:
            return self._store.replace_approval_for_edit(
                invalidated,
                replacement,
                expected_statuses=(ApprovalStatus.PENDING,),
                invalidated_event_payload={
                    "approval_id": invalidated.approval_id,
                    "invalidated_by": cleaned_actor,
                    "reason": "proposal_edited",
                    "superseded_by_approval_id": replacement.approval_id,
                },
                replacement_event_payload=_request_event_payload(replacement),
            )
        except ApprovalStateConflictError as exc:
            raise ApprovalStateError("approval changed before edit replacement committed") from exc

    def _active_record(self, approval_id: str, now: datetime) -> ApprovalRecord:
        record = self._load(approval_id)
        if now < record.created_at:
            raise ApprovalError("approval clock moved before request creation")
        expired = self._expire_if_due(record, now)
        if expired.status == ApprovalStatus.EXPIRED:
            raise ApprovalExpiredError(f"approval expired: {expired.approval_id}")
        return expired

    def _expire_if_due(self, record: ApprovalRecord, now: datetime) -> ApprovalRecord:
        if record.status not in {ApprovalStatus.PENDING, ApprovalStatus.GRANTED}:
            return record
        if now < record.expires_at:
            return record
        expired = record.model_copy(
            update={
                "status": ApprovalStatus.EXPIRED,
                "updated_at": now,
            }
        )
        return self._transition(
            expired,
            expected_statuses=(record.status,),
            event_type="approval.expired",
            event_payload={"approval_id": expired.approval_id},
        )

    def _load(self, approval_id: str) -> ApprovalRecord:
        cleaned_id = approval_id.strip()
        if not cleaned_id:
            raise ApprovalNotFoundError("approval ID cannot be blank")
        try:
            return self._store.get_approval(cleaned_id)
        except StorageApprovalNotFoundError as exc:
            raise ApprovalNotFoundError(str(exc)) from exc

    def _transition(
        self,
        record: ApprovalRecord,
        *,
        expected_statuses: tuple[ApprovalStatus, ...],
        event_type: str,
        event_payload: dict[str, JsonValue],
    ) -> ApprovalRecord:
        try:
            return self._store.transition_approval(
                record,
                expected_statuses=expected_statuses,
                event_type=event_type,
                event_payload=event_payload,
            )
        except ApprovalStateConflictError as exc:
            raise ApprovalStateError("approval changed before transition committed") from exc

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ApprovalError("approval clock must return a timezone-aware time")
        return value.astimezone(UTC)


def _validate_ask_decision(proposal: ActionProposal, decision: Decision) -> None:
    if decision.tier != AutonomyTier.ASK:
        raise ApprovalDecisionError("only an ASK decision may create an approval")
    if (
        decision.proposal_id != proposal.proposal_id
        or decision.proposal_version != proposal.version
    ):
        raise ApprovalDecisionError("decision does not identify the exact proposal version")
    capability = get_capability(proposal.action_type)
    if not capability.enabled or capability.minimum_tier == AutonomyTier.ESCALATE:
        raise ApprovalDecisionError("escalated or disabled capability cannot be approved")
    if decision.capability_floor != capability.minimum_tier:
        raise ApprovalDecisionError("decision contains the wrong capability floor")
    expected_tier = most_restrictive(
        capability.minimum_tier,
        decision.preference_tier,
        decision.content_floor,
    )
    if decision.tier != expected_tier:
        raise ApprovalDecisionError("decision is inconsistent with its recorded safety floors")


def _verify_binding(record: ApprovalRecord, proposal: ActionProposal) -> None:
    if record.proposal_id != proposal.proposal_id or record.proposal_version != proposal.version:
        raise ApprovalBindingError("approval is bound to a different proposal version")
    actual_hash = approval_payload_hash(proposal)
    if not compare_digest(record.payload_hash, actual_hash):
        raise ApprovalBindingError("proposal payload changed after the approval request")


def _require_status(record: ApprovalRecord, *allowed: ApprovalStatus) -> None:
    if record.status not in allowed:
        expected = ", ".join(status.value for status in allowed)
        raise ApprovalStateError(
            f"approval is {record.status.value}; operation requires {expected}"
        )


def _clean_actor(actor: str) -> str:
    cleaned = actor.strip()
    if not cleaned:
        raise ApprovalError("approval actor cannot be blank")
    if len(cleaned) > 200:
        raise ApprovalError("approval actor cannot exceed 200 characters")
    return cleaned


def _validate_ttl(ttl: timedelta) -> None:
    if ttl <= timedelta(0):
        raise ApprovalError("approval TTL must be positive")
    if ttl > MAX_APPROVAL_TTL:
        raise ApprovalError("approval TTL cannot exceed 24 hours")


def _request_event_payload(record: ApprovalRecord) -> dict[str, JsonValue]:
    return {
        "approval_id": record.approval_id,
        "proposal_id": record.proposal_id,
        "proposal_version": record.proposal_version,
        "payload_hash": record.payload_hash,
        "expires_at": record.expires_at.isoformat(),
    }
