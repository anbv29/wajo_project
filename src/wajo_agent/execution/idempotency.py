"""Stable effect identity for mailbox side-effect deduplication."""

from __future__ import annotations

import json
from hashlib import sha256

from wajo_agent.domain import ActionProposal

EFFECT_KEY_SCHEMA_VERSION = 1


def canonical_effect(
    mailbox_id: str,
    provider_message_id: str,
    proposal: ActionProposal,
) -> bytes:
    """Return canonical bytes for the intended effect, excluding autonomy tier."""
    cleaned_mailbox_id = _required_identity(mailbox_id, "mailbox_id")
    cleaned_message_id = _required_identity(provider_message_id, "provider_message_id")
    document = {
        "schema_version": EFFECT_KEY_SCHEMA_VERSION,
        "mailbox_id": cleaned_mailbox_id,
        "provider_message_id": cleaned_message_id,
        "proposal_version": proposal.version,
        "action_type": proposal.action_type.value,
        "payload": proposal.payload.model_dump(mode="json"),
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def effect_idempotency_key(
    mailbox_id: str,
    provider_message_id: str,
    proposal: ActionProposal,
) -> str:
    """Return the stable SHA-256 key for one intended mailbox effect."""
    return sha256(canonical_effect(mailbox_id, provider_message_id, proposal)).hexdigest()


def _required_identity(value: str, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} cannot be blank")
    if len(cleaned) > 512:
        raise ValueError(f"{field} cannot exceed 512 characters")
    return cleaned
