"""Stable byte representation used to bind approval to one exact proposal."""

from __future__ import annotations

import json
from hashlib import sha256

from wajo_agent.domain import ActionProposal

APPROVAL_PAYLOAD_SCHEMA_VERSION = 1


def canonical_approval_payload(proposal: ActionProposal) -> bytes:
    """Return deterministic UTF-8 bytes for every execution-relevant proposal field."""
    document = {
        "schema_version": APPROVAL_PAYLOAD_SCHEMA_VERSION,
        "proposal_id": proposal.proposal_id,
        "proposal_version": proposal.version,
        "email_id": proposal.email_id,
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


def approval_payload_hash(proposal: ActionProposal) -> str:
    """Return the lowercase SHA-256 digest stored in an approval record."""
    return sha256(canonical_approval_payload(proposal)).hexdigest()
