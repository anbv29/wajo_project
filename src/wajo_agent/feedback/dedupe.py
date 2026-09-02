"""Stable semantic identity for exactly-once user feedback."""

from __future__ import annotations

import json
from hashlib import sha256

from wajo_agent.domain import ActionProposal, Decision, FeedbackType

FEEDBACK_KEY_SCHEMA_VERSION = 1


def feedback_dedupe_key(
    decision: Decision,
    proposal: ActionProposal,
    feedback: FeedbackType,
) -> str:
    """Identify one feedback meaning independent of retries or UI request IDs."""
    document = {
        "schema_version": FEEDBACK_KEY_SCHEMA_VERSION,
        "decision_id": decision.decision_id,
        "proposal_id": proposal.proposal_id,
        "proposal_version": proposal.version,
        "feedback_type": feedback.value,
    }
    encoded = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()
