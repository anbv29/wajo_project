"""Build narrow, stable identities for contextual preference learning."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from contextlib import suppress
from email.utils import parseaddr
from hashlib import sha256

from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    DraftPayload,
    EmailEnvelope,
    LabelPayload,
    PreferenceContext,
    RecipientScope,
    ReplyPayload,
)


def normalize_email_address(value: str) -> str:
    """Return one stable comparison form without changing address semantics."""
    _, parsed = parseaddr(value)
    candidate = parsed or value
    normalized = unicodedata.normalize("NFKC", candidate).strip().casefold()

    local_part, separator, domain = normalized.rpartition("@")
    if separator and local_part and domain:
        try:
            ascii_domain = domain.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_domain = domain
        return f"{local_part}@{ascii_domain}"
    return normalized


def identity_hash(value: str) -> str:
    """Hash normalized identity so storage keys do not expose an email address."""
    normalized = normalize_email_address(value)
    return sha256(normalized.encode("utf-8")).hexdigest()


def _normalized_domains(domains: Iterable[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for domain in domains:
        cleaned = unicodedata.normalize("NFKC", domain).strip().casefold().lstrip("@")
        if not cleaned:
            continue
        with suppress(UnicodeError):
            cleaned = cleaned.encode("idna").decode("ascii")
        normalized.add(cleaned)
    return frozenset(normalized)


def _proposal_recipients(proposal: ActionProposal) -> tuple[str, ...] | None:
    payload = proposal.payload
    if isinstance(payload, (DraftPayload, ReplyPayload)):
        return payload.recipients
    return None


def recipient_scope_for(
    proposal: ActionProposal,
    *,
    internal_domains: Iterable[str] = (),
) -> RecipientScope:
    """Classify who an action targets without storing recipient addresses."""
    recipients = _proposal_recipients(proposal)
    if recipients is None:
        if proposal.action_type == ActionType.NO_ACTION:
            return RecipientScope.NONE
        return RecipientScope.INTERNAL_MAILBOX
    if not recipients:
        return RecipientScope.NONE

    unique_recipients = tuple(
        dict.fromkeys(normalize_email_address(recipient) for recipient in recipients)
    )
    owned_domains = _normalized_domains(internal_domains)
    recipient_domains = {
        address.rpartition("@")[2] if "@" in address else "" for address in unique_recipients
    }
    internal_count = sum(domain in owned_domains for domain in recipient_domains)

    if internal_count == len(recipient_domains):
        return RecipientScope.INTERNAL_RECIPIENTS
    if internal_count:
        return RecipientScope.MIXED_RECIPIENTS
    if len(unique_recipients) == 1:
        return RecipientScope.EXTERNAL_SINGLE
    return RecipientScope.EXTERNAL_MULTIPLE


def action_variant_for(proposal: ActionProposal) -> str:
    """Separate materially different variants of the same action."""
    if isinstance(proposal.payload, LabelPayload):
        normalized_label = unicodedata.normalize("NFKC", proposal.payload.label).strip().casefold()
        label_hash = sha256(normalized_label.encode("utf-8")).hexdigest()
        return f"label:{label_hash}"
    return "default"


def build_preference_context(
    email: EmailEnvelope,
    proposal: ActionProposal,
    *,
    internal_domains: Iterable[str] = (),
) -> PreferenceContext:
    """Construct the only context identity accepted by preference learning."""
    if proposal.email_id != email.email_id:
        raise ValueError("proposal and email must refer to the same observation")

    return PreferenceContext(
        action_type=proposal.action_type,
        intent=proposal.intent,
        sender_bucket=email.sender_bucket,
        sender_identity_hash=identity_hash(email.sender),
        recipient_scope=recipient_scope_for(proposal, internal_domains=internal_domains),
        action_variant=action_variant_for(proposal),
    )
