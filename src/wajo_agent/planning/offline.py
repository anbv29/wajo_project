"""Deterministic, network-free planner for fixtures, demos, and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from email.utils import parseaddr

from wajo_agent.domain import (
    ActionPayload,
    ActionType,
    DraftPayload,
    Intent,
    LabelPayload,
    MessagePayload,
    NoActionPayload,
    PlannerOutput,
    PlannerRequest,
    SenderBucket,
)
from wajo_agent.planning.contracts import PlannerContractError


@dataclass(frozen=True, slots=True)
class IntentRule:
    intent: Intent
    phrases: tuple[str, ...]


INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        Intent.ACCOUNT_RECOVERY,
        (
            "account recovery",
            "recover your account",
            "reset your password",
            "password reset",
            "one-time code",
            "one time code",
            "otp",
        ),
    ),
    IntentRule(
        Intent.LEGAL,
        (
            "sign the contract",
            "sign this agreement",
            "accept the terms",
            "legal notice",
            "non-disclosure agreement",
            "nda",
        ),
    ),
    IntentRule(
        Intent.RECEIPT,
        (
            "receipt",
            "order confirmation",
            "payment received",
            "purchase confirmation",
        ),
    ),
    IntentRule(
        Intent.FINANCIAL,
        (
            "invoice due",
            "payment required",
            "wire transfer",
            "bank account",
            "routing number",
            "credit card",
        ),
    ),
    IntentRule(
        Intent.SPAM,
        (
            "you are a winner",
            "claim your prize",
            "lottery winner",
            "suspected spam",
            "phishing attempt",
        ),
    ),
    IntentRule(
        Intent.NEWSLETTER,
        (
            "newsletter",
            "weekly digest",
            "monthly digest",
            "manage preferences",
            "email preferences",
            "mailing list",
        ),
    ),
    IntentRule(
        Intent.MEETING,
        (
            "meeting invitation",
            "calendar invitation",
            "calendar invite",
            "schedule a meeting",
            "reschedule our meeting",
            "availability for",
        ),
    ),
    IntentRule(
        Intent.REQUEST,
        (
            "could you",
            "can you",
            "please review",
            "please send",
            "please confirm",
            "action required",
        ),
    ),
)


class OfflinePlanner:
    """Produce conservative typed proposals using only local deterministic rules."""

    def plan(self, request: PlannerRequest) -> PlannerOutput:
        intent, evidence = self.classify_intent(request)
        action_type, payload, uncertainty = self._proposal_for(request, intent)

        if action_type not in request.allowed_actions:
            if ActionType.NO_ACTION not in request.allowed_actions:
                raise PlannerContractError(
                    f"offline planner cannot safely replace disallowed {action_type.value}"
                )
            preferred_action = action_type
            action_type = ActionType.NO_ACTION
            payload = NoActionPayload()
            uncertainty = (
                *uncertainty,
                f"preferred action {preferred_action.value} was not allowed; chose no_action",
            )

        return PlannerOutput(
            action_type=action_type,
            intent=intent,
            summary=f"Classified the email as {intent.value} and proposed {action_type.value}.",
            payload=payload,
            evidence=evidence,
            uncertainty_reasons=uncertainty,
        )

    @staticmethod
    def classify_intent(request: PlannerRequest) -> tuple[Intent, tuple[str, ...]]:
        """Apply ordered rules and return only small matched phrases as evidence."""
        email = request.email
        subject = email.subject.casefold()
        body = email.body_text.casefold()

        for rule in INTENT_RULES:
            for phrase in rule.phrases:
                if _contains_phrase(subject, phrase):
                    return rule.intent, (f"subject matched '{phrase}'",)
                if _contains_phrase(body, phrase):
                    return rule.intent, (f"body matched '{phrase}'",)

        if email.sender_bucket == SenderBucket.KNOWN_PERSON:
            return Intent.PERSONAL, ("sender is in the known-person bucket",)
        if not subject and not body:
            return Intent.UNKNOWN, ("email has no visible subject or body text",)
        return Intent.INFORMATIONAL, ("no higher-priority intent rule matched",)

    @staticmethod
    def _proposal_for(
        request: PlannerRequest,
        intent: Intent,
    ) -> tuple[ActionType, ActionPayload, tuple[str, ...]]:
        email = request.email

        if intent in {Intent.ACCOUNT_RECOVERY, Intent.FINANCIAL, Intent.LEGAL}:
            return (
                ActionType.NO_ACTION,
                NoActionPayload(),
                (f"{intent.value} content requires the separate safety policy",),
            )
        if intent == Intent.NEWSLETTER:
            return (
                ActionType.ARCHIVE,
                MessagePayload(
                    kind=ActionType.ARCHIVE,
                    message_id=email.provider_message_id,
                ),
                (),
            )
        if intent == Intent.RECEIPT:
            return (
                ActionType.ADD_LABEL,
                LabelPayload(
                    message_id=email.provider_message_id,
                    label="Receipts",
                ),
                (),
            )
        if intent == Intent.SPAM:
            return (
                ActionType.TRASH,
                MessagePayload(
                    kind=ActionType.TRASH,
                    message_id=email.provider_message_id,
                ),
                ("trash remains subject to its ASK capability floor",),
            )
        if intent in {Intent.MEETING, Intent.REQUEST}:
            return (
                ActionType.CREATE_DRAFT,
                DraftPayload(
                    recipients=(_sender_address(email.sender),),
                    subject=_reply_subject(email.subject),
                    body="Thanks for your message. I will review this and get back to you.",
                ),
                ("offline planner creates a draft but never sends it",),
            )
        if intent == Intent.INFORMATIONAL:
            return (
                ActionType.MARK_READ,
                MessagePayload(
                    kind=ActionType.MARK_READ,
                    message_id=email.provider_message_id,
                ),
                (),
            )
        return ActionType.NO_ACTION, NoActionPayload(), ()


def _sender_address(sender: str) -> str:
    _, parsed = parseaddr(sender)
    return (parsed or sender).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a phrase at word boundaries to avoid substring false positives."""
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def _reply_subject(subject: str) -> str:
    cleaned = subject.strip()
    if not cleaned:
        return "Draft response"
    if cleaned.casefold().startswith("re:"):
        return cleaned
    return f"Re: {cleaned}"
