"""Fast checks for preference-context isolation and stable identity."""

from __future__ import annotations

from wajo_agent.domain import (
    ActionProposal,
    ActionType,
    DraftPayload,
    EmailEnvelope,
    Intent,
    LabelPayload,
    MessagePayload,
    RecipientScope,
    SenderBucket,
)
from wajo_agent.learning import build_preference_context


def _email(sender: str) -> EmailEnvelope:
    return EmailEnvelope(
        provider_message_id=f"provider-{sender}",
        sender=sender,
        subject="Context check",
        sender_bucket=SenderBucket.KNOWN_BULK,
    )


def _label_proposal(email: EmailEnvelope, label: str, intent: Intent) -> ActionProposal:
    return ActionProposal(
        email_id=email.email_id,
        action_type=ActionType.ADD_LABEL,
        intent=intent,
        summary="Add a test label",
        payload=LabelPayload(message_id=email.provider_message_id, label=label),
    )


def _draft_proposal(email: EmailEnvelope, recipients: tuple[str, ...]) -> ActionProposal:
    return ActionProposal(
        email_id=email.email_id,
        action_type=ActionType.CREATE_DRAFT,
        intent=Intent.REQUEST,
        summary="Create a test draft",
        payload=DraftPayload(recipients=recipients, body="Draft body"),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    checks = 0
    first_email = _email("Newsletter <NEWS@example.com>")
    same_sender = _email("news@example.com")
    different_sender = _email("updates@example.com")

    first = build_preference_context(
        first_email,
        _label_proposal(first_email, "Newsletter", Intent.NEWSLETTER),
    )
    equivalent = build_preference_context(
        same_sender,
        _label_proposal(same_sender, "newsletter", Intent.NEWSLETTER),
    )
    _require(first.key == equivalent.key, "equivalent context did not produce a stable key")
    checks += 1

    sender_changed = build_preference_context(
        different_sender,
        _label_proposal(different_sender, "Newsletter", Intent.NEWSLETTER),
    )
    _require(first.key != sender_changed.key, "trust leaked across different senders")
    checks += 1

    intent_changed = build_preference_context(
        first_email,
        _label_proposal(first_email, "Newsletter", Intent.PERSONAL),
    )
    _require(first.key != intent_changed.key, "trust leaked across different intents")
    checks += 1

    label_changed = build_preference_context(
        first_email,
        _label_proposal(first_email, "Finance", Intent.NEWSLETTER),
    )
    _require(first.key != label_changed.key, "trust leaked across different labels")
    checks += 1

    archive = ActionProposal(
        email_id=first_email.email_id,
        action_type=ActionType.ARCHIVE,
        intent=Intent.NEWSLETTER,
        summary="Archive a newsletter",
        payload=MessagePayload(
            kind=ActionType.ARCHIVE,
            message_id=first_email.provider_message_id,
        ),
    )
    action_changed = build_preference_context(first_email, archive)
    _require(first.key != action_changed.key, "trust leaked across different actions")
    checks += 1

    _require(first.recipient_scope == RecipientScope.INTERNAL_MAILBOX, "wrong mailbox scope")
    checks += 1
    _require("news@example.com" not in first.key, "context key exposed sender address")
    checks += 1
    _require("newsletter" not in first.key, "context key exposed the label")
    checks += 1

    scope_examples = (
        (("teammate@company.com",), RecipientScope.INTERNAL_RECIPIENTS),
        (("person@outside.com",), RecipientScope.EXTERNAL_SINGLE),
        (
            ("first@outside.com", "second@outside.com"),
            RecipientScope.EXTERNAL_MULTIPLE,
        ),
        (
            ("teammate@company.com", "person@outside.com"),
            RecipientScope.MIXED_RECIPIENTS,
        ),
        ((), RecipientScope.NONE),
    )
    for recipients, expected_scope in scope_examples:
        context = build_preference_context(
            first_email,
            _draft_proposal(first_email, recipients),
            internal_domains=("company.com",),
        )
        _require(context.recipient_scope == expected_scope, "recipient scope was incorrect")
        checks += 1

    mismatched_email = _email("news@example.com")
    try:
        build_preference_context(
            mismatched_email,
            _label_proposal(first_email, "Newsletter", Intent.NEWSLETTER),
        )
    except ValueError:
        checks += 1
    else:
        raise RuntimeError("context builder accepted a proposal for a different email")

    print(f"Context checks passed: {checks}")


if __name__ == "__main__":
    main()
