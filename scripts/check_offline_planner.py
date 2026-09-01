"""Fast scenario checks for the deterministic offline planner."""

from __future__ import annotations

from wajo_agent.domain import (
    ActionType,
    AttachmentMetadata,
    EmailEnvelope,
    Intent,
    PlannerRequest,
    SenderBucket,
)
from wajo_agent.normalization import normalize_email
from wajo_agent.planning import (
    OfflinePlanner,
    Planner,
    PlannerContractError,
    build_planner_request,
    validate_planner_output,
)


def _request(
    subject: str,
    body: str = "",
    *,
    sender_bucket: SenderBucket = SenderBucket.UNKNOWN,
) -> PlannerRequest:
    email = EmailEnvelope(
        provider_message_id=f"message-{subject or 'empty'}",
        sender="Sender <sender@example.com>",
        subject=subject,
        body_text=body,
        sender_bucket=sender_bucket,
    )
    normalized, report = normalize_email(email)
    return build_planner_request(normalized, normalization_changed=report.changed)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    checks = 0
    planner = OfflinePlanner()
    _require(isinstance(planner, Planner), "offline planner does not satisfy Planner")
    checks += 1

    scenarios = (
        (
            _request("Weekly newsletter", "Reset your password to continue"),
            Intent.ACCOUNT_RECOVERY,
            ActionType.NO_ACTION,
            "account recovery precedence",
        ),
        (
            _request("Legal notice", "Please sign the contract"),
            Intent.LEGAL,
            ActionType.NO_ACTION,
            "legal handling",
        ),
        (
            _request("Your receipt", "Payment received for order 42"),
            Intent.RECEIPT,
            ActionType.ADD_LABEL,
            "receipt handling",
        ),
        (
            _request("Payment required", "Send a wire transfer today"),
            Intent.FINANCIAL,
            ActionType.NO_ACTION,
            "financial handling",
        ),
        (
            _request("You are a winner", "Claim your prize now"),
            Intent.SPAM,
            ActionType.TRASH,
            "spam handling",
        ),
        (
            _request("Weekly newsletter", "Here is this week's digest"),
            Intent.NEWSLETTER,
            ActionType.ARCHIVE,
            "newsletter handling",
        ),
        (
            _request("Meeting invitation", "Can we schedule a meeting?"),
            Intent.MEETING,
            ActionType.CREATE_DRAFT,
            "meeting handling",
        ),
        (
            _request("Project update", "Could you please review the attached notes?"),
            Intent.REQUEST,
            ActionType.CREATE_DRAFT,
            "request handling",
        ),
        (
            _request("Hello", "Hope you are well", sender_bucket=SenderBucket.KNOWN_PERSON),
            Intent.PERSONAL,
            ActionType.NO_ACTION,
            "personal handling",
        ),
        (
            _request("Service update", "The maintenance window has completed"),
            Intent.INFORMATIONAL,
            ActionType.MARK_READ,
            "informational handling",
        ),
        (
            _request("Standard policy update", "The standard process is unchanged"),
            Intent.INFORMATIONAL,
            ActionType.MARK_READ,
            "word-boundary false-positive control",
        ),
    )
    for request, expected_intent, expected_action, label in scenarios:
        output = planner.plan(request)
        validate_planner_output(request, output)
        _require(output.intent == expected_intent, f"wrong intent for {label}")
        _require(output.action_type == expected_action, f"wrong action for {label}")
        checks += 2

    attachment_email = EmailEnvelope(
        provider_message_id="attachment-only",
        sender="sender@example.com",
        attachments=(
            AttachmentMetadata(filename="notes.txt", content_type="text/plain", size_bytes=10),
        ),
    )
    attachment_normalized, attachment_report = normalize_email(attachment_email)
    attachment_request = build_planner_request(
        attachment_normalized,
        normalization_changed=attachment_report.changed,
    )
    attachment_output = planner.plan(attachment_request)
    _require(attachment_output.intent == Intent.UNKNOWN, "attachment-only email was guessed")
    _require(attachment_output.action_type == ActionType.NO_ACTION, "attachment was acted upon")
    checks += 2

    deterministic_request = _request("Weekly newsletter", "Weekly digest")
    _require(
        planner.plan(deterministic_request) == planner.plan(deterministic_request),
        "same request produced different outputs",
    )
    checks += 1

    fallback_request = deterministic_request.model_copy(
        update={"allowed_actions": (ActionType.NO_ACTION,)}
    )
    fallback = planner.plan(fallback_request)
    _require(fallback.action_type == ActionType.NO_ACTION, "disallowed action did not fail safe")
    _require(bool(fallback.uncertainty_reasons), "safe fallback was not explained")
    checks += 2

    impossible_request = deterministic_request.model_copy(
        update={"allowed_actions": (ActionType.MARK_READ,)}
    )
    try:
        planner.plan(impossible_request)
    except PlannerContractError:
        checks += 1
    else:
        raise RuntimeError("planner invented an unsafe fallback")

    meeting_output = planner.plan(_request("Meeting invitation"))
    _require(
        meeting_output.action_type != ActionType.SEND_REPLY,
        "offline planner attempted to send externally",
    )
    _require(not hasattr(meeting_output, "tier"), "planner output contains an autonomy decision")
    checks += 2

    print(f"Offline planner checks passed: {checks}")


if __name__ == "__main__":
    main()
