"""Offline contract checks for the optional Gmail reader and executor adapter."""

from __future__ import annotations

import base64
import os
from email import policy
from email.parser import BytesParser

from pydantic import JsonValue

from wajo_agent.domain import (
    ActionType,
    AutonomyTier,
    DraftPayload,
    ExecutionCommand,
    ExecutionState,
    LabelPayload,
    MessagePayload,
    NoActionPayload,
    ReplyPayload,
)
from wajo_agent.execution import (
    ExecutorOutcomeUnknownError,
    ExecutorUnavailableError,
    execute_checked,
)
from wajo_agent.gmail import (
    GMAIL_MODIFY_SCOPE,
    EnvironmentAccessTokenProvider,
    GmailAdapterConfig,
    GmailAmbiguousError,
    GmailConfigurationError,
    GmailDefiniteError,
    GmailMailboxExecutor,
    GmailMessageError,
    GmailReader,
    GmailResponse,
    GmailTransport,
    map_gmail_message,
)
from wajo_agent.gmail.contracts import GmailMethod


class FakeGmailTransport:
    def __init__(self, responses: list[GmailResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[
            tuple[
                GmailMethod,
                str,
                dict[str, str] | None,
                dict[str, JsonValue] | None,
            ]
        ] = []
        self.failure: Exception | None = None

    def request(
        self,
        method: GmailMethod,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, JsonValue] | None = None,
    ) -> GmailResponse:
        self.requests.append((method, path, query, body))
        if self.failure is not None:
            raise self.failure
        if not self.responses:
            raise RuntimeError("fake Gmail response queue is empty")
        return self.responses.pop(0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _gmail_message(message_id: str = "gmail-message-1") -> dict[str, JsonValue]:
    return {
        "id": message_id,
        "threadId": "gmail-thread-1",
        "internalDate": "1767225600000",
        "payload": {
            "mimeType": "multipart/mixed",
            "filename": "",
            "headers": [
                {"name": "From", "value": "Sender <sender@example.com>"},
                {"name": "To", "value": "User <user@example.com>"},
                {"name": "Cc", "value": "copy@example.com"},
                {"name": "Subject", "value": "Gmail adapter check"},
            ],
            "body": {"size": 0},
            "parts": [
                {
                    "mimeType": "text/plain",
                    "filename": "",
                    "headers": [],
                    "body": {"size": 18, "data": _encoded("Hello from Gmail.")},
                },
                {
                    "mimeType": "text/html",
                    "filename": "",
                    "headers": [],
                    "body": {"size": 24, "data": _encoded("<p>Hello from Gmail.</p>")},
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "report.pdf",
                    "headers": [
                        {"name": "Content-Disposition", "value": "attachment"},
                    ],
                    "body": {"size": 2048, "attachmentId": "attachment-1"},
                },
            ],
        },
    }


def _command(
    action: ActionType,
    payload: NoActionPayload | MessagePayload | LabelPayload | DraftPayload | ReplyPayload,
    *,
    tier: AutonomyTier = AutonomyTier.SILENT,
    approval_id: str | None = None,
) -> ExecutionCommand:
    return ExecutionCommand(
        idempotency_key=f"gmail-check-{action.value}",
        email_id="email-1",
        proposal_id=f"proposal-{action.value}",
        proposal_version=1,
        decision_id=f"decision-{action.value}",
        action_type=action,
        payload=payload,
        authorized_tier=tier,
        approval_id=approval_id,
    )


def _archive_command() -> ExecutionCommand:
    return _command(
        ActionType.ARCHIVE,
        MessagePayload(kind=ActionType.ARCHIVE, message_id="gmail-message-1"),
    )


def main() -> None:
    checks = 0
    mapped = map_gmail_message(_gmail_message())
    _require(mapped.provider_message_id == "gmail-message-1", "Gmail ID was lost")
    _require(mapped.provider_thread_id == "gmail-thread-1", "thread ID was lost")
    _require(mapped.sender == "Sender <sender@example.com>", "sender header was lost")
    _require(mapped.recipients == ("user@example.com", "copy@example.com"), "recipients wrong")
    _require(mapped.body_text == "Hello from Gmail.", "plain body decode failed")
    _require(mapped.body_html == "<p>Hello from Gmail.</p>", "HTML body decode failed")
    _require(mapped.source == "gmail", "Gmail source was not marked")
    _require(
        len(mapped.attachments) == 1
        and mapped.attachments[0].provider_attachment_id == "attachment-1"
        and mapped.attachments[0].size_bytes == 2048,
        "attachment metadata was not mapped",
    )
    checks += 8

    malformed = _gmail_message()
    payload = malformed["payload"]
    assert isinstance(payload, dict)
    payload["headers"] = []
    try:
        map_gmail_message(malformed)
    except GmailMessageError:
        checks += 1
    else:
        raise RuntimeError("Gmail mapper accepted a message without a sender")

    reader_transport = FakeGmailTransport([GmailResponse(200, _gmail_message())])
    _require(isinstance(reader_transport, GmailTransport), "fake does not satisfy transport")
    read = GmailReader(reader_transport).get_message("gmail-message-1")
    _require(read.provider_message_id == "gmail-message-1", "reader returned wrong message")
    _require(
        reader_transport.requests
        == [("GET", "/messages/gmail-message-1", {"format": "full"}, None)],
        "reader used the wrong Gmail endpoint",
    )
    checks += 3

    mismatch_transport = FakeGmailTransport([GmailResponse(200, _gmail_message("another-message"))])
    try:
        GmailReader(mismatch_transport).get_message("gmail-message-1")
    except GmailMessageError:
        checks += 1
    else:
        raise RuntimeError("reader accepted a different Gmail message")

    dry_transport = FakeGmailTransport()
    dry_executor = GmailMailboxExecutor(dry_transport)
    dry_result = execute_checked(dry_executor, _archive_command())
    _require(dry_result.state == ExecutionState.FAILED_SAFE, "dry run claimed a real effect")
    _require(not dry_transport.requests, "dry run called Gmail")
    checks += 2

    config = GmailAdapterConfig(
        enabled=True,
        dry_run=False,
        allowed_recipients=("test-recipient@example.com",),
        label_ids=(("Receipts", "Label_Receipts"),),
    )
    live_transport = FakeGmailTransport(
        [GmailResponse(200, {"id": "gmail-message-1"}, "gmail-request-1")]
    )
    live_executor = GmailMailboxExecutor(live_transport, config)
    archive_result = execute_checked(live_executor, _archive_command())
    _require(archive_result.state == ExecutionState.SUCCEEDED, "live archive failed")
    _require(
        live_transport.requests[0]
        == (
            "POST",
            "/messages/gmail-message-1/modify",
            None,
            {"removeLabelIds": ["INBOX"]},
        ),
        "archive request shape was wrong",
    )
    _require(
        archive_result.provider_operation_id == "gmail:modify:gmail-message-1",
        "provider reconciliation ID was lost",
    )
    checks += 3

    read_operation = live_executor.prepare(
        _command(
            ActionType.MARK_READ,
            MessagePayload(kind=ActionType.MARK_READ, message_id="gmail-message-1"),
        )
    )
    unread_operation = live_executor.prepare(
        _command(
            ActionType.MARK_UNREAD,
            MessagePayload(kind=ActionType.MARK_UNREAD, message_id="gmail-message-1"),
        )
    )
    label_operation = live_executor.prepare(
        _command(
            ActionType.ADD_LABEL,
            LabelPayload(message_id="gmail-message-1", label="Receipts"),
        )
    )
    trash_operation = live_executor.prepare(
        _command(
            ActionType.TRASH,
            MessagePayload(kind=ActionType.TRASH, message_id="gmail-message-1"),
            tier=AutonomyTier.ASK,
            approval_id="approval-trash",
        )
    )
    _require(read_operation.body == {"removeLabelIds": ["UNREAD"]}, "mark-read wrong")
    _require(unread_operation.body == {"addLabelIds": ["UNREAD"]}, "mark-unread wrong")
    _require(label_operation.body == {"addLabelIds": ["Label_Receipts"]}, "label wrong")
    _require(
        trash_operation.path.endswith("/trash") and trash_operation.body is None, "trash wrong"
    )
    checks += 4

    draft_operation = live_executor.prepare(
        _command(
            ActionType.CREATE_DRAFT,
            DraftPayload(
                recipients=("test-recipient@example.com",),
                subject="Draft subject",
                body="Draft body",
            ),
            tier=AutonomyTier.NOTIFY,
        )
    )
    _require(draft_operation.path == "/drafts", "draft used the wrong endpoint")
    _require(draft_operation.body is not None, "draft omitted message body")
    checks += 2

    send_command = _command(
        ActionType.SEND_REPLY,
        ReplyPayload(
            kind=ActionType.SEND_REPLY,
            recipients=("test-recipient@example.com",),
            subject="Approved reply",
            body="Approved body",
        ),
        tier=AutonomyTier.ASK,
        approval_id="approval-send",
    )
    send_operation = live_executor.prepare(send_command)
    _require(send_operation.path == "/messages/send", "send used wrong endpoint")
    assert send_operation.body is not None
    raw = send_operation.body["raw"]
    assert isinstance(raw, str)
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    parsed = BytesParser(policy=policy.default).parsebytes(decoded)
    _require(parsed["To"] == "test-recipient@example.com", "send recipient changed")
    _require(
        parsed.get_body(preferencelist=("plain",)).get_content() == "Approved body\n",
        "body changed",
    )
    checks += 3

    blocked_send = send_command.model_copy(
        update={
            "payload": send_command.payload.model_copy(
                update={"recipients": ("outside@example.net",)}
            )
        }
    )
    try:
        live_executor.prepare(blocked_send)
    except ExecutorUnavailableError:
        checks += 1
    else:
        raise RuntimeError("Gmail sent to a recipient outside the allowlist")

    for failure, expected in (
        (GmailDefiniteError("definite"), ExecutorUnavailableError),
        (GmailAmbiguousError("ambiguous"), ExecutorOutcomeUnknownError),
    ):
        failing_transport = FakeGmailTransport()
        failing_transport.failure = failure
        failing_executor = GmailMailboxExecutor(failing_transport, config)
        try:
            execute_checked(failing_executor, _archive_command())
        except expected:
            checks += 1
        else:
            raise RuntimeError("Gmail transport failure was classified incorrectly")

    wrong_response = FakeGmailTransport([GmailResponse(200, {"id": "wrong-message"})])
    try:
        execute_checked(GmailMailboxExecutor(wrong_response, config), _archive_command())
    except ExecutorOutcomeUnknownError:
        checks += 1
    else:
        raise RuntimeError("Gmail identity mismatch was presented as success")

    for status, expected in (
        (400, ExecutorUnavailableError),
        (503, ExecutorOutcomeUnknownError),
    ):
        status_transport = FakeGmailTransport([GmailResponse(status, {"id": "gmail-message-1"})])
        try:
            execute_checked(GmailMailboxExecutor(status_transport, config), _archive_command())
        except expected:
            checks += 1
        else:
            raise RuntimeError("non-success Gmail status was classified incorrectly")

    read_status = FakeGmailTransport([GmailResponse(500, _gmail_message())])
    try:
        GmailReader(read_status).get_message("gmail-message-1")
    except GmailMessageError:
        checks += 1
    else:
        raise RuntimeError("Gmail reader accepted a server-error response")

    no_action = _command(ActionType.NO_ACTION, NoActionPayload())
    no_action_result = execute_checked(GmailMailboxExecutor(FakeGmailTransport()), no_action)
    _require(no_action_result.state == ExecutionState.SUCCEEDED, "no-action required Gmail")
    checks += 1

    unsupported = _command(
        ActionType.FORWARD,
        ReplyPayload(
            kind=ActionType.FORWARD,
            recipients=("test-recipient@example.com",),
            subject="Forward",
            body="Not implemented",
        ),
        tier=AutonomyTier.ASK,
        approval_id="approval-forward",
    )
    try:
        live_executor.prepare(unsupported)
    except ExecutorUnavailableError:
        checks += 1
    else:
        raise RuntimeError("unsupported Gmail action did not fail closed")

    token_variable = "WAJO_TEST_MISSING_GMAIL_TOKEN"
    os.environ.pop(token_variable, None)
    try:
        EnvironmentAccessTokenProvider(token_variable).access_token()
    except GmailConfigurationError:
        checks += 1
    else:
        raise RuntimeError("missing Gmail token was accepted")

    _require(GMAIL_MODIFY_SCOPE.endswith("/gmail.modify"), "least-privilege scope changed")
    _require("mail.google.com" not in GMAIL_MODIFY_SCOPE, "full-mailbox scope was requested")
    checks += 2

    print(f"Gmail checks passed: {checks}")


if __name__ == "__main__":
    main()
