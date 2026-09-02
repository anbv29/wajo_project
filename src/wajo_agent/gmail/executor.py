"""Gmail implementation of the narrow mailbox-executor capability boundary."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from email.message import EmailMessage
from urllib.parse import quote

from pydantic import JsonValue

from wajo_agent.domain import (
    ActionType,
    AutonomyTier,
    DraftPayload,
    ExecutionCommand,
    ExecutionResult,
    ExecutionState,
    LabelPayload,
    MessagePayload,
    ReplyPayload,
)
from wajo_agent.execution import ExecutorOutcomeUnknownError, ExecutorUnavailableError
from wajo_agent.gmail.config import GmailAdapterConfig
from wajo_agent.gmail.contracts import (
    GmailAmbiguousError,
    GmailConfigurationError,
    GmailDefiniteError,
    GmailMethod,
    GmailTransport,
)


@dataclass(frozen=True, slots=True)
class GmailOperation:
    """Inspectable REST operation prepared from an already-authorized command."""

    method: GmailMethod
    path: str
    body: dict[str, JsonValue] | None
    operation_kind: str


class GmailMailboxExecutor:
    """Map safe commands to Gmail REST while failing closed by default."""

    def __init__(self, transport: GmailTransport, config: GmailAdapterConfig | None = None) -> None:
        self._transport = transport
        self._config = config or GmailAdapterConfig()

    def execute(self, command: ExecutionCommand) -> ExecutionResult:
        if command.action_type == ActionType.NO_ACTION:
            return _result(
                command,
                state=ExecutionState.SUCCEEDED,
                detail="No Gmail effect was required",
                provider_operation_id=None,
            )

        try:
            operation = self.prepare(command)
        except GmailConfigurationError as exc:
            raise ExecutorUnavailableError("Gmail operation is not configured") from exc
        if not self._config.live_effects_enabled:
            return _result(
                command,
                state=ExecutionState.FAILED_SAFE,
                detail="Gmail dry-run validated the operation; no provider request was sent",
                provider_operation_id=None,
            )

        try:
            response = self._transport.request(
                operation.method,
                operation.path,
                body=operation.body,
            )
        except GmailAmbiguousError as exc:
            raise ExecutorOutcomeUnknownError("Gmail could not prove the mutation result") from exc
        except (GmailDefiniteError, GmailConfigurationError) as exc:
            raise ExecutorUnavailableError("Gmail rejected the mutation before an effect") from exc

        if not 200 <= response.status_code < 300:
            if response.status_code in {408, 429} or response.status_code >= 500:
                raise ExecutorOutcomeUnknownError("Gmail mutation returned an ambiguous status")
            raise ExecutorUnavailableError("Gmail mutation was rejected before an effect")
        provider_id = _response_id(response.body)
        if operation.operation_kind in {"modify", "trash"}:
            expected_id = _message_id(command)
            if provider_id != expected_id:
                raise ExecutorOutcomeUnknownError("Gmail returned a different message identity")
        return _result(
            command,
            state=ExecutionState.SUCCEEDED,
            detail=f"Gmail {operation.operation_kind} completed",
            provider_operation_id=f"gmail:{operation.operation_kind}:{provider_id}",
        )

    def prepare(self, command: ExecutionCommand) -> GmailOperation:
        """Validate adapter-specific constraints and build one exact REST operation."""
        action = command.action_type
        if action == ActionType.MARK_READ:
            return _modify(command, remove_labels=("UNREAD",))
        if action == ActionType.MARK_UNREAD:
            return _modify(command, add_labels=("UNREAD",))
        if action == ActionType.ARCHIVE:
            return _modify(command, remove_labels=("INBOX",))
        if action == ActionType.ADD_LABEL:
            if not isinstance(command.payload, LabelPayload):
                raise ExecutorUnavailableError("Gmail label command has the wrong payload")
            label_id = self._config.label_id_for(command.payload.label)
            return _modify(command, add_labels=(label_id,))
        if action == ActionType.TRASH:
            return GmailOperation(
                method="POST",
                path=f"/messages/{quote(_message_id(command), safe='')}/trash",
                body=None,
                operation_kind="trash",
            )
        if action == ActionType.CREATE_DRAFT:
            if not isinstance(command.payload, DraftPayload):
                raise ExecutorUnavailableError("Gmail draft command has the wrong payload")
            raw = _raw_message(
                command.payload.recipients,
                command.payload.subject,
                command.payload.body,
            )
            return GmailOperation(
                method="POST",
                path="/drafts",
                body={"message": {"raw": raw}},
                operation_kind="draft_create",
            )
        if action == ActionType.SEND_REPLY:
            return self._send(command)
        raise ExecutorUnavailableError(f"Gmail adapter does not implement {action.value}")

    def _send(self, command: ExecutionCommand) -> GmailOperation:
        if not isinstance(command.payload, ReplyPayload):
            raise ExecutorUnavailableError("Gmail send command has the wrong payload")
        if command.authorized_tier != AutonomyTier.ASK or command.approval_id is None:
            raise ExecutorUnavailableError("Gmail send requires exact approval authority")
        if not command.payload.recipients:
            raise ExecutorUnavailableError("Gmail send has no recipient")
        if any(
            not self._config.allows_recipient(recipient) for recipient in command.payload.recipients
        ):
            raise ExecutorUnavailableError("Gmail recipient is outside the configured allowlist")
        raw = _raw_message(
            command.payload.recipients,
            command.payload.subject,
            command.payload.body,
        )
        return GmailOperation(
            method="POST",
            path="/messages/send",
            body={"raw": raw},
            operation_kind="send",
        )


def _modify(
    command: ExecutionCommand,
    *,
    add_labels: tuple[str, ...] = (),
    remove_labels: tuple[str, ...] = (),
) -> GmailOperation:
    body: dict[str, JsonValue] = {}
    if add_labels:
        body["addLabelIds"] = list(add_labels)
    if remove_labels:
        body["removeLabelIds"] = list(remove_labels)
    return GmailOperation(
        method="POST",
        path=f"/messages/{quote(_message_id(command), safe='')}/modify",
        body=body,
        operation_kind="modify",
    )


def _message_id(command: ExecutionCommand) -> str:
    if not isinstance(command.payload, (MessagePayload, LabelPayload)):
        raise ExecutorUnavailableError("Gmail message action has the wrong payload")
    return command.payload.message_id


def _raw_message(recipients: tuple[str, ...], subject: str, body: str) -> str:
    message = EmailMessage()
    if recipients:
        message["To"] = ", ".join(recipients)
    if subject:
        message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")


def _response_id(body: dict[str, JsonValue]) -> str:
    value = body.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ExecutorOutcomeUnknownError("Gmail success response omitted its provider identity")
    return value.strip()


def _result(
    command: ExecutionCommand,
    *,
    state: ExecutionState,
    detail: str,
    provider_operation_id: str | None,
) -> ExecutionResult:
    return ExecutionResult(
        execution_id=command.execution_id,
        command_id=command.command_id,
        idempotency_key=command.idempotency_key,
        proposal_id=command.proposal_id,
        state=state,
        detail=detail,
        provider_operation_id=provider_operation_id,
    )
