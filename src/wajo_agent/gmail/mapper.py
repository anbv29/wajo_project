"""Convert untrusted Gmail message JSON into the agent's inert observation model."""

from __future__ import annotations

import base64
import binascii
from datetime import UTC, datetime
from email.utils import getaddresses

from pydantic import JsonValue, ValidationError

from wajo_agent.domain import AttachmentMetadata, EmailEnvelope, SenderBucket
from wajo_agent.gmail.contracts import GmailMessageError


def map_gmail_message(message: dict[str, JsonValue]) -> EmailEnvelope:
    """Map a Gmail `format=full` response without downloading attachment bytes."""
    try:
        message_id = _required_string(message, "id")
        thread_id = _optional_string(message.get("threadId"))
        payload = _object(message.get("payload"), "payload")
        headers = _headers(payload.get("headers"))
        sender = _required_header(headers, "from")
        recipients = tuple(
            address
            for _, address in getaddresses([headers.get("to", ""), headers.get("cc", "")])
            if address.strip()
        )
        plain_parts: list[str] = []
        html_parts: list[str] = []
        attachments: list[AttachmentMetadata] = []
        _walk_parts(payload, plain_parts, html_parts, attachments)
        received_at = _internal_date(message.get("internalDate"))
        return EmailEnvelope(
            provider_message_id=message_id,
            provider_thread_id=thread_id,
            sender=sender,
            recipients=tuple(dict.fromkeys(recipients)),
            subject=headers.get("subject", ""),
            body_text="\n".join(part for part in plain_parts if part),
            body_html="\n".join(part for part in html_parts if part) or None,
            received_at=received_at,
            sender_bucket=SenderBucket.UNKNOWN,
            source="gmail",
            attachments=tuple(attachments),
        )
    except (KeyError, TypeError, ValueError, ValidationError, binascii.Error) as exc:
        raise GmailMessageError("Gmail message response is malformed") from exc


def _walk_parts(
    part: dict[str, JsonValue],
    plain_parts: list[str],
    html_parts: list[str],
    attachments: list[AttachmentMetadata],
) -> None:
    mime_type = _optional_string(part.get("mimeType")) or "application/octet-stream"
    filename = _optional_string(part.get("filename"))
    body = _object(part.get("body", {}), "part body")
    attachment_id = _optional_string(body.get("attachmentId"))
    size = _nonnegative_integer(body.get("size", 0), "attachment size")

    if filename:
        attachments.append(
            AttachmentMetadata(
                filename=filename,
                content_type=mime_type,
                size_bytes=size,
                provider_attachment_id=attachment_id,
                is_inline=_is_inline(part.get("headers")),
            )
        )
    elif mime_type in {"text/plain", "text/html"}:
        encoded = _optional_string(body.get("data"))
        if encoded:
            text = _decode_body(encoded)
            (plain_parts if mime_type == "text/plain" else html_parts).append(text)

    nested = part.get("parts", [])
    if not isinstance(nested, list):
        raise TypeError("Gmail MIME parts must be a list")
    for child in nested:
        _walk_parts(_object(child, "MIME part"), plain_parts, html_parts, attachments)


def _headers(value: JsonValue | None) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    result: dict[str, str] = {}
    for item in value:
        header = _object(item, "header")
        name = _required_string(header, "name").casefold()
        header_value = _required_string(header, "value")
        if name not in result:
            result[name] = header_value
    return result


def _is_inline(value: JsonValue | None) -> bool:
    headers = _headers(value)
    disposition = headers.get("content-disposition", "").casefold()
    return disposition.startswith("inline")


def _decode_body(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    return decoded.decode("utf-8", errors="replace")


def _internal_date(value: JsonValue | None) -> datetime:
    if not isinstance(value, str) or not value.isdigit():
        raise ValueError("Gmail internalDate is missing")
    milliseconds = int(value)
    if milliseconds < 0:
        raise ValueError("Gmail internalDate cannot be negative")
    return datetime.fromtimestamp(milliseconds / 1_000, tz=UTC)


def _required_header(headers: dict[str, str], name: str) -> str:
    value = headers.get(name, "").strip()
    if not value:
        raise ValueError(f"Gmail {name} header is missing")
    return value


def _required_string(values: dict[str, JsonValue], key: str) -> str:
    value = _optional_string(values.get(key))
    if value is None:
        raise ValueError(f"Gmail {key} is missing")
    return value


def _optional_string(value: JsonValue | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _nonnegative_integer(value: JsonValue | None, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Gmail {label} must be a nonnegative integer")
    return value


def _object(value: JsonValue | None, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError(f"Gmail {label} must be an object")
    return value
