"""Read-only Gmail observation service."""

from __future__ import annotations

from urllib.parse import quote

from wajo_agent.domain import EmailEnvelope
from wajo_agent.gmail.contracts import GmailMessageError, GmailTransport
from wajo_agent.gmail.mapper import map_gmail_message


class GmailReader:
    """Fetch one full Gmail message and map it into an inert EmailEnvelope."""

    def __init__(self, transport: GmailTransport) -> None:
        self._transport = transport

    def get_message(self, message_id: str) -> EmailEnvelope:
        cleaned = message_id.strip()
        if not cleaned:
            raise GmailMessageError("Gmail message ID cannot be blank")
        response = self._transport.request(
            "GET",
            f"/messages/{quote(cleaned, safe='')}",
            query={"format": "full"},
        )
        if not 200 <= response.status_code < 300:
            raise GmailMessageError(f"Gmail read returned HTTP {response.status_code}")
        mapped = map_gmail_message(response.body)
        if mapped.provider_message_id != cleaned:
            raise GmailMessageError("Gmail returned a different message identity")
        return mapped
