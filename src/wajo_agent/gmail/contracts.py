"""Small, injectable contracts around the Gmail REST boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import JsonValue

GmailMethod = Literal["GET", "POST"]
GMAIL_MODIFY_SCOPE = "https://www.googleapis.com/auth/gmail.modify"


class GmailError(RuntimeError):
    """Base class for safe Gmail adapter failures."""


class GmailConfigurationError(GmailError):
    """Local Gmail settings are absent or unsafe."""


class GmailDefiniteError(GmailError):
    """Gmail rejected the request before an effect could occur."""


class GmailAmbiguousError(GmailError):
    """The caller cannot prove whether a mutating request took effect."""


class GmailMessageError(GmailError):
    """A Gmail message response could not be mapped safely."""


@dataclass(frozen=True, slots=True)
class GmailResponse:
    status_code: int
    body: dict[str, JsonValue]
    request_id: str | None = None


@runtime_checkable
class GmailAccessTokenProvider(Protocol):
    """Return a short-lived OAuth access token without exposing it to the adapter."""

    def access_token(self) -> str: ...


@runtime_checkable
class GmailTransport(Protocol):
    """Execute one authenticated Gmail request."""

    def request(
        self,
        method: GmailMethod,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, JsonValue] | None = None,
    ) -> GmailResponse: ...
