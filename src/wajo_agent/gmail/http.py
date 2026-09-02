"""Dependency-free authenticated Gmail REST transport."""

from __future__ import annotations

import json
import os
from typing import Never
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import JsonValue, TypeAdapter, ValidationError

from wajo_agent.gmail.contracts import (
    GmailAccessTokenProvider,
    GmailAmbiguousError,
    GmailConfigurationError,
    GmailDefiniteError,
    GmailMethod,
    GmailResponse,
)

GMAIL_API_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class EnvironmentAccessTokenProvider:
    """Read a short-lived OAuth token from an environment variable on each call."""

    def __init__(self, variable: str = "WAJO_GMAIL_ACCESS_TOKEN") -> None:
        cleaned = variable.strip()
        if not cleaned:
            raise GmailConfigurationError("Gmail token variable cannot be blank")
        self._variable = cleaned

    def access_token(self) -> str:
        token = os.getenv(self._variable, "").strip()
        if not token:
            raise GmailConfigurationError(
                f"Gmail OAuth access token is missing from {self._variable}"
            )
        return token


class GmailHttpTransport:
    """Send JSON requests without logging tokens, bodies, or email content."""

    def __init__(
        self,
        token_provider: GmailAccessTokenProvider,
        *,
        timeout_seconds: float = 15.0,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise GmailConfigurationError("Gmail timeout must be between 0 and 60 seconds")
        self._tokens = token_provider
        self._timeout = timeout_seconds
        self._base_url = GMAIL_API_BASE_URL

    def request(
        self,
        method: GmailMethod,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, JsonValue] | None = None,
    ) -> GmailResponse:
        normalized_path = _safe_path(path)
        encoded_query = f"?{urlencode(query)}" if query else ""
        url = f"{self._base_url}{normalized_path}{encoded_query}"
        data = None
        if body is not None:
            data = json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._tokens.access_token()}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                status = response.status
                raw = response.read()
                request_id = response.headers.get("x-guploader-uploadid")
        except HTTPError as exc:
            _raise_http_error(method, exc.code)
        except (TimeoutError, URLError) as exc:
            _raise_network_error(method, exc)

        try:
            parsed = JSON_OBJECT.validate_json(raw) if raw else {}
        except ValidationError as exc:
            if method == "POST":
                raise GmailAmbiguousError(
                    "Gmail returned an invalid response after a mutation"
                ) from exc
            raise GmailDefiniteError("Gmail returned an invalid read response") from exc
        return GmailResponse(status_code=status, body=parsed, request_id=request_id)


def _safe_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned.startswith("/") or ".." in cleaned or "://" in cleaned:
        raise GmailConfigurationError("Gmail request path is unsafe")
    return cleaned


def _raise_http_error(method: GmailMethod, status: int) -> Never:
    if method == "POST" and (status == 408 or status == 429 or status >= 500):
        raise GmailAmbiguousError(f"Gmail mutation returned ambiguous HTTP {status}")
    raise GmailDefiniteError(f"Gmail request was rejected with HTTP {status}")


def _raise_network_error(method: GmailMethod, exc: Exception) -> Never:
    if method == "POST":
        raise GmailAmbiguousError("Gmail mutation ended without a provable result") from exc
    raise GmailDefiniteError("Gmail read request failed before a result was received") from exc
