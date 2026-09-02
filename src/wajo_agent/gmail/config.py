"""Fail-closed Gmail adapter configuration."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr

from wajo_agent.gmail.contracts import GmailConfigurationError


@dataclass(frozen=True, slots=True)
class GmailAdapterConfig:
    """Gates live effects and narrowly configures recipients and label IDs."""

    enabled: bool = False
    dry_run: bool = True
    allowed_recipients: tuple[str, ...] = ()
    label_ids: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        recipients = tuple(_address(value) for value in self.allowed_recipients)
        if len(recipients) != len(set(recipients)):
            raise GmailConfigurationError("allowed Gmail recipients must be unique")
        labels = tuple(
            (name.strip().casefold(), label_id.strip()) for name, label_id in self.label_ids
        )
        if any(not name or not label_id for name, label_id in labels):
            raise GmailConfigurationError("Gmail label mappings cannot be blank")
        if len(labels) != len({name for name, _ in labels}):
            raise GmailConfigurationError("Gmail label names must be unique")
        object.__setattr__(self, "allowed_recipients", recipients)
        object.__setattr__(self, "label_ids", labels)

    @property
    def live_effects_enabled(self) -> bool:
        return self.enabled and not self.dry_run

    def allows_recipient(self, value: str) -> bool:
        return _address(value) in self.allowed_recipients

    def label_id_for(self, label_name: str) -> str:
        wanted = label_name.strip().casefold()
        for name, label_id in self.label_ids:
            if name == wanted:
                return label_id
        raise GmailConfigurationError(f"Gmail label is not configured: {label_name.strip()}")


def _address(value: str) -> str:
    _, parsed = parseaddr(value.strip())
    cleaned = (parsed or value).strip().casefold()
    if not cleaned or "@" not in cleaned:
        raise GmailConfigurationError("Gmail recipient must be an email address")
    return cleaned
