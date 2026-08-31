from __future__ import annotations

import re

from wajo_agent.domain import EmailEnvelope, RiskAssessment

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(r"\b(ignore|disregard|override)\b.{0,40}\b(instruction|prompt|rule)s?\b", re.I),
    ),
    (
        "fake_system_role",
        re.compile(r"\b(system|developer|administrator)\s*(message|instruction|override)\b", re.I),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(reveal|print|send|forward)\b.{0,50}\b"
            r"(secret|api key|password|token|system prompt)\b",
            re.I,
        ),
    ),
    ("fake_approval", re.compile(r"\b(approved by user|approval token|already approved)\b", re.I)),
)

SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credentials",
        re.compile(r"\b(password|passcode|one[- ]time code|otp|api key|secret token)\b", re.I),
    ),
    (
        "account_recovery",
        re.compile(r"\b(reset your password|account recovery|recover your account)\b", re.I),
    ),
    ("banking", re.compile(r"\b(bank account|routing number|wire transfer|credit card)\b", re.I)),
    ("payment", re.compile(r"\b(pay|payment|purchase|invoice transfer)\b", re.I)),
    (
        "legal_commitment",
        re.compile(r"\b(sign|accept)\b.{0,30}\b(contract|agreement|terms)\b", re.I),
    ),
)


class RiskScanner:
    def scan(self, email: EmailEnvelope, *, normalized_changed: bool = False) -> RiskAssessment:
        content = f"{email.subject}\n{email.body_text}"
        suspicious = tuple(name for name, pattern in INJECTION_PATTERNS if pattern.search(content))
        sensitive = frozenset(
            name for name, pattern in SENSITIVE_PATTERNS if pattern.search(content)
        )
        return RiskAssessment(
            injection_detected=bool(suspicious),
            sensitive_categories=sensitive,
            suspicious_patterns=suspicious,
            normalized_changed=normalized_changed,
        )
