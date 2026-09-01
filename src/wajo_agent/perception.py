from __future__ import annotations

import re
from dataclasses import dataclass

from wajo_agent.domain import (
    EmailEnvelope,
    InjectionSignal,
    NormalizationFlag,
    RiskAssessment,
    RiskEvidence,
    RiskEvidenceSource,
    SensitiveCategory,
)
from wajo_agent.normalization import EmailNormalizationReport


@dataclass(frozen=True, slots=True)
class InjectionRule:
    signal: InjectionSignal
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class SensitiveRule:
    category: SensitiveCategory
    pattern: re.Pattern[str]


INJECTION_RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        InjectionSignal.INSTRUCTION_OVERRIDE,
        re.compile(
            r"\b(ignore|disregard|override|forget)\b.{0,80}\b"
            r"(previous|prior|system|developer|security|instruction|rule|prompt)s?\b",
            re.I,
        ),
    ),
    InjectionRule(
        InjectionSignal.FAKE_PRIVILEGED_ROLE,
        re.compile(r"\b(system|developer|administrator)\s*(message|instruction|override)\b", re.I),
    ),
    InjectionRule(
        InjectionSignal.SECRET_EXFILTRATION,
        re.compile(
            r"\b(reveal|print|send|forward|expose)\b.{0,80}\b"
            r"(secret|api key|password|token|system prompt|developer message)\b",
            re.I,
        ),
    ),
    InjectionRule(
        InjectionSignal.FAKE_APPROVAL,
        re.compile(
            r"\b(approved by (the )?user|approval token|already approved|user has approved)\b",
            re.I,
        ),
    ),
    InjectionRule(
        InjectionSignal.ENCODED_INSTRUCTIONS,
        re.compile(r"\b(base64|decode|encoded)\b.{0,60}\b(instruction|command|payload)s?\b", re.I),
    ),
)

SENSITIVE_RULES: tuple[SensitiveRule, ...] = (
    SensitiveRule(
        SensitiveCategory.CREDENTIALS,
        re.compile(r"\b(password|passcode|one[- ]time code|otp|api key|secret token)\b", re.I),
    ),
    SensitiveRule(
        SensitiveCategory.ACCOUNT_RECOVERY,
        re.compile(r"\b(reset your password|account recovery|recover your account)\b", re.I),
    ),
    SensitiveRule(
        SensitiveCategory.BANKING,
        re.compile(
            r"\b(bank account|routing number|wire transfer|credit card|iban|swift code)\b", re.I
        ),
    ),
    SensitiveRule(
        SensitiveCategory.PAYMENT,
        re.compile(
            r"\b(make|send|authorize|complete|process)\s+(a\s+)?"
            r"(payment|purchase|wire|transfer)\b|\b(invoice|payment)\s+(due|required|request)\b",
            re.I,
        ),
    ),
    SensitiveRule(
        SensitiveCategory.LEGAL_COMMITMENT,
        re.compile(r"\b(sign|accept)\b.{0,30}\b(contract|agreement|terms)\b", re.I),
    ),
    SensitiveRule(
        SensitiveCategory.PERSONAL_DATA,
        re.compile(r"\b(social security number|ssn|passport number|tax id)\b", re.I),
    ),
)


class RiskScanner:
    """Find risk evidence in inert text; never decide or execute an action."""

    def scan(
        self,
        email: EmailEnvelope,
        *,
        normalization_report: EmailNormalizationReport | None = None,
    ) -> RiskAssessment:
        injection_signals: set[InjectionSignal] = set()
        sensitive_categories: set[SensitiveCategory] = set()
        normalization_flags: set[NormalizationFlag] = set()
        evidence: list[RiskEvidence] = []

        text_sources = (
            (RiskEvidenceSource.SUBJECT, email.subject),
            (RiskEvidenceSource.BODY, email.body_text),
        )
        for source, text in text_sources:
            for rule in INJECTION_RULES:
                match = rule.pattern.search(text)
                if match is not None:
                    injection_signals.add(rule.signal)
                    evidence.append(self._evidence(rule.signal, source, match.group(0)))

            for rule in SENSITIVE_RULES:
                match = rule.pattern.search(text)
                if match is not None:
                    sensitive_categories.add(rule.category)
                    evidence.append(self._evidence(rule.category, source, match.group(0)))

        if normalization_report is not None:
            self._add_normalization_findings(
                normalization_report,
                injection_signals,
                normalization_flags,
                evidence,
            )

        return RiskAssessment(
            injection_signals=frozenset(injection_signals),
            sensitive_categories=frozenset(sensitive_categories),
            normalization_flags=frozenset(normalization_flags),
            evidence=tuple(evidence),
            normalization_changed=(
                normalization_report.changed if normalization_report is not None else False
            ),
        )

    @staticmethod
    def _evidence(
        signal: InjectionSignal | SensitiveCategory | NormalizationFlag,
        source: RiskEvidenceSource,
        matched_text: str,
    ) -> RiskEvidence:
        safe_excerpt = " ".join(matched_text.split())[:160]
        return RiskEvidence(signal=signal.value, source=source, matched_text=safe_excerpt)

    def _add_normalization_findings(
        self,
        report: EmailNormalizationReport,
        injection_signals: set[InjectionSignal],
        flags: set[NormalizationFlag],
        evidence: list[RiskEvidence],
    ) -> None:
        if report.truncated_fields:
            flags.add(NormalizationFlag.TRUNCATED_CONTENT)
            evidence.append(
                self._evidence(
                    NormalizationFlag.TRUNCATED_CONTENT,
                    RiskEvidenceSource.NORMALIZATION,
                    "truncated fields: " + ", ".join(report.truncated_fields),
                )
            )

        if report.zero_width_removed:
            flags.add(NormalizationFlag.INVISIBLE_CHARACTERS)
            evidence.append(
                self._evidence(
                    NormalizationFlag.INVISIBLE_CHARACTERS,
                    RiskEvidenceSource.NORMALIZATION,
                    f"removed {report.zero_width_removed} invisible characters",
                )
            )
            if report.zero_width_removed >= 3:
                injection_signals.add(InjectionSignal.OBFUSCATED_INSTRUCTIONS)
                evidence.append(
                    self._evidence(
                        InjectionSignal.OBFUSCATED_INSTRUCTIONS,
                        RiskEvidenceSource.NORMALIZATION,
                        "repeated invisible characters may be hiding instructions",
                    )
                )

        if report.control_characters_removed:
            flags.add(NormalizationFlag.CONTROL_CHARACTERS)
            evidence.append(
                self._evidence(
                    NormalizationFlag.CONTROL_CHARACTERS,
                    RiskEvidenceSource.NORMALIZATION,
                    f"removed {report.control_characters_removed} control characters",
                )
            )

        if report.visible_content_removed:
            flags.add(NormalizationFlag.NO_VISIBLE_CONTENT)
            injection_signals.add(InjectionSignal.HIDDEN_ONLY_CONTENT)
            evidence.append(
                self._evidence(
                    InjectionSignal.HIDDEN_ONLY_CONTENT,
                    RiskEvidenceSource.NORMALIZATION,
                    "no visible content remained after normalization",
                )
            )
