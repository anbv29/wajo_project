"""Convert raw email content into inert, consistent text for the agent."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

from wajo_agent.domain import EmailEnvelope

ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_SPACE = re.compile(r"[ \t]+")
SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
EXCESS_NEWLINES = re.compile(r"\n{3,}")

SKIPPED_HTML_TAGS = frozenset({"script", "style", "noscript", "template", "head", "svg"})
BLOCK_HTML_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
)


@dataclass(frozen=True, slots=True)
class TextNormalizationResult:
    text: str
    changed: bool
    truncated: bool
    zero_width_removed: int
    control_characters_removed: int


@dataclass(frozen=True, slots=True)
class EmailNormalizationReport:
    changed_fields: tuple[str, ...]
    truncated_fields: tuple[str, ...]
    zero_width_removed: int
    control_characters_removed: int
    html_was_present: bool
    html_alternative_added: bool
    visible_content_removed: bool

    @property
    def changed(self) -> bool:
        return bool(self.changed_fields)


class _VisibleHTMLExtractor(HTMLParser):
    """Extract visible text without executing or preserving active HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered in SKIPPED_HTML_TAGS:
            self.skip_depth += 1
        elif self.skip_depth == 0 and lowered in BLOCK_HTML_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self.skip_depth == 0 and tag.casefold() in BLOCK_HTML_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in SKIPPED_HTML_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
        elif self.skip_depth == 0 and lowered in BLOCK_HTML_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)


def _visible_html_text(value: str) -> str:
    parser = _VisibleHTMLExtractor()
    parser.feed(value)
    parser.close()
    return "".join(parser.parts)


def normalize_text(
    value: str,
    *,
    max_chars: int,
    treat_as_html: bool = False,
) -> TextNormalizationResult:
    """Normalize one text field and report every lossy transformation."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    original = value
    visible = _visible_html_text(value) if treat_as_html else value
    visible = html.unescape(visible)
    visible = unicodedata.normalize("NFKC", visible)

    zero_width_count = len(ZERO_WIDTH.findall(visible))
    visible = ZERO_WIDTH.sub("", visible)

    control_count = len(CONTROL_CHARACTERS.findall(visible))
    visible = CONTROL_CHARACTERS.sub("", visible)

    visible = visible.replace("\r\n", "\n").replace("\r", "\n")
    visible = HORIZONTAL_SPACE.sub(" ", visible)
    visible = SPACE_AROUND_NEWLINE.sub("\n", visible)
    visible = EXCESS_NEWLINES.sub("\n\n", visible).strip()

    truncated = len(visible) > max_chars
    if truncated:
        visible = visible[:max_chars].rstrip()

    return TextNormalizationResult(
        text=visible,
        changed=visible != original,
        truncated=truncated,
        zero_width_removed=zero_width_count,
        control_characters_removed=control_count,
    )


def normalize_email(email: EmailEnvelope) -> tuple[EmailEnvelope, EmailNormalizationReport]:
    """Create a new normalized email while preserving the original observation."""
    subject = normalize_text(email.subject, max_chars=500)
    plain_body = normalize_text(email.body_text, max_chars=50_000)

    html_body: TextNormalizationResult | None = None
    if email.body_html is not None:
        html_body = normalize_text(email.body_html, max_chars=50_000, treat_as_html=True)

    combined_body = plain_body.text
    html_alternative_added = False
    if html_body is not None and html_body.text:
        same_visible_content = html_body.text.casefold() == plain_body.text.casefold()
        if not plain_body.text:
            combined_body = html_body.text
        elif not same_visible_content:
            combined_body = f"{plain_body.text}\n\n[HTML alternative]\n{html_body.text}"
            html_alternative_added = True

    final_body = normalize_text(combined_body, max_chars=50_000)
    combined_body = final_body.text

    visible_content_removed = False
    if not subject.text and not combined_body and not email.attachments:
        combined_body = "[No visible content remained after normalization.]"
        visible_content_removed = True

    changed_fields: list[str] = []
    truncated_fields: list[str] = []
    if subject.changed:
        changed_fields.append("subject")
    if plain_body.changed or combined_body != email.body_text:
        changed_fields.append("body_text")
    if email.body_html is not None:
        changed_fields.append("body_html")
    if subject.truncated:
        truncated_fields.append("subject")
    if (
        plain_body.truncated
        or (html_body is not None and html_body.truncated)
        or final_body.truncated
    ):
        truncated_fields.append("body")

    normalized_data = email.model_dump(mode="python")
    normalized_data.update(
        {
            "subject": subject.text,
            "body_text": combined_body,
            "body_html": None,
        }
    )
    normalized = EmailEnvelope.model_validate(normalized_data)

    report = EmailNormalizationReport(
        changed_fields=tuple(dict.fromkeys(changed_fields)),
        truncated_fields=tuple(dict.fromkeys(truncated_fields)),
        zero_width_removed=subject.zero_width_removed
        + plain_body.zero_width_removed
        + (html_body.zero_width_removed if html_body is not None else 0),
        control_characters_removed=subject.control_characters_removed
        + plain_body.control_characters_removed
        + (html_body.control_characters_removed if html_body is not None else 0),
        html_was_present=email.body_html is not None,
        html_alternative_added=html_alternative_added,
        visible_content_removed=visible_content_removed,
    )
    return normalized, report
