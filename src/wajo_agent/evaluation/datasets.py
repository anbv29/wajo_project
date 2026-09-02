"""Typed loading and integrity verification for frozen evaluation data."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ValidationError

from wajo_agent.evaluation.schemas import DatasetManifest


class DatasetError(RuntimeError):
    """A dataset is missing, malformed, or differs from its frozen manifest."""


def load_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    """Load nonblank JSONL rows through one strict Pydantic schema."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"dataset could not be read: {path}") from exc
    records: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetError(f"invalid dataset row {path}:{line_number}") from exc
    if not records:
        raise DatasetError(f"dataset is empty: {path}")
    return tuple(records)


def load_manifest(path: Path) -> DatasetManifest:
    try:
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"manifest could not be read: {path}") from exc
    except ValidationError as exc:
        raise DatasetError(f"manifest is malformed: {path}") from exc


def file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DatasetError(f"dataset could not be hashed: {path}") from exc


def verify_manifest(root: Path, manifest: DatasetManifest) -> None:
    """Reject missing, modified, or count-mismatched frozen files."""
    for item in manifest.files:
        path = root / item.path
        if file_sha256(path) != item.sha256:
            raise DatasetError(f"dataset hash differs from manifest: {item.path}")
        try:
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError as exc:
            raise DatasetError(f"dataset could not be counted: {item.path}") from exc
        if count != item.record_count:
            raise DatasetError(f"dataset count differs from manifest: {item.path}")
