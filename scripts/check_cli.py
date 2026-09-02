"""End-to-end checks for the installed-style Typer CLI and curated demo."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from typer.testing import CliRunner

from wajo_agent.cli import app
from wajo_agent.domain import ApprovalStatus, AutonomyTier, OutcomeRoute
from wajo_agent.storage import SCHEMA_VERSION, SQLiteStore


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@contextmanager
def _workspace_database_files() -> Generator[tuple[Path, Path], None, None]:
    databases = (Path.cwd() / "cli_check.sqlite3", Path.cwd() / "cli_demo_check.sqlite3")
    artifacts = tuple(
        Path(f"{database}{suffix}") for database in databases for suffix in ("", "-wal", "-shm")
    )
    if any(path.exists() for path in artifacts):
        raise RuntimeError("CLI-check artifact already exists; refusing to overwrite it")
    try:
        yield databases
    finally:
        for path in artifacts:
            path.unlink(missing_ok=True)


def _invoke_json(runner: CliRunner, db_path: Path, *arguments: str) -> object:
    result = runner.invoke(app, ["--db", str(db_path), "--json", *arguments])
    _require(result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.exception}")
    return json.loads(result.stdout)


def main() -> None:
    checks = 0
    runner = CliRunner()
    fixture_root = Path.cwd() / "data" / "fixtures"

    with _workspace_database_files() as (db_path, demo_db_path):
        initialized = _invoke_json(runner, db_path, "init-db")
        _require(isinstance(initialized, dict), "init-db did not return an object")
        _require(initialized["schema_version"] == SCHEMA_VERSION == 6, "wrong CLI schema")
        checks += 2

        first = _invoke_json(
            runner,
            db_path,
            "ingest",
            str(fixture_root / "newsletter.json"),
        )
        _require(isinstance(first, dict), "ingest did not return an outcome")
        _require(first["decision"]["tier"] == AutonomyTier.ASK.value, "cold start did not ask")
        _require(first["route"] == OutcomeRoute.AWAITING_APPROVAL.value, "ASK route was wrong")
        approval_id = str(first["approval"]["approval_id"])
        decision_id = str(first["decision"]["decision_id"])
        checks += 3

        listed = _invoke_json(runner, db_path, "inbox")
        _require(isinstance(listed, list) and len(listed) == 1, "inbox missed the run")
        _require(listed[0]["status"] == ApprovalStatus.PENDING.value, "status was stale")
        detail = _invoke_json(runner, db_path, "decision", decision_id)
        _require(detail == first, "decision lookup did not recover the exact outcome")
        checks += 3

        without_consent = runner.invoke(
            app,
            ["--db", str(db_path), "--json", "approve", approval_id],
        )
        _require(without_consent.exit_code == 1, "JSON approval did not require explicit consent")
        checks += 1

        approved = _invoke_json(runner, db_path, "approve", approval_id, "--yes")
        _require(approved["execution"]["state"] == "succeeded", "approval did not execute")
        _require(approved["feedback_applied"], "approval feedback was not learned")
        with SQLiteStore(db_path) as store:
            persisted = store.get_agent_outcome_by_decision(decision_id)
            _require(persisted is not None and persisted.preference is not None, "outcome vanished")
            preference = store.get_preference(persisted.preference.context_key)
            _require(
                preference.observations == 1 and preference.alpha == 2, "approval evidence wrong"
            )
        checks += 4

        duplicate = runner.invoke(app, ["--db", str(db_path), "approve", approval_id])
        _require(duplicate.exit_code == 1, "consumed approval executed twice")
        checks += 1

        request = _invoke_json(
            runner,
            db_path,
            "ingest",
            str(fixture_root / "request.json"),
        )
        original_approval = str(request["approval"]["approval_id"])
        payload = json.dumps(
            {
                "kind": "create_draft",
                "recipients": ["teammate@example.com"],
                "subject": "Re: Please review the launch note",
                "body": "Thanks. I reviewed the wording and will send comments tomorrow.",
            }
        )
        edited = _invoke_json(
            runner,
            db_path,
            "edit",
            original_approval,
            "--payload-json",
            payload,
        )
        new_approval = str(edited["new_approval"]["approval_id"])
        _require(edited["proposal"]["version"] == 2, "edit did not create version 2")
        _require(new_approval != original_approval, "edit reused approval authority")
        _require(edited["feedback_applied"], "edit feedback was not learned")
        checks += 3

        rejected = _invoke_json(runner, db_path, "reject", new_approval)
        _require(rejected["approval"]["status"] == ApprovalStatus.REJECTED.value, "reject failed")
        _require(rejected["feedback_applied"], "rejection did not update evidence")
        checks += 2

        preferences = _invoke_json(runner, db_path, "preferences")
        recent_events = _invoke_json(runner, db_path, "events", "--limit", "10")
        _require(isinstance(preferences, list) and preferences, "preferences output was empty")
        _require(
            isinstance(recent_events, list) and len(recent_events) == 10, "events limit failed"
        )
        checks += 2

        demo = _invoke_json(runner, demo_db_path, "demo", "--reset")
        _require(isinstance(demo, dict), "demo did not return an object")
        scenarios = demo["scenarios"]
        _require(len(scenarios) == 5, "demo did not contain five scenarios")
        tiers = {row["scenario"]: row["tier"] for row in scenarios}
        routes = {row["scenario"]: row["route"] for row in scenarios}
        _require(tiers["cold_start"] == AutonomyTier.ASK.value, "demo missed cold ASK")
        _require(tiers["learned_notify"] == AutonomyTier.NOTIFY.value, "demo missed NOTIFY")
        _require(tiers["learned_silent"] == AutonomyTier.SILENT.value, "demo missed SILENT")
        _require(
            tiers["prompt_injection"] == AutonomyTier.ESCALATE.value,
            "demo injection did not escalate",
        )
        _require(
            routes["unknown_effect"] == OutcomeRoute.EXECUTION_UNKNOWN.value,
            "demo hid unknown execution",
        )
        checks += 7

        with SQLiteStore(demo_db_path) as reopened:
            _require(len(reopened.list_agent_outcomes(limit=100)) >= 5, "demo was not durable")
            _require(reopened.schema_version == 6, "demo database did not reopen")
        checks += 2

    print(f"CLI checks passed: {checks}")


if __name__ == "__main__":
    main()
