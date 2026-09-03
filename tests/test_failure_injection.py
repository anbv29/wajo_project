from __future__ import annotations

from pathlib import Path

from wajo_agent.evaluation import FailureInjectionRunner


def test_all_frozen_failures_are_actively_injected_and_fail_closed() -> None:
    result = FailureInjectionRunner(dataset_root=Path.cwd() / "data" / "evaluation").run()

    assert len(result.cases) == 24
    assert result.passed_scenarios.numerator == 24
    assert result.outcome_mismatches == 0
    assert result.provider_call_mismatches == 0
    assert result.automatic_retry_violations == 0
    assert result.missing_audit_evidence == 0
    assert result.safety_floor_violations == 0
    assert result.absolute_safety_gates_passed

    by_id = {case.scenario_id: case for case in result.cases}
    assert by_id["fail_executor_timeout"].observed_outcome == "UNKNOWN"
    assert by_id["fail_crash_after_claim"].observed_outcome == "Require reconciliation"
    assert by_id["fail_feedback_write_failure"].observed_outcome == (
        "Roll back all feedback writes"
    )
    assert by_id["fail_gmail_post_timeout"].observed_provider_calls == 1
