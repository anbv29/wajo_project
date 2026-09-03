from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from wajo_agent.evaluation import (
    DatasetSplit,
    EvaluationReportWriter,
    EvaluationRunner,
    FailureCaseResult,
    FailureEvaluation,
    RateMetric,
    wilson_interval,
)
from wajo_agent.planning import OfflinePlanner


def _passing_failure_evaluation() -> FailureEvaluation:
    case = FailureCaseResult(
        scenario_id="fail_test",
        component="test",
        expected_outcome="safe",
        observed_outcome="safe",
        outcome_matched=True,
        expected_provider_call=False,
        observed_provider_calls=0,
        provider_call_matched=True,
        automatic_retry_attempts=0,
        audit_evidence_present=True,
        safety_floor_preserved=True,
        passed=True,
        duration_ms=0.1,
        detail="Synthetic reporting fixture passed safely.",
    )
    return FailureEvaluation(
        cases=(case,),
        passed_scenarios=RateMetric(numerator=1, denominator=1),
        outcome_mismatches=0,
        provider_call_mismatches=0,
        automatic_retry_violations=0,
        missing_audit_evidence=0,
        safety_floor_violations=0,
    )


def test_wilson_interval_is_bounded_and_handles_extremes() -> None:
    none = wilson_interval(RateMetric(numerator=0, denominator=0))
    zero = wilson_interval(RateMetric(numerator=0, denominator=10))
    one = wilson_interval(RateMetric(numerator=10, denominator=10))

    assert none.estimate is None
    assert zero.lower == 0.0
    assert zero.upper is not None and 0.27 < zero.upper < 0.28
    assert one.lower is not None and 0.72 < one.lower < 0.73
    assert one.upper == 1.0


def test_report_writer_preserves_raw_evidence_metadata_and_pngs() -> None:
    project_root = Path.cwd()
    dataset_root = project_root / "data" / "evaluation"
    output_directory = project_root / f".report-test-{uuid4().hex}"
    evaluation = EvaluationRunner(
        dataset_root=dataset_root,
        planner=OfflinePlanner(),
        planner_name="offline",
        split=DatasetSplit.DEVELOPMENT,
    ).run()

    try:
        artifacts = EvaluationReportWriter(
            project_root=project_root,
            dataset_root=dataset_root,
            output_directory=output_directory,
        ).write(evaluation, _passing_failure_evaluation())

        assert len(artifacts.files) == 10
        assert artifacts.absolute_safety_gates_passed
        assert artifacts.quality_gates_passed
        metadata = json.loads((output_directory / "run_metadata.json").read_text(encoding="utf-8"))
        semantic = json.loads(
            (output_directory / "semantic_results.json").read_text(encoding="utf-8")
        )
        assert metadata["report_schema_version"] == "1.0.0"
        assert metadata["dataset_files"]["semantic.jsonl"]["record_count"] == 72
        assert metadata["token_usage"]["available"] is False
        assert len(semantic["results"]["cases"]) == 48
        assert semantic["confidence_intervals"]["action_accuracy"]["method"] == "wilson_score"
        assert "These are measured results" in (output_directory / "summary.md").read_text(
            encoding="utf-8"
        )
        for name in (
            "confusion_matrix.png",
            "calibration_curve.png",
            "autonomy_over_time.png",
        ):
            assert (output_directory / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        if output_directory.exists():
            for path in output_directory.iterdir():
                path.unlink()
            output_directory.rmdir()
