from __future__ import annotations

from math import isclose

import pytest

from wajo_agent.evaluation import brier_score, classification_metrics, latency_metrics


def test_classification_metrics_include_errors_and_zero_support() -> None:
    result = classification_metrics(
        ("a", "a", "b", "b"),
        ("a", "b", "b", "planner_error"),
        labels=("a", "b", "c"),
    )

    assert result.accuracy.numerator == 2
    assert result.accuracy.denominator == 4
    assert result.confusion_matrix["b"]["planner_error"] == 1
    assert result.per_class["c"].support == 0
    assert result.per_class["c"].f1 == 0.0


def test_brier_and_latency_metrics_are_deterministic() -> None:
    assert isclose(brier_score((0.9, 0.2), (True, False)), 0.025)
    latency = latency_metrics((1.0, 2.0, 3.0, 100.0))
    assert latency.mean_ms == 26.5
    assert latency.median_ms == 2.5
    assert latency.p95_ms == 100.0


def test_metric_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        classification_metrics(("a",), (), labels=("a",))
    with pytest.raises(ValueError):
        brier_score((1.1,), (True,))
    with pytest.raises(ValueError):
        latency_metrics((-1.0,))
