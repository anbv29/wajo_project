"""Dependency-free classification, rate, latency, and calibration metrics."""

from __future__ import annotations

from statistics import fmean, median

from wajo_agent.evaluation.results import (
    ClassificationMetrics,
    ClassMetrics,
    LatencyMetrics,
    RateMetric,
)


def rate(numerator: int, denominator: int) -> RateMetric:
    return RateMetric(numerator=numerator, denominator=denominator)


def classification_metrics(
    expected: tuple[str, ...],
    predicted: tuple[str, ...],
    *,
    labels: tuple[str, ...],
) -> ClassificationMetrics:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted classifications must have equal length")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("classification labels must be non-empty and unique")
    if any(value not in labels for value in expected):
        raise ValueError("expected classification is outside the declared labels")

    confusion = {
        actual: {prediction: 0 for prediction in (*labels, "planner_error")} for actual in labels
    }
    for actual, prediction in zip(expected, predicted, strict=True):
        column = prediction if prediction in labels else "planner_error"
        confusion[actual][column] += 1

    per_class: dict[str, ClassMetrics] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(row[label] for actual, row in confusion.items() if actual != label)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=true_positive + false_negative,
        )

    correct = sum(
        actual == prediction for actual, prediction in zip(expected, predicted, strict=True)
    )
    return ClassificationMetrics(
        accuracy=rate(correct, len(expected)),
        macro_precision=fmean(item.precision for item in per_class.values()),
        macro_recall=fmean(item.recall for item in per_class.values()),
        macro_f1=fmean(item.f1 for item in per_class.values()),
        per_class=per_class,
        confusion_matrix=confusion,
    )


def latency_metrics(values_ms: tuple[float, ...]) -> LatencyMetrics:
    if any(value < 0 for value in values_ms):
        raise ValueError("latency cannot be negative")
    if not values_ms:
        return LatencyMetrics(count=0, mean_ms=0.0, median_ms=0.0, p95_ms=0.0)
    ordered = sorted(values_ms)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return LatencyMetrics(
        count=len(values_ms),
        mean_ms=fmean(values_ms),
        median_ms=median(values_ms),
        p95_ms=ordered[p95_index],
    )


def brier_score(probabilities: tuple[float, ...], outcomes: tuple[bool, ...]) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if not probabilities:
        raise ValueError("Brier score requires at least one observation")
    if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
        raise ValueError("probabilities must be between zero and one")
    return fmean(
        (probability - float(outcome)) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
