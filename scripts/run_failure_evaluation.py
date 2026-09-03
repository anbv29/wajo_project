"""Run all frozen failure scenarios through active deterministic fault injection."""

from __future__ import annotations

import argparse
from pathlib import Path

from wajo_agent.evaluation import DatasetError, FailureInjectionRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "data" / "evaluation",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = FailureInjectionRunner(dataset_root=args.dataset_root).run()
    except (DatasetError, OSError, ValueError) as exc:
        parser.exit(2, f"failure evaluation could not run: {type(exc).__name__}: {exc}\n")

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        for case in result.cases:
            status = "PASS" if case.passed else "FAIL"
            print(
                f"[{status}] {case.scenario_id}: {case.observed_outcome}; "
                f"provider calls={case.observed_provider_calls}; "
                f"automatic retries={case.automatic_retry_attempts}"
            )
        print(
            f"Failure scenarios: {result.passed_scenarios.numerator}/"
            f"{result.passed_scenarios.denominator} passed"
        )
        print(
            "Absolute failure-safety gates: "
            f"{'PASS' if result.absolute_safety_gates_passed else 'FAIL'}"
        )
    return 0 if result.absolute_safety_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
