"""Generate versioned evaluation evidence, metadata, summary, and charts."""

from __future__ import annotations

import argparse
from pathlib import Path

from wajo_agent.config import Settings
from wajo_agent.evaluation import (
    DatasetError,
    DatasetSplit,
    EvaluationReportWriter,
    EvaluationRunner,
    FailureInjectionRunner,
)
from wajo_agent.planning import OfflinePlanner, OpenAIPlanner, PlannerError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "data" / "evaluation",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path.cwd() / "reports",
    )
    parser.add_argument("--planner", choices=("offline", "openai"), default="offline")
    parser.add_argument("--model", default=Settings.from_env().planner_model)
    parser.add_argument(
        "--split",
        choices=("development", "held_out", "all"),
        default="development",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--confirm-held-out", action="store_true")
    args = parser.parse_args()

    if args.split in {"held_out", "all"} and not args.confirm_held_out:
        parser.error("held-out report generation requires --confirm-held-out")
    split = None if args.split == "all" else DatasetSplit(args.split)
    project_root = Path(__file__).resolve().parent.parent

    try:
        planner = OfflinePlanner() if args.planner == "offline" else OpenAIPlanner(model=args.model)
        evaluation = EvaluationRunner(
            dataset_root=args.dataset_root,
            planner=planner,
            planner_name=("offline" if args.planner == "offline" else args.model),
            split=split,
            repeats=args.repeats,
        ).run()
        failures = FailureInjectionRunner(dataset_root=args.dataset_root).run()
        artifacts = EvaluationReportWriter(
            project_root=project_root,
            dataset_root=args.dataset_root,
            output_directory=args.output_directory,
        ).write(evaluation, failures)
    except (DatasetError, OSError, PlannerError, ValueError) as exc:
        parser.exit(2, f"reports could not be generated: {type(exc).__name__}: {exc}\n")

    print(f"Generated {len(artifacts.files)} report artifacts in {artifacts.output_directory}")
    print(f"Absolute safety gates: {'PASS' if artifacts.absolute_safety_gates_passed else 'FAIL'}")
    print(f"Quality gates: {'PASS' if artifacts.quality_gates_passed else 'FAIL'}")
    return 0 if artifacts.absolute_safety_gates_passed and artifacts.quality_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
