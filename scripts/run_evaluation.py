"""Run the frozen evaluation suite without granting mailbox execution access."""

from __future__ import annotations

import argparse
from pathlib import Path

from wajo_agent.config import Settings
from wajo_agent.evaluation import DatasetError, DatasetSplit, EvaluationRunner
from wajo_agent.planning import OfflinePlanner, OpenAIPlanner, PlannerError


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.cwd() / "data" / "evaluation",
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.split in {"held_out", "all"} and not args.confirm_held_out:
        parser.error("held-out evaluation requires --confirm-held-out")
    split = None if args.split == "all" else DatasetSplit(args.split)

    try:
        planner = OfflinePlanner() if args.planner == "offline" else OpenAIPlanner(model=args.model)
        result = EvaluationRunner(
            dataset_root=args.dataset_root,
            planner=planner,
            planner_name=("offline" if args.planner == "offline" else args.model),
            split=split,
            repeats=args.repeats,
        ).run()
    except (DatasetError, OSError, PlannerError, ValueError) as exc:
        parser.exit(2, f"evaluation could not run: {type(exc).__name__}: {exc}\n")

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        semantic = result.semantic
        injection = result.injection
        learning = result.learning
        brier_text = "n/a" if learning.brier_score is None else f"{learning.brier_score:.3f}"
        print(f"Wajo evaluation - {args.split} - planner: {result.planner}")
        print(
            "Semantic: "
            f"schema {_percentage(semantic.schema_valid_rate.value)}, "
            f"action {_percentage(semantic.action_accuracy.value)}, "
            f"intent {_percentage(semantic.intent_metrics.accuracy.value)}, "
            f"macro-F1 {semantic.intent_metrics.macro_f1:.3f}, "
            f"risk recall {_percentage(semantic.required_risk_recall.value)}"
        )
        print(
            "Injection: "
            f"attack recall {_percentage(injection.attack_detection_recall.value)}, "
            f"benign false positives {_percentage(injection.benign_false_positive_rate.value)}, "
            f"paired accuracy {_percentage(injection.correctly_classified_pairs.value)}, "
            f"policy bypass {_percentage(injection.attack_policy_bypass_rate.value)}"
        )
        print(
            "Learning: "
            f"ASK {_percentage(learning.ask_rate.value)}, "
            f"NOTIFY {_percentage(learning.notify_rate.value)}, "
            f"SILENT {_percentage(learning.silent_rate.value)}, "
            f"baseline ASK {_percentage(learning.no_learning_baseline_ask_rate.value)}, "
            f"Brier {brier_text}"
        )
        print(f"Absolute safety gates: {'PASS' if result.absolute_safety_gates_passed else 'FAIL'}")
    return 0 if result.absolute_safety_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
