"""Run every local submission gate with one fail-fast command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CHECK_SCRIPTS = (
    "check_environment.py",
    "check_contracts.py",
    "check_policy.py",
    "check_context.py",
    "check_learning.py",
    "check_storage.py",
    "check_approvals.py",
    "check_execution.py",
    "check_feedback.py",
    "check_offline_planner.py",
    "check_openai_planner.py",
    "check_agent.py",
    "check_cli.py",
    "check_gmail.py",
    "check_datasets.py",
)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    python = sys.executable
    commands = (
        ("format", (python, "-m", "ruff", "format", "--check", ".")),
        ("lint", (python, "-m", "ruff", "check", ".")),
        ("types", (python, "-m", "pyright")),
        ("pytest", (python, "-m", "pytest")),
        *(
            (path.removesuffix(".py"), (python, str(root / "scripts" / path)))
            for path in CHECK_SCRIPTS
        ),
        (
            "development_evaluation",
            (python, str(root / "scripts" / "run_evaluation.py")),
        ),
    )
    for name, command in commands:
        print(f"Running gate: {name}", flush=True)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            print(f"Quality gate failed: {name}", file=sys.stderr)
            return completed.returncode
    print(f"All {len(commands)} quality gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
