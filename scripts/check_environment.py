"""Verify that the local machine can develop and run the Wajo email agent.

This script uses only Python's standard library, so it can explain missing dependencies even when
the project environment has not been set up correctly yet.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass

MINIMUM_PYTHON = (3, 12)
REQUIRED_MODULES = (
    "openai",
    "pydantic",
    "rich",
    "typer",
    "hypothesis",
    "pytest",
    "ruff",
    "pyright",
)


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    name: str
    passed: bool
    detail: str


def collect_checks() -> tuple[EnvironmentCheck, ...]:
    python_ok = sys.version_info >= MINIMUM_PYTHON
    checks: list[EnvironmentCheck] = [
        EnvironmentCheck(
            name="Python version",
            passed=python_ok,
            detail=(
                f"{platform.python_version()} installed; "
                f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer required"
            ),
        )
    ]

    for module_name in REQUIRED_MODULES:
        installed = importlib.util.find_spec(module_name) is not None
        checks.append(
            EnvironmentCheck(
                name=module_name,
                passed=installed,
                detail="installed" if installed else "missing",
            )
        )
    return tuple(checks)


def main() -> int:
    checks = collect_checks()
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")

    failed = tuple(check for check in checks if not check.passed)
    if failed:
        print(f"\nEnvironment is not ready: {len(failed)} check(s) failed.")
        return 1

    print("\nEnvironment is ready for Wajo agent development.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
