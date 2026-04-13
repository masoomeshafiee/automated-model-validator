from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("./config.yaml")

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level dictionary.")

    return config

def load_evaluation_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_gate_failures(report: Dict[str, Any]) -> list[str]:
    failures: list[str] = []

    for key, value in report.items():
        if key.endswith("_gate") and isinstance(value, dict):
            if value.get("passed") is False:
                failures.append(key)

    return failures


def check_ci_gate(report: Dict[str, Any]) -> tuple[bool, list[str]]:
    failures = collect_gate_failures(report)
    return len(failures) == 0, failures

# ==============================================================================
# Main CI gate evaluation function
# ==============================================================================
def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> int:
    config = load_config(config_path)
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    report_path = artifacts_dir / "evaluation_report.json"

    report = load_evaluation_report(report_path)
    passed, failures = check_ci_gate(report)

    if passed:
        print("CI gate PASSED. All evaluation gates passed.")
        return 0

    print("CI gate FAILED. The following gates failed:")
    for failure in failures:
        print(f" - {failure}: {report.get(failure, {})}")

    return 1


if __name__ == "__main__":
    sys.exit(main())