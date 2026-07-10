from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VALID_SCORING_METRICS = {"f1", "precision", "recall", "roc_auc", "accuracy"}


def load_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a valid YAML dictionary.")

    return config


def require_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in config:
        raise ValueError(f"Missing required section: '{section}'")

    if not isinstance(config[section], dict):
        raise ValueError(f"Section '{section}' must be a dictionary.")

    return config[section]


def require_key(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key: '{section_name}.{key}'")

    return section[key]


def validate_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths = require_section(config, "paths")

    train_data = Path(require_key(paths, "train_data", "paths"))
    test_data = Path(require_key(paths, "test_data", "paths"))
    artifacts_dir = Path(require_key(paths, "artifacts_dir", "paths"))

    if not train_data.exists():
        raise ValueError(f"Training data file not found: {train_data}")

    if not test_data.exists():
        raise ValueError(f"Test data file not found: {test_data}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return train_data, test_data, artifacts_dir


def validate_data_section(config: dict[str, Any]) -> str:
    data = require_section(config, "data")

    target_col = require_key(data, "target_col", "data")

    if not isinstance(target_col, str) or not target_col.strip():
        raise ValueError("'data.target_col' must be a non-empty string.")

    return target_col


def validate_training_section(config: dict[str, Any]) -> None:
    training = require_section(config, "training")

    scoring = require_key(training, "scoring", "training")
    cv_folds = require_key(training, "cv_folds", "training")
    random_state = require_key(training, "random_state", "training")
    model_name = require_key(training, "model_name", "training")
    hyperparameters = require_key(training, "hyperparameters", "training")

    if scoring not in VALID_SCORING_METRICS:
        raise ValueError(
            f"'training.scoring' must be one of {sorted(VALID_SCORING_METRICS)}. "
            f"Got: {scoring}"
        )

    if not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError("'training.cv_folds' must be an integer >= 2.")

    if not isinstance(random_state, int):
        raise ValueError("'training.random_state' must be an integer.")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("'training.model_name' must be a non-empty string.")

    if not isinstance(hyperparameters, dict) or not hyperparameters:
        raise ValueError("'training.hyperparameters' must be a non-empty dictionary.")

    for param_name, values in hyperparameters.items():
        if not isinstance(param_name, str) or not param_name.strip():
            raise ValueError("Hyperparameter names must be non-empty strings.")

        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Hyperparameter '{param_name}' must contain a non-empty list of values."
            )


def validate_evaluation_section(config: dict[str, Any]) -> None:
    evaluation = require_section(config, "evaluation")
    thresholds = require_key(evaluation, "thresholds", "evaluation")

    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("'evaluation.thresholds' must be a non-empty dictionary.")

    for metric_name, settings in thresholds.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Threshold settings for '{metric_name}' must be a dictionary.")

        if "min" in settings:
            min_value = settings["min"]
            if not isinstance(min_value, (int, float)):
                raise ValueError(f"'evaluation.thresholds.{metric_name}.min' must be numeric.")

            if not 0 <= min_value <= 1:
                raise ValueError(
                    f"'evaluation.thresholds.{metric_name}.min' must be between 0 and 1."
                )

        if "baseline" in settings:
            baseline = settings["baseline"]
            if not isinstance(baseline, (int, float)):
                raise ValueError(
                    f"'evaluation.thresholds.{metric_name}.baseline' must be numeric."
                )

            if not 0 <= baseline <= 1:
                raise ValueError(
                    f"'evaluation.thresholds.{metric_name}.baseline' must be between 0 and 1."
                )

        if "delta" in settings:
            delta = settings["delta"]
            if not isinstance(delta, (int, float)):
                raise ValueError(f"'evaluation.thresholds.{metric_name}.delta' must be numeric.")

            if delta < 0:
                raise ValueError(f"'evaluation.thresholds.{metric_name}.delta' must be >= 0.")


def validate_target_column(train_data: Path, test_data: Path, target_col: str) -> None:
    train_df = pd.read_csv(train_data, nrows=5)
    test_df = pd.read_csv(test_data, nrows=5)

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test data.")


def validate_config(config_path: Path) -> None:
    config = load_yaml(config_path)

    train_data, test_data, _ = validate_paths(config)
    target_col = validate_data_section(config)

    validate_training_section(config)
    validate_evaluation_section(config)
    validate_target_column(train_data, test_data, target_col)

    print(f"Config validation passed: {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project config file.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        validate_config(args.config)
    except Exception as exc:
        print(f"Config validation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()