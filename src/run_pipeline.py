from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from evaluate import evaluate_model
from train import train

import yaml


DEFAULT_CONFIG_PATH = Path("./config.yaml")


# ==============================================================================
# Utility functions for the pipeline
# ==============================================================================

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level dictionary.")

    return config
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format for {path}. Use .csv or .parquet.")


def split_features_target(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def build_training_summary(train_results: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": train_results["model_name"],
        "best_params": train_results["best_params"],
        "best_cv_score": float(train_results["best_cv_score"]),
    }

# ==============================================================================
# Main pipeline function
# ==============================================================================

def run_pipeline(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config = load_config(config_path)

    train_data_path = Path(config["paths"]["train_data"])
    test_data_path = Path(config["paths"]["test_data"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])

    target_col = config["data"]["target_col"]

    training_cfg = config.get("training", {})
    scoring = training_cfg.get("scoring", "f1")
    cv_folds = training_cfg.get("cv_folds", 5)
    random_state = training_cfg.get("random_state", 42)

    evaluation_cfg = config.get("evaluation", {})
    thresholds = evaluation_cfg.get("thresholds", None)

    print("Loading processed datasets...")
    train_df = load_dataset(train_data_path)
    test_df = load_dataset(test_data_path)

    print("Preparing train/test splits...")
    x_train, y_train = split_features_target(train_df, target_col)
    x_test, y_test = split_features_target(test_df, target_col)

    print("Training and selecting best model...")
    train_results = train(
        x=x_train,
        y=y_train,
        scoring=scoring,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    best_model = train_results["best_model"]

    print(f"Best model selected: {train_results['model_name']}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Saving trained model...")
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(best_model, model_path)

    print("Saving training summary...")
    training_summary = build_training_summary(train_results)
    save_json(training_summary, artifacts_dir / "training_summary.json")

    print("Evaluating best model on held-out test data...")
    evaluation_results = evaluate_model(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
        thresholds=thresholds,
        path=artifacts_dir,
    )

    print("Saving pipeline summary...")
    pipeline_summary = {
        "config_path": str(config_path),
        "train_data_path": str(train_data_path),
        "test_data_path": str(test_data_path),
        "target_col": target_col,
        "artifacts_dir": str(artifacts_dir),
        "best_model_name": train_results["model_name"],
        "best_cv_score": float(train_results["best_cv_score"]),
        "evaluation_report_path": str(artifacts_dir / "evaluation_report.json"),
        "model_path": str(model_path),
    }
    save_json(pipeline_summary, artifacts_dir / "pipeline_summary.json")

    print("Pipeline completed successfully.")

    return {
        "train_results": train_results,
        "training_summary": training_summary,
        "evaluation_results": evaluation_results,
        "pipeline_summary": pipeline_summary,
        "model_path": model_path,
    }


if __name__ == "__main__":
    run_pipeline()