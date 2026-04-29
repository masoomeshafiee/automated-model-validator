from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml

from evaluate import evaluate_model
from preprocess import MODELS
from train import train


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# Metric used to decide if a new run beats the current best.
# AUC is more stable than F1 across runs, so we use it for selection.
BEST_MODEL_METRIC = "auc"


# ==============================================================================
# Utility functions
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


def is_better_run(current_metrics: Dict[str, Any], best_metrics_path: Path) -> bool:
    """
    Return True if the current run beats the previous best on BEST_MODEL_METRIC.
    Returns True if no previous best exists.
    """
    if not best_metrics_path.exists():
        return True
    with open(best_metrics_path, "r", encoding="utf-8") as f:
        prev_best = json.load(f)
    return current_metrics[BEST_MODEL_METRIC] > prev_best.get(BEST_MODEL_METRIC, -float("inf"))


# ==============================================================================
# Main pipeline
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

    model_name = training_cfg.get("model_name", None)
    custom_grid = training_cfg.get("hyperparameters", None)

    # Decide which models to train
    if model_name:
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in predefined MODELS.")
        base_model, default_grid = MODELS[model_name]
        classifier_step = [
            step_name for step_name, _ in base_model.named_steps.items()
            if step_name.startswith("classifier")
        ][0]
        if custom_grid:
            selected_grid = {
                f"{classifier_step}__{param}": values
                for param, values in custom_grid.items()
            }
        else:
            selected_grid = default_grid
        models_to_train = {model_name: (base_model, selected_grid)}
    else:
        models_to_train = MODELS

    # Load data
    print("Loading processed datasets...")
    train_df = load_dataset(train_data_path)
    test_df = load_dataset(test_data_path)

    print("Preparing train/test splits...")
    x_train, y_train = split_features_target(train_df, target_col)
    x_test, y_test = split_features_target(test_df, target_col)

    # Train
    print("Training and selecting best model from this run...")
    train_results = train(
        x=x_train,
        y=y_train,
        models=models_to_train,
        scoring=scoring,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    best_model = train_results["best_model"]
    print(f"Best model from this run: {train_results['model_name']}")

    # ==========================================================================
    # Save this run in its own timestamped folder
    # ==========================================================================
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving this run to: {run_dir}")

    # Save model for this run
    run_model_path = run_dir / "model.joblib"
    joblib.dump(best_model, run_model_path)

    # Save training summary for this run
    training_summary = build_training_summary(train_results)
    save_json(training_summary, run_dir / "training_summary.json")

    # Evaluate and save evaluation report for this run
    print("Evaluating on held-out test set...")
    evaluation_results = evaluate_model(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
        thresholds=thresholds,
        path=run_dir,
    )

    # ==========================================================================
    # Update "best" artifacts only if this run beats the previous best
    # ==========================================================================
    best_model_path = artifacts_dir / "best_model.joblib"
    best_metrics_path = artifacts_dir / "best_metrics.json"
    best_run_info_path = artifacts_dir / "best_run_info.json"

    if is_better_run(evaluation_results, best_metrics_path):
        print(f"New best run! Updating best_model.joblib (based on {BEST_MODEL_METRIC}).")
        shutil.copy(run_model_path, best_model_path)
        save_json(evaluation_results, best_metrics_path)
        save_json(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "model_name": train_results["model_name"],
                "selection_metric": BEST_MODEL_METRIC,
                f"best_{BEST_MODEL_METRIC}": evaluation_results[BEST_MODEL_METRIC],
            },
            best_run_info_path,
        )
    else:
        print(f"This run did not beat previous best on {BEST_MODEL_METRIC}. Keeping existing best.")

    # ==========================================================================
    # Pipeline summary (always points to this run + the current best)
    # ==========================================================================
    pipeline_summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "train_data_path": str(train_data_path),
        "test_data_path": str(test_data_path),
        "target_col": target_col,
        "this_run_model_path": str(run_model_path),
        "this_run_metrics_path": str(run_dir / "evaluation_report.json"),
        "best_model_path": str(best_model_path),
        "best_metrics_path": str(best_metrics_path),
        "best_model_name": train_results["model_name"],
        "best_cv_score": float(train_results["best_cv_score"]),
    }
    save_json(pipeline_summary, run_dir / "pipeline_summary.json")

    # Also keep a top-level evaluation_report.json for the CI gate to read
    save_json(evaluation_results, artifacts_dir / "evaluation_report.json")

    print("Pipeline completed successfully.")

    return {
        "train_results": train_results,
        "training_summary": training_summary,
        "evaluation_results": evaluation_results,
        "pipeline_summary": pipeline_summary,
        "run_model_path": run_model_path,
        "best_model_path": best_model_path,
    }


if __name__ == "__main__":
    run_pipeline()