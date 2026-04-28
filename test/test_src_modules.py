from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from textwrap import dedent

import pandas as pd
import numpy as np

from CI_gate import check_ci_gate, collect_gate_failures, load_config, load_evaluation_report, main as ci_gate_main
from evaluate import evaluate_model
import run_pipeline
from run_pipeline import build_training_summary, load_dataset, load_config as load_pipeline_config, split_features_target
from train import train

#When mimicking a behavior of an objects, it is better to create a mock class that has the same interface as the original object, rather than just a simple function. 
#Wehn only a few attributes are needed, simple namespace or dictionary can be used. But when the object has a more complex behavior, such as having methods that are called in a specific way, it is better to create a mock class that can simulate that behavior.
#if you want to have a betteer insight how the object is being called etc, especially useful for frontend testing, then creating magicmock class is better. But for simple cases, creating a mock class with the necessary attributes and methods is sufficient and more straightforward.

#monkeypatch is a fixture provided by pytest that allows you to temporarily modify or replace attributes, functions, or classes during testing. It is commonly used to mock external dependencies, such as APIs, databases, or complex objects, to isolate the unit of code being tested and control its behavior. 
#By using monkeypatch, you can ensure that your tests are deterministic and do not rely on external factors, making them more reliable and easier to debug.

#tmp_path is a fixture provided by pytest that creates a temporary directory for the duration of a test. It is useful for creating temporary files or directories that are needed during testing, without affecting the actual file system. 
#The temporary directory created by tmp_path is automatically cleaned up after the test is completed, ensuring that there are no leftover files or directories that could interfere with other tests or the development environment.

class DummyModel:
    def __init__(self, *, predict_output, predict_proba_output=None, classes_=None):
        self._predict_output = predict_output
        self._predict_proba_output = predict_proba_output
        if classes_ is not None:
            self.classes_ = classes_
    def predict(self, x_test):
        return self._predict_output
    def predict_proba(self, x_test):
        if self._predict_proba_output is None:
            raise AttributeError("predict_proba not configured")
        return self._predict_proba_output

   
def test_collect_gate_failures_and_check_ci_gate():
    report_1 = {"gates":
                {
                    "auc": {"passed": True, "value": 0.91},
                    "f1": {"passed": False, "value": 0.5},
                    "recall": {"passed": False, "value": 0.4},
                }
            }

    report_2 = {"auc_gate":{"passed": False, "value": 0.91},
                "f1_gate": {"passed": True, "value": 0.5},
                "recall_gate": {"passed": False, "value": 0.4},
            }
    
    failures_1 = collect_gate_failures(report_1)
    failures_2 = collect_gate_failures(report_2)

    assert failures_1 == ["gates.f1", "gates.recall"]
    assert failures_2 == ["auc_gate", "recall_gate"]

    assert check_ci_gate(report_1) == (False, ["gates.f1", "gates.recall"])
    assert check_ci_gate(report_2) == (False, ["auc_gate", "recall_gate"])


def test_load_config_and_report(tmp_path):
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n artifacts_dir: artifacts\n")

    report_path = tmp_path / "evaluation_report.json"
    report_path.write_text(json.dumps({"auc_gate": {"passed": True}}))

    assert load_config(config_path)["paths"]["artifacts_dir"] == "artifacts"
    assert load_evaluation_report(report_path)["auc_gate"]["passed"] is True


def test_load_dataset_and_split_features_target(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "Churn": [0, 1, 0],
        }
    ).to_csv(csv_path, index=False)

    df = load_dataset(csv_path)
    x, y = split_features_target(df, "Churn")
    assert list(x.columns) == ["feature1", "feature2"]
    assert list(y) == [0, 1, 0]


def test_build_training_summary():
    train_results = {
        "model_name": "RandomForest",
        "best_params": {"n_estimators": 100, "max_depth": 10},
        "best_cv_score": 0.85, 
        "search": None,
    }

    summary = build_training_summary(train_results)
    assert summary =={
        "model_name": "RandomForest",
        "best_params": {"n_estimators": 100, "max_depth": 10},
        "best_cv_score": 0.85,
    }


def test_ci_gate_main_uses_nested_gates(tmp_path):
    config_path = tmp_path / "config.yaml"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    config_path.write_text(f"paths: \n artifacts_dir: {artifacts_dir}\n")
    report_path = artifacts_dir / "evaluation_report.json"

    report_path.write_text(json.dumps({"gates": {"auc": {"passed": False, "value": 0.5}}}))
    assert ci_gate_main(config_path) == 1

    report_path.write_text(json.dumps({"gates": {"auc": {"passed": True, "value": 0.91}}}))
    assert ci_gate_main(config_path) == 0


def test_evaluate_model_writes_report_and_gate_status(tmp_path):
    model = DummyModel(
        predict_output=[0, 1, 1, 0],
        predict_proba_output=np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]]),
        classes_=[0, 1],
    )

    x_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 1, 1])

    artifacts_dir = tmp_path / "artifacts"
    report_path = artifacts_dir / "evaluation_report.json"

    evaluate_model(
        model,
        x_test,
        y_test,
        thresholds={
            "auc": {
                "baseline": 0.8,
                "delta": 0.02
            },
            "f1": {
                "min": 0.5
            },
            "precision": {
                "min": 0.5
            },
        },
        path=artifacts_dir 
    )
    report = load_evaluation_report(report_path)
    assert report["auc_gate"]["passed"] is True
    assert report["f1_gate"]["passed"] is True
    assert report["precision_gate"]["passed"] is True
    assert report["all_gates_passed"] is True


def test_train_with_single_model_and_prefixed_grid(monkeypatch):
   
    class SearchResult:
        def __init__(self):
            self.best_params_ = {"classifier_xgb__n_estimators": 100}
            self.best_score_ = 0.93
            self.best_estimator_ = "best_estimator_placeholder"
        
        def fit(self, x, y):
            return None

    class MockGridSearchCV:
        def __init__(self, estimator, param_grid, cv, scoring, n_jobs):
            self.estimator = estimator
            self.param_grid = param_grid
            self.cv = cv
            self.scoring = scoring
            self.n_jobs = n_jobs
            self.best_params_ = None
            self.best_score_ = None
            self.best_estimator_ = None

        def fit(self, x, y):
            self.best_params_ = {"classifier_xgb__n_estimators": 100}
            self.best_score_ = 0.93
            self.best_estimator_ = "best_estimator_placeholder"
            return self
        


    fake_pipeline = SimpleNamespace(
        named_steps={"preprocess": object(), "classifier_xgb": object()},
        steps=[("preprocess", object()), ("classifier_xgb", object())],
    )

    monkeypatch.setattr("train.GridSearchCV", MockGridSearchCV)
    results = train(
        x=pd.DataFrame({"feature": [1, 2]}),
        y=pd.Series([0, 1]),
        models={"XGBoost": (fake_pipeline, {"classifier_xgb__n_estimators": [50, 100]})},
        scoring="f1",
        cv_folds=3,
        random_state=42,
    )
    assert results["model_name"] == "XGBoost"
    assert results["best_params"] == {"classifier_xgb__n_estimators": 100}
    assert results["best_cv_score"] == 0.93


#integration test
def test_run_pipeline_selects_single_model_and_prefixes_hyperparameters(tmp_path, monkeypatch):
    #create dummy train/test csv files
    train_csv = tmp_path / "train_data.csv"
    test_csv = tmp_path / "test_data.csv"
    artifacts_dir = tmp_path / "artifacts"

    #write dummy data
    pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "Churn": [0, 1, 0, 1]
    }).to_csv(train_csv, index=False)

    pd.DataFrame({
        "feature": [5, 6],
        "Churn": [0, 1]
    }).to_csv(test_csv, index=False)

    #write config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text(dedent(f"""
    paths:
        train_data: {train_csv}
        test_data: {test_csv}
        artifacts_dir: {artifacts_dir}
    data:
        target_col: Churn
    training:
        scoring: f1
        model_name: XGBoost
        hyperparameters:
            n_estimators: [50, 100]
            max_depth: [3, 5]

    evaluation: 
        thresholds:
            auc:
                baseline: 0.8
                delta: 0.02
            f1:
                min: 0.5
            precision:
                min: 0.5
    """)
    )  

    #fake pipeline and functions to simulate training and evaluation without actually running them
    fake_pipeline = SimpleNamespace(
        named_steps={"preprocess": object(), "classifier_xgb": object()},
    )

    captured = {}
    def fake_train(*, x, y, models, scoring, cv_folds, random_state):
       captured["models"] = models
       return{
        "model_name": "XGBoost",
        "best_model": fake_pipeline,
        "best_params": {"classifier_xgb__n_estimators": 100, "classifier_xgb__max_depth": 5},
        "best_cv_score": 0.95,
       }
    
    def fake_evaluate_model(**kwargs):
        report = {
            "auc": 0.85,
            "f1": 0.74,
            "precision": 0.72,
            "recall": 0.70,
            "f1_gate": {"passed": True, "value": 0.74, "minimum": 0.5},
            "precision_gate": {"passed": True, "value": 0.72, "minimum": 0.5},
            "recall_gate": {"passed": True, "value": 0.70, "minimum": 0.5},
            "auc_gate": {"passed": True, "value": 0.85, "baseline": 0.8, "delta": 0.02, "minimum": 0.78},
            "all_gates_passed": True,
        }

        output_path = Path(kwargs["path"]) / "evaluation_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report))
        return report
    
    monkeypatch.setitem(run_pipeline.MODELS, "XGBoost", (fake_pipeline, {"classifier_xgb__n_estimators": [50, 100], "classifier_xgb__max_depth": [3, 5]}))
    monkeypatch.setattr(run_pipeline, "train", fake_train)
    monkeypatch.setattr(run_pipeline, "evaluate_model", fake_evaluate_model)

    result = run_pipeline.run_pipeline(config_file)

    assert list(captured["models"].keys()) == ["XGBoost"]
    assert captured["models"]["XGBoost"][1] == {
        "classifier_xgb__n_estimators": [50, 100],
        "classifier_xgb__max_depth": [3, 5],
    }       
    assert (artifacts_dir / "training_summary.json").exists()
    assert (artifacts_dir / "pipeline_summary.json").exists()
    assert result["training_summary"]["model_name"] == "XGBoost"