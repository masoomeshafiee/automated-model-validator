from __future__  import annotations

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from typing import Any, Dict, Optional, Union
import json
import numpy as np
from pathlib import Path

def _build_gate_status(metric_value, threshold_rule, metric_name):
    gate = {"passed": True, "value": float(metric_value)}

    if threshold_rule is None:
        return gate

    #usually auc is the baseline, and the rest are minimum thresholds
    if metric_name == "auc":
        baseline = threshold_rule.get("baseline")
        delta = float(threshold_rule.get("delta", 0.0))
        if baseline is None:
            raise ValueError("AUC gate requires a baseline value.")
        minimum = float(baseline) - delta
        gate.update({"baseline": float(baseline), "delta": delta, "minimum": minimum})
        gate["passed"] = metric_value >= minimum
        return gate

    minimum = threshold_rule.get("min")
    if minimum is None:
        raise ValueError(f"{metric_name.upper()} gate requires a min value.")

    minimum = float(minimum)
    gate["minimum"] = minimum
    gate["passed"] = metric_value >= minimum
    return gate


def evaluate_model(model, x_test, y_test, thresholds: Optional[Dict[str, Dict[str, float]]], path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    #we assumed the lables are binary and the positive class is the one we want to predict
    y_test_array = np.array(y_test)

    try:#handle cases where model doesn't have predict method 
        y_pred = model.predict(x_test)
    except Exception as e:
        return {
            "error": f"Model prediction failed: {str(e)}"
        }
    
    try:#handle cases where model doesn't have predict_proba method
        y_score = model.predict_proba(x_test)[:, 1] #probaboility of the positive class
    except Exception as e:
        return {
            "error": f"Model probability prediction failed: {str(e)}"
        }
    
    auc = roc_auc_score(y_test_array, y_score)
    f1 = f1_score(y_test_array, y_pred)
    precision = precision_score(y_test_array, y_pred)
    recall = recall_score(y_test_array, y_pred)

    metrics = {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": confusion_matrix(y_test_array, y_pred).tolist(),
        "classification_report": classification_report(y_test_array, y_pred, output_dict=True),
    }

    if thresholds:
        for metric_name, threshold_rule in thresholds.items():
            if metric_name in metrics:
                gate = _build_gate_status(metrics[metric_name], threshold_rule, metric_name)
                metrics[f"{metric_name}_gate"] = gate


    if path:
        path = Path(path) / "evaluation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
    return metrics
