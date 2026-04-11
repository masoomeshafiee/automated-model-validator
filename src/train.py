from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import  GridSearchCV, StratifiedKFold

from preprocess import MODELS


def _fit_with_grid_search(x, y, model_name, model, param_grid, scoring, cv):

    print(f"Training model: {model_name}")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    grid_search.fit(x, y)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best score for {model_name}: {grid_search.best_score_}")

    return grid_search


def _build_gate_status(metric_value, threshold_rule, metric_name):
    gate = {"value": float(metric_value), "passed": True}
    if not threshold_rule:
        return gate
    
    #usually auc is the baseline, and the rest are minimum thresholds

    if metric_name == "auc":
        baseline = threshold_rule.get("baseline")
        if baseline is None:
            raise ValueError("AUC gate requires a baseline value.")    
        delta = float(threshold_rule.get("delta", 0.0))
        minimum = float(baseline) - delta
        gate.update(
            {
                "baseline": float(baseline),
                "delta": delta,
                "minimum": minimum,
            }
        )

    else:
        minimum = threshold_rule.get("min")
        if minimum is None:
            raise ValueError(f"{metric_name.upper()} gate requires a min value.")     
        gate["minimum"] = float(minimum)

    gate["passed"] = gate["value"] >= gate["minimum"]
    return gate


def train(x, y, models=MODELS, scoring='f1'):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = None
    best_search = None
    best_score = -float('inf')

    for model_name, (model, param_grid) in models.items():
        search = _fit_with_grid_search(
            x,
            y,
            model_name,
            model,
            param_grid,
            scoring='f1',
            cv=cv,
        )  
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model_name = model_name
            best_search = search  
    
    return {
        "model_name": best_model_name,
        "best_model": best_search.best_estimator_,
        "best_params": best_search.best_params_,
        "best_cv_score": best_search.best_score_,
        "search": best_search,
    } 
