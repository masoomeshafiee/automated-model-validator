# Automated Model Validator

Train, evaluate, and gate a churn model with reproducible artifacts and CI-friendly pass/fail checks.

## What This Project Does

- Trains a model pipeline using processed train data
- Runs hyperparameter search with `GridSearchCV`
- Evaluates the best model on held-out test data
- Applies metric gates (AUC/F1/precision/recall) from `config.yaml`
- Stores per-run artifacts and keeps a top-level report for CI checks

## Repository Layout

- `src/preprocess.py`: preprocessing pipelines, model registry, default parameter grids
- `src/train.py`: model selection using cross-validation
- `src/evaluate.py`: metric calculation and gate evaluation
- `src/run_pipeline.py`: end-to-end orchestration and artifact writing
- `src/CI_gate.py`: exits non-zero when any configured gate fails
- `config.yaml`: paths, training settings, and evaluation thresholds
- `test/`: pytest suite
- `artifacts/`: generated model and report outputs

## Requirements

- Python 3.11+ recommended for local runs
- Processed datasets available at paths in `config.yaml`:
  - `paths.train_data`
  - `paths.test_data`

By default this project points to:
- `data/processed/train_data.csv`
- `data/processed/test_data.csv`

## Setup

### Option 1: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate churn_ML_CI_CD
```

## Run the Pipeline

```bash
python src/run_pipeline.py
```

The pipeline reads `config.yaml`, trains the selected model(s), evaluates on test data, and writes artifacts.

## Run the CI Gate Locally

```bash
python src/CI_gate.py --config config.yaml
```

Exit codes:
- `0`: all gates passed
- `1`: one or more gates failed

## Run Tests

```bash
pytest -q test/
```

Run a single file:

```bash
pytest -q test/test_sources.py
```

## Configuration Guide

Main file: `config.yaml`

### Select a single model

Set `training.model_name` to one of:
- `Logistic Regression`
- `Random Forest`
- `Gradient Boosting`
- `XGBoost`

### Custom hyperparameters

Add under `training.hyperparameters`:

```yaml
training:
  model_name: XGBoost
  hyperparameters:
    n_estimators: [100, 200]
    max_depth: [3, 5]
```

The pipeline automatically prefixes parameter names with the classifier step (for example, `classifier_xgb__max_depth`) before `GridSearchCV`.

### Metric gates

Set thresholds in `evaluation.thresholds`, for example:

```yaml
evaluation:
  thresholds:
    auc:
      baseline: 0.84
      delta: 0.02
    f1:
      min: 0.70
    precision:
      min: 0.60
    recall:
      min: 0.65
```

## Artifacts Produced

Per run:
- `artifacts/runs/<timestamp>/model.joblib`
- `artifacts/runs/<timestamp>/training_summary.json`
- `artifacts/runs/<timestamp>/evaluation_report.json`
- `artifacts/runs/<timestamp>/pipeline_summary.json`

Best/latest pointers:
- `artifacts/best_model.joblib`
- `artifacts/best_metrics.json`
- `artifacts/best_run_info.json`
- `artifacts/evaluation_report.json` (top-level report used by `CI_gate.py`)

## Notes

- Project status: **Under Construction**
- In progress: FastAPI module
- In progress: AWS deployment
- In progress: Dockerization

- The workflow file in `.github/workflows/ci.yml` currently includes commands that should be aligned with the current CLI behavior in `src/run_pipeline.py`.
- If you change dataset file types (`.csv`/`.parquet`), update `config.yaml` paths accordingly.
