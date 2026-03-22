import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),   
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=object)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ]
)

logreg_param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

logreg_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_logreg', LogisticRegression(max_iter=1000))
])


rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    '_min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}
rf_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_rf', RandomForestClassifier(random_state=42))
])


gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'class_weight': [None, 'balanced']
}
gb_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_gb', GradientBoostingClassifier(random_state=42))
])


xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'class_weight': [None, 'balanced']
}
xgb_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

MODELS = {
    'Logistic Regression': (logreg_pipeline, logreg_param_grid),
    'Random Forest': (rf_pipeline, rf_param_grid),
    'Gradient Boosting': (gb_pipeline, gb_param_grid),
    'XGBoost': (xgb_pipeline, xgb_param_grid),
}
