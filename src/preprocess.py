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
        ('num', numeric_transformer, selector(dtype_include=np.number)),
        ('cat', categorical_transformer, selector(dtype_include=['object', 'category', 'bool']))
    ]
)

logreg_param_grid = {
    'classifier_logreg__C': [0.1, 1, 10], # Inverse of regularization strength
    'classifier_logreg__penalty': ['l1', 'l2'],
    'classifier_logreg__solver': ['liblinear', 'saga']
}

logreg_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_logreg', LogisticRegression(max_iter=1000))
])


rf_param_grid = {
    'classifier_rf__n_estimators': [100, 200], # Number of trees in the forest
    'classifier_rf__max_depth': [None, 10, 20],
    'classifier_rf__min_samples_split': [2, 5], # Minimum number of samples required to split an internal node
    'classifier_rf__min_samples_leaf': [1, 2], # Minimum number of samples required to be at a leaf node
    'classifier_rf__class_weight': [None, 'balanced'] # Adjust weights inversely proportional to class frequencies
}
rf_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_rf', RandomForestClassifier(random_state=42))
])


gb_param_grid = {
    'classifier_gb__n_estimators': [100, 200], # Number of boosting stages to perform, meaning how many small trees are added sequentially.
    'classifier_gb__learning_rate': [0.01, 0.1],
    'classifier_gb__max_depth': [3, 5],
    #'classifier_gb__class_weight': [None, 'balanced'] does not exist for GradientBoostingClassifier
}
gb_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_gb', GradientBoostingClassifier(random_state=42))
])


xgb_param_grid = {
    'classifier_xgb__n_estimators': [100, 200],
    'classifier_xgb__learning_rate': [0.01, 0.1],
    'classifier_xgb__max_depth': [3, 5],
    'classifier_xgb__min_child_weight': [1, 3], # Minimum sum of instance weight (hessian) needed in a child. Higher values prevent overfitting by requiring more samples in a leaf.
    'classifier_xgb__gamma': [0.0, 0.2], # minimum loss reduction required to make a split
    'classifier_xgb__subsample': [0.8, 1.0], # fraction of samples to be used for fitting the individual base learners
    'classifier_xgb__colsample_bytree': [0.8, 1.0] # fraction of features sampled for each tree
    #'classifier_xgb__max_cat_threshold': [32, 64]
}
xgb_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('classifier_xgb', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))#, use_label_encoder=False
])

MODELS = {
    'Logistic Regression': (logreg_pipeline, logreg_param_grid),
    'Random Forest': (rf_pipeline, rf_param_grid),
    'Gradient Boosting': (gb_pipeline, gb_param_grid),
    'XGBoost': (xgb_pipeline, xgb_param_grid),
}
