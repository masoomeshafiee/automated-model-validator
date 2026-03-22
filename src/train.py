from sklearn.metrics import classification_report, confusion_matrix
from preprocess import MODELS
from sklearn.model_selection import  GridSearchCV, StratifiedKFold
from evaluate import evaluate_model

#it seems the XGBOOST should be handeled differently

def train_and_evaluate(x, y):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = None
    best_search = None
    best_score = -float('inf')

    for model_name, (model, param_grid) in MODELS.items():
        
        print("Training model:", model_name)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_search.fit(x,y)
        print("Best parameters for", model_name, ":", grid_search.best_params_)
        print("Best score for", model_name, ":", grid_search.best_score_)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model_name = model_name
            best_search = grid_search

    
    return {} 




