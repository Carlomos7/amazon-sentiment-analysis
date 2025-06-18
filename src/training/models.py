"""
ML model training with hyperparameter tuning.
Simple wrapper around your original tuning logic.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from config import settings


def tune_model(model, param_grid, model_name, X_train, y_train):
    """Tune hyperparameters using GridSearchCV - same as original function."""
    if settings.verbose:
        print(f"\nTuning {model_name}...")

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    if settings.verbose:
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def get_models_and_params():
    """Get all models with their parameter grids."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000],
            },
        },
        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {"alpha": [0.01, 0.1, 0.5, 1.0]},
        },
        "SVM": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear"],
                "probability": [True],
            },
        },
    }
