"""
Model builders for Pipeline A (naive replication of the source article) and
Pipeline B (corrected methodology: deduplicated, stratified, scaled where
needed).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def build_naive_models(random_seed: int = RANDOM_SEED) -> dict:
    """Replicate the source's model configuration using sklearn defaults.

    The source's KNN notebooks call ``KNeighborsClassifier()`` with no
    arguments, which uses sklearn's default ``n_neighbors=5``. We use k=5
    here to faithfully reproduce the source's executable behaviour (its
    README ambiguously describes the method as "k=1 KNN").
    """
    return {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_seed, n_jobs=-1
        ),
    }


def build_corrected_models(random_seed: int = RANDOM_SEED) -> dict:
    """Corrected pipeline models.

    KNN retains k=5 (same as the naive replication) but is wrapped in a
    sklearn Pipeline so StandardScaler is fit exclusively on the training
    fold during cross-validation, preventing test-set leakage. Tree models
    are scale-invariant and do not need a scaler wrapper.
    """
    return {
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=random_seed, max_depth=12),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_seed, n_jobs=-1, max_depth=16
        ),
    }


def train_and_predict(X_tr, X_te, y_tr, models: dict) -> dict:
    """Fit each model on training data and return a dict name -> predictions on X_te."""
    return {name: model.fit(X_tr, y_tr).predict(X_te) for name, model in models.items()}
