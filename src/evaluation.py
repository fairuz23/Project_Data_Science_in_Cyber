"""
Evaluation metrics used throughout the project. Malware (label = 1) is
treated as the positive class.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, y_pred, label: str) -> dict:
    """Compute the metric suite used in Section 5 (Experimental Results).

    We deliberately do not report micro-averaged F1 (the source's headline
    metric): for binary classification it is mathematically identical to
    accuracy, so reporting it as "F1" hides the precision/recall trade-off
    that matters most for malware detection.
    """
    return {
        "pipeline": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "f2": fbeta_score(y_true, y_pred, beta=2, pos_label=1),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def get_score(model, X):
    """Return a continuous malware-class score for ranking-based metrics (ROC-AUC, PR-AUC).

    Prefers predict_proba (probability of the positive/malware class); falls back to
    decision_function for models that only expose that (e.g. some SVMs).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_auc_metrics(y_true, y_score, label: str) -> dict:
    """Threshold-independent ranking metrics: ROC-AUC and PR-AUC (average precision).

    ROC-AUC summarizes true-positive-rate vs. false-positive-rate across all thresholds.
    PR-AUC (average precision) summarizes precision vs. recall across all thresholds and
    is more informative than ROC-AUC under class imbalance, since it does not credit a
    model for correctly predicting the (here, potentially large) majority class.
    """
    return {
        "model": label,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }
