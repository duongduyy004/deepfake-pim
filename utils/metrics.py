import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute binary classification metrics.
    Display order: Acc, F1, Precision, Recall, AUC.

    Args:
        y_true: ground-truth labels (0/1)
        y_prob: predicted probability for class 1
        y_pred: predicted class labels (0/1)

    Returns:
        dict with keys: acc, f1, precision, recall, auc
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # AUC requires at least 2 classes present; fall back to 0.0 if not
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }
