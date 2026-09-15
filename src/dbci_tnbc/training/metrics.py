"""Evaluation metrics."""
from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[float], y_prob: List[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Compute classification metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return metrics