"""Evaluation metrics and threshold tuning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DEFAULT_RECALL_TARGET


def evaluate_classifier(
    y_true,
    y_prob,
    threshold: float = 0.5,
    set_name: str = "",
) -> dict:
    """Return evaluation metrics for a binary classifier using probability scores."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "set": set_name,
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def print_evaluation(
    y_true,
    y_prob,
    set_name: str = "validation",
    threshold: float = 0.5,
) -> dict:
    """Print metrics and confusion matrix; return metrics dict."""
    metrics = evaluate_classifier(y_true, y_prob, threshold, set_name)
    print(f"\n=== Evaluation: {set_name} (threshold={threshold}) ===")
    print(f"  PR-AUC (primary):  {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    print(f"\n  Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
    return metrics


def metrics_to_df(metrics_list: list[dict]) -> pd.DataFrame:
    """Convert a list of metric dicts to a DataFrame."""
    return pd.DataFrame(metrics_list)


def threshold_for_recall(
    y_true,
    y_prob,
    target_recall: float = DEFAULT_RECALL_TARGET,
) -> float:
    """
    Find the highest threshold that still achieves target recall on validation.

    Higher threshold = fewer alerts; picking the highest threshold meeting
    recall maximizes precision while hitting the recall goal.
    """
    _, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_mask = recalls[:-1] >= target_recall
    if valid_mask.any():
        return float(thresholds[np.where(valid_mask)[0][-1]])
    return float(thresholds[0])


def threshold_max_f1(y_true, y_prob) -> float:
    """Return threshold that maximizes F1 on the given set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    return float(thresholds[np.argmax(f1_scores[:-1])])
