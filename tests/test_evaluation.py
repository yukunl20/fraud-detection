"""Tests for src.evaluation."""

import numpy as np

from src.evaluation import (
    evaluate_classifier,
    metrics_to_df,
    threshold_for_recall,
    threshold_max_f1,
)


def test_evaluate_classifier_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_evaluate_classifier_random_guess():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    metrics = evaluate_classifier(y_true, y_prob)

    assert metrics["roc_auc"] == 0.5
    assert metrics["recall"] == 1.0  # all predicted positive at threshold 0.5


def test_metrics_to_df():
    rows = [
        {"set": "a", "pr_auc": 0.1, "roc_auc": 0.5, "precision": 0.2, "recall": 0.3, "f1": 0.25, "threshold": 0.5},
        {"set": "b", "pr_auc": 0.2, "roc_auc": 0.6, "precision": 0.3, "recall": 0.4, "f1": 0.35, "threshold": 0.5},
    ]
    df = metrics_to_df(rows)
    assert len(df) == 2


def test_threshold_helpers_return_float():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.6, 0.7, 0.85, 0.95])

    t_recall = threshold_for_recall(y_true, y_prob, target_recall=0.75)
    t_f1 = threshold_max_f1(y_true, y_prob)

    assert 0.0 <= t_recall <= 1.0
    assert 0.0 <= t_f1 <= 1.0
