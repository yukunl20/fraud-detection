#!/usr/bin/env python3
"""Train a fraud detection model from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import DEFAULT_DATA_DIR, DEFAULT_RECALL_TARGET, RANDOM_STATE, TARGET
from src.data_loader import get_feature_columns, load_creditcard_csv, prepare_modeling_df
from src.evaluation import evaluate_classifier, print_evaluation, threshold_for_recall
from src.imbalance import smote_resample
from src.models import (
    predict_lgb_proba,
    predict_rf_proba,
    predict_xgb_proba,
    train_lightgbm,
    train_random_forest,
    train_xgboost,
)
from src.splits import get_xy, split_report, stratified_split, time_based_split

MODEL_CHOICES = ("rf", "xgb", "lgb")
SPLIT_CHOICES = ("stratified", "time")
IMBALANCE_CHOICES = ("class_weight", "smote")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a credit card fraud detection model.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing creditcard.csv")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="rf", help="Model type")
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="stratified", help="Split strategy")
    parser.add_argument(
        "--imbalance",
        choices=IMBALANCE_CHOICES,
        default="class_weight",
        help="Imbalance handling (SMOTE applies to train only)",
    )
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET, help="Recall target for threshold")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to save model and metrics JSON",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce training logs")
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> dict:
    """Run the full training pipeline and return summary metrics."""
    df = load_creditcard_csv(args.data_dir)
    modeling_df, n_duplicates_removed = prepare_modeling_df(df)
    features = get_feature_columns(modeling_df)

    split = (
        stratified_split(modeling_df, random_state=args.random_state)
        if args.split == "stratified"
        else time_based_split(modeling_df)
    )

    X_train, y_train = get_xy(split.train_df, features)
    X_val, y_val = get_xy(split.val_df, features)
    X_test, y_test = get_xy(split.test_df, features)

    print("=== Data ===")
    print(f"Rows after dedup: {len(modeling_df):,} (removed {n_duplicates_removed:,} duplicates)")
    print(split_report(modeling_df, split).to_string(index=False))

    if args.imbalance == "smote":
        X_train, y_train = smote_resample(X_train, y_train, random_state=args.random_state)
        rf_class_weight = None
    else:
        rf_class_weight = "balanced"

    print(f"\n=== Training: model={args.model}, split={args.split}, imbalance={args.imbalance} ===")

    if args.model == "rf":
        model = train_random_forest(
            X_train, y_train,
            class_weight=rf_class_weight,
            random_state=args.random_state,
        )
        y_prob_val = predict_rf_proba(model, X_val)
        y_prob_test = predict_rf_proba(model, X_test)
        model_label = "Random Forest"
    elif args.model == "xgb":
        model = train_xgboost(
            X_train, y_train, X_val, y_val, features,
            random_state=args.random_state,
            verbose_eval=False if args.quiet else 50,
        )
        y_prob_val = predict_xgb_proba(model, X_val, features)
        y_prob_test = predict_xgb_proba(model, X_test, features)
        model_label = "XGBoost"
    else:
        model = train_lightgbm(
            X_train, y_train, X_val, y_val, features,
            random_state=args.random_state,
            log_period=0 if args.quiet else 50,
        )
        y_prob_val = predict_lgb_proba(model, X_val)
        y_prob_test = predict_lgb_proba(model, X_test)
        model_label = "LightGBM"

    chosen_threshold = threshold_for_recall(y_val, y_prob_val, target_recall=args.recall_target)

    print(f"\n=== Validation threshold tuning (recall ≥ {args.recall_target:.0%}) ===")
    print(f"Chosen threshold: {chosen_threshold:.4f}")
    val_metrics = print_evaluation(y_val, y_prob_val, set_name="validation", threshold=chosen_threshold)

    print("\n=== Final test evaluation (one-time) ===")
    test_metrics = print_evaluation(
        y_test, y_prob_test,
        set_name="test",
        threshold=chosen_threshold,
    )

    summary = {
        "model": model_label,
        "split": args.split,
        "imbalance": args.imbalance,
        "chosen_threshold": chosen_threshold,
        "recall_target": args.recall_target,
        "validation": val_metrics,
        "test": test_metrics,
        "features": features,
        "n_duplicates_removed": n_duplicates_removed,
    }

    if args.output_dir:
        save_artifacts(model, summary, args.output_dir, args.model)

    return summary


def save_artifacts(model, summary: dict, output_dir: str, model_key: str) -> None:
    """Persist model and metrics to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "metrics.json"
    serializable = {
        k: v for k, v in summary.items() if k not in {"features"}
    }
    serializable["n_features"] = len(summary["features"])
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    if model_key == "rf":
        joblib.dump(model, out / "model.joblib")
    elif model_key == "xgb":
        model.save_model(out / "model.xgb")
    else:
        model.save_model(out / "model.lgb")

    print(f"\nSaved model and metrics to {out.resolve()}")


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
