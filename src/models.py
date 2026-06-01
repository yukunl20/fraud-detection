"""Model training and prediction helpers."""

from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE
from src.imbalance import compute_scale_pos_weight


def train_random_forest(
    X_train,
    y_train,
    class_weight: str | None = "balanced",
    n_estimators: int = 100,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 4,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    features: list[str],
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_STATE,
    verbose_eval: int | bool = 50,
) -> xgb.Booster:
    """Train XGBoost with scale_pos_weight and early stopping on validation PR-AUC."""
    scale_pos_weight = compute_scale_pos_weight(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "verbosity": 0,
    }

    return xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    features: list[str],
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    random_state: int = RANDOM_STATE,
    log_period: int = 50,
) -> lgb.Booster:
    """Train LightGBM with scale_pos_weight and early stopping on average_precision."""
    scale_pos_weight = compute_scale_pos_weight(y_train)

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=features)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=features)

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 4,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": random_state,
    }

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=log_period),
        ],
    )


def predict_xgb_proba(model: xgb.Booster, X, features: list[str]):
    """Return fraud probabilities from an XGBoost booster."""
    dmatrix = xgb.DMatrix(X, feature_names=features)
    best_iter = model.best_iteration if model.best_iteration is not None else 0
    return model.predict(dmatrix, iteration_range=(0, best_iter + 1))


def predict_lgb_proba(model: lgb.Booster, X):
    """Return fraud probabilities from a LightGBM booster."""
    return model.predict(X, num_iteration=model.best_iteration)


def predict_rf_proba(model: RandomForestClassifier, X):
    """Return fraud probabilities from a Random Forest."""
    return model.predict_proba(X)[:, 1]


def get_model_probabilities(
    model_name: str,
    X,
    features: list[str],
    rf_model: RandomForestClassifier | None = None,
    xgb_model: xgb.Booster | None = None,
    lgb_model: lgb.Booster | None = None,
):
    """Get fraud probabilities from a named model."""
    if model_name == "Random Forest":
        if rf_model is None:
            raise ValueError("rf_model is required for Random Forest predictions.")
        return predict_rf_proba(rf_model, X)
    if model_name == "XGBoost":
        if xgb_model is None:
            raise ValueError("xgb_model is required for XGBoost predictions.")
        return predict_xgb_proba(xgb_model, X, features)
    if model_name == "LightGBM":
        if lgb_model is None:
            raise ValueError("lgb_model is required for LightGBM predictions.")
        return predict_lgb_proba(lgb_model, X)
    raise ValueError(f"Unknown model name: {model_name}")
