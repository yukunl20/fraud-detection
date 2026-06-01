"""Tests for src.models."""

from src.models import (
    get_model_probabilities,
    predict_rf_proba,
    train_random_forest,
)


def test_train_random_forest_and_predict(synthetic_df):
    from src.data_loader import get_feature_columns, prepare_modeling_df
    from src.splits import get_xy, stratified_split

    modeling_df, _ = prepare_modeling_df(synthetic_df)
    features = get_feature_columns(modeling_df)
    split = stratified_split(modeling_df)
    X_train, y_train = get_xy(split.train_df, features)
    X_val, y_val = get_xy(split.val_df, features)

    model = train_random_forest(X_train, y_train, n_estimators=10, n_jobs=1, random_state=42)
    y_prob = predict_rf_proba(model, X_val)

    assert len(y_prob) == len(y_val)
    assert y_prob.min() >= 0.0
    assert y_prob.max() <= 1.0


def test_get_model_probabilities_requires_model():
    import pytest

    with pytest.raises(ValueError, match="rf_model is required"):
        get_model_probabilities("Random Forest", None, [], rf_model=None)
