"""Tests for src.imbalance."""

import numpy as np
import pytest

from src.imbalance import compute_scale_pos_weight, smote_resample


def test_compute_scale_pos_weight():
    y = np.array([0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == 3.0


def test_compute_scale_pos_weight_raises_without_positives():
    y = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="no positive"):
        compute_scale_pos_weight(y)


def test_smote_resample_balances_training_data(synthetic_df):
    from src.data_loader import get_feature_columns, prepare_modeling_df
    from src.splits import get_xy, stratified_split

    modeling_df, _ = prepare_modeling_df(synthetic_df)
    features = get_feature_columns(modeling_df)
    split = stratified_split(modeling_df)
    X_train, y_train = get_xy(split.train_df, features)

    X_res, y_res = smote_resample(X_train, y_train, random_state=42)

    assert len(X_res) == len(y_res)
    assert len(X_res) > len(X_train)
    assert abs(y_res.mean() - 0.5) < 0.01
