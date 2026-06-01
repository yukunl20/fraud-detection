"""Tests for src.splits."""

from src.config import TARGET, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from src.data_loader import get_feature_columns, prepare_modeling_df
from src.splits import get_xy, split_report, stratified_split, time_based_split


def test_stratified_split_sizes(synthetic_df):
    modeling_df, _ = prepare_modeling_df(synthetic_df)
    split = stratified_split(modeling_df, random_state=42)

    n = len(modeling_df)
    assert len(split.train_df) == int(n * TRAIN_RATIO)
    assert len(split.val_df) == int(n * VAL_RATIO)
    assert len(split.test_df) == int(n * TEST_RATIO)


def test_stratified_split_preserves_fraud_rate(synthetic_df):
    modeling_df, _ = prepare_modeling_df(synthetic_df)
    base_rate = modeling_df[TARGET].mean()
    split = stratified_split(modeling_df, random_state=42)

    for name, part in [("train", split.train_df), ("val", split.val_df), ("test", split.test_df)]:
        assert abs(part[TARGET].mean() - base_rate) < 0.01, f"fraud rate drift in {name}"


def test_time_based_split_is_chronological(synthetic_df):
    modeling_df, _ = prepare_modeling_df(synthetic_df)
    split = time_based_split(modeling_df)

    assert split.train_df["Time"].max() <= split.val_df["Time"].min()
    assert split.val_df["Time"].max() <= split.test_df["Time"].min()


def test_split_report_has_four_rows(synthetic_df):
    modeling_df, _ = prepare_modeling_df(synthetic_df)
    split = stratified_split(modeling_df)
    report = split_report(modeling_df, split)

    assert len(report) == 4
    assert set(report["set"]) == {"train", "validation", "test", "full (deduped)"}


def test_get_xy_shapes(synthetic_df):
    modeling_df, _ = prepare_modeling_df(synthetic_df)
    features = get_feature_columns(modeling_df)
    split = stratified_split(modeling_df)

    X, y = get_xy(split.train_df, features)
    assert len(X) == len(y)
    assert list(X.columns) == features
