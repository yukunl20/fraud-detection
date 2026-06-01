"""Tests for src.data_loader."""

import pandas as pd

from src.config import TARGET
from src.data_loader import get_feature_columns, prepare_modeling_df


def test_prepare_modeling_df_drops_duplicates(synthetic_df):
    df = synthetic_df.copy()
    duplicate_row = df.iloc[[0]]
    df_with_dup = pd.concat([df, duplicate_row], ignore_index=True)

    modeling_df, n_removed = prepare_modeling_df(df_with_dup)

    assert n_removed == 1
    assert len(modeling_df) == len(df)


def test_get_feature_columns_excludes_target_and_hour(synthetic_df):
    df = synthetic_df.copy()
    df["Hour"] = df["Time"] // 3600

    features = get_feature_columns(df)

    assert TARGET not in features
    assert "Hour" not in features
    assert "Time" in features
    assert "Amount" in features
    assert "V1" in features
