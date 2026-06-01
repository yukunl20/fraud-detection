"""Load and prepare the credit card fraud dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_DATA_DIR, DEFAULT_DATA_FILE, EXCLUDE_FEATURES, TARGET


def load_creditcard_csv(data_dir: str = DEFAULT_DATA_DIR, filename: str = DEFAULT_DATA_FILE) -> pd.DataFrame:
    """Load the raw Kaggle creditcard CSV."""
    path = Path(data_dir) / filename
    return pd.read_csv(path)


def prepare_modeling_df(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, int]:
    """
    Prepare dataframe for modeling: optionally drop duplicate rows.

    Returns (modeling_df, n_duplicates_removed).
    """
    n_before = len(df)
    modeling_df = df.drop_duplicates().reset_index(drop=True) if drop_duplicates else df.copy()
    n_duplicates_removed = n_before - len(modeling_df)
    return modeling_df, n_duplicates_removed


def get_feature_columns(
    df: pd.DataFrame,
    target: str = TARGET,
    exclude: tuple[str, ...] = EXCLUDE_FEATURES,
) -> list[str]:
    """Return feature column names (exclude target and derived columns like Hour)."""
    return [col for col in df.columns if col not in {target, *exclude}]
