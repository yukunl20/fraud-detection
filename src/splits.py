"""Train / validation / test splitting strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET, TIME_TRAIN_RATIO, TIME_VAL_RATIO, TRAIN_RATIO, VAL_RATIO


@dataclass
class DataSplit:
    """Container for train, validation, and test dataframes."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def split_summary(df: pd.DataFrame, name: str, target: str = TARGET) -> dict:
    """Return row count, fraud count, and fraud rate for a split."""
    return {
        "set": name,
        "rows": len(df),
        "fraud_cases": int(df[target].sum()),
        "fraud_rate": f"{df[target].mean():.4%}",
    }


def split_report(modeling_df: pd.DataFrame, split: DataSplit) -> pd.DataFrame:
    """Build a summary table for all splits."""
    return pd.DataFrame([
        split_summary(split.train_df, "train"),
        split_summary(split.val_df, "validation"),
        split_summary(split.test_df, "test"),
        split_summary(modeling_df, "full (deduped)"),
    ])


def stratified_split(
    modeling_df: pd.DataFrame,
    target: str = TARGET,
    random_state: int = RANDOM_STATE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> DataSplit:
    """
    Stratified 60/20/20 split (preserves fraud rate in each set).

    Step A: (1 - test_ratio) train+val / test_ratio test
    Step B: val_ratio/(train_ratio + val_ratio) of train+val → validation
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    train_val_df, test_df = train_test_split(
        modeling_df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=modeling_df[target],
        shuffle=True,
    )

    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df[target],
        shuffle=True,
    )

    return DataSplit(train_df=train_df, val_df=val_df, test_df=test_df)


def time_based_split(
    modeling_df: pd.DataFrame,
    time_col: str = "Time",
    train_ratio: float = TIME_TRAIN_RATIO,
    val_ratio: float = TIME_VAL_RATIO,
) -> DataSplit:
    """
    Chronological split: earliest rows → train, middle → val, latest → test.
    No shuffling — mimics predicting future transactions from past data.
    """
    time_df = modeling_df.sort_values(time_col).reset_index(drop=True)
    n = len(time_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return DataSplit(
        train_df=time_df.iloc[:train_end].copy(),
        val_df=time_df.iloc[train_end:val_end].copy(),
        test_df=time_df.iloc[val_end:].copy(),
    )


def get_xy(df: pd.DataFrame, features: list[str], target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label vector y from a dataframe."""
    return df[features], df[target]
