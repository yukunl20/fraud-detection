"""Imbalance handling: class weights and SMOTE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE


def compute_scale_pos_weight(y_train) -> float:
    """Ratio of negative to positive samples — used by XGBoost / LightGBM."""
    y = np.asarray(y_train)
    pos = (y == 1).sum()
    if pos == 0:
        raise ValueError("Training set contains no positive (fraud) samples.")
    return float((y == 0).sum() / pos)


def smote_resample(
    X_train: pd.DataFrame,
    y_train,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to training data only.

    Never call this on validation or test sets — that causes data leakage.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
