"""Shared pytest fixtures."""

import sys
from pathlib import Path

# Allow imports from project root (src package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small imbalanced dataset for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 2000
    n_fraud = 40

    data = {
        "Time": np.sort(rng.uniform(0, 172800, n)),
        "Amount": rng.exponential(50, n),
        TARGET: np.array([1] * n_fraud + [0] * (n - n_fraud)),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
