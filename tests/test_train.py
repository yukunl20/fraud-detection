"""Tests for train.py CLI."""

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from train import parse_args, train_and_evaluate


def test_parse_args():
    with patch("sys.argv", ["train.py", "--model", "xgb", "--split", "time", "--quiet"]):
        args = parse_args()
    assert args.model == "xgb"
    assert args.split == "time"
    assert args.quiet is True


@pytest.mark.skipif(
    not Path("data/creditcard.csv").exists(),
    reason="creditcard.csv not available",
)
def test_train_and_evaluate_smoke(tmp_path):
    args = Namespace(
        data_dir="./data",
        model="rf",
        split="stratified",
        imbalance="class_weight",
        recall_target=0.80,
        random_state=42,
        output_dir=str(tmp_path),
        quiet=True,
    )

    summary = train_and_evaluate(args)

    assert summary["model"] == "Random Forest"
    assert "validation" in summary
    assert "test" in summary
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "model.joblib").exists()

    with (tmp_path / "metrics.json").open(encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["split"] == "stratified"
