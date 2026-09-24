"""Unit tests for dataset splitter and Golden Benchmark generator."""

import pandas as pd

from parallax.data.splitter import create_golden_benchmark, create_kfold_splits


def test_create_kfold_splits():
    df = pd.DataFrame(
        {
            "id": range(100),
            "target": ["class_a"] * 50 + ["class_b"] * 50,
        }
    )
    folded = create_kfold_splits(df, target_col="target", n_splits=5, seed=42)
    assert "fold" in folded.columns
    assert set(folded["fold"].unique()) == {0, 1, 2, 3, 4}
    assert len(folded) == 100


def test_create_golden_benchmark():
    df = pd.DataFrame(
        {
            "id": range(200),
            "target": ["cat"] * 100 + ["dog"] * 100,
        }
    )
    train_df, val_df = create_golden_benchmark(
        df,
        target_col="target",
        train_size=50,
        val_size=20,
        seed=42,
    )
    assert len(train_df) == 50
    assert len(val_df) == 20
    # Overlap check
    overlap = set(train_df["id"]).intersection(set(val_df["id"]))
    assert len(overlap) == 0
