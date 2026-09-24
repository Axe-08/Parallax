"""Unit tests for evaluation metrics suite."""

import pytest

from parallax.metrics.evaluator import (
    calculate_exact_match,
    calculate_f1_score,
    calculate_smape,
    evaluate_predictions,
)


def test_calculate_exact_match():
    y_true = ["10 gram", "20 kg", "5 cm"]
    y_pred = ["10 gram", "20 kg", "6 cm"]
    score = calculate_exact_match(y_true, y_pred)
    assert score == pytest.approx(2 / 3)


def test_calculate_f1_score():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "B", "B"]
    macro_f1 = calculate_f1_score(y_true, y_pred, average="macro")
    assert 0.0 < macro_f1 <= 1.0


def test_calculate_smape():
    y_true = [100.0, 200.0]
    y_pred = [100.0, 200.0]
    score = calculate_smape(y_true, y_pred)
    assert score == pytest.approx(0.0)

    # Known non-zero difference
    y_pred2 = [110.0, 190.0]
    score2 = calculate_smape(y_true, y_pred2)
    assert score2 > 0.0


def test_evaluate_predictions_composite():
    y_true = ["10", "20", "30"]
    y_pred = ["10", "25", "30"]
    metrics = evaluate_predictions(y_true, y_pred)
    assert "exact_match" in metrics
    assert "f1_macro" in metrics
    assert "smape" in metrics
    assert metrics["exact_match"] == pytest.approx(2 / 3)
