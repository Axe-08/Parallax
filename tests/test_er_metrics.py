"""Tests for Parallax ER Metric Evaluator."""

import pytest

from parallax.metrics.evaluator import (
    calculate_entity_f05,
    evaluate_blocking_candidates,
    evaluate_resolution_predictions,
)


def test_singleton_scoring():
    # Singleton with correct empty prediction scores 1.0
    assert calculate_entity_f05(set(), set()) == 1.0

    # Singleton with false prediction scores 0.0
    assert calculate_entity_f05(set(), {"S2-001"}) == 0.0


def test_f05_precision_weighting():
    true_matches = {"S2-001", "S3-002"}

    # Perfect prediction scores 1.0
    assert calculate_entity_f05(true_matches, {"S2-001", "S3-002"}) == 1.0

    # False negative: Precision = 1.0, Recall = 0.5
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    score_fn = calculate_entity_f05(true_matches, {"S2-001"})
    assert pytest.approx(score_fn, rel=1e-3) == 0.8333

    # False positive: Precision = 0.5, Recall = 1.0
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555
    score_fp = calculate_entity_f05(true_matches, {"S2-001", "S3-002", "S2-999", "S3-999"})
    assert pytest.approx(score_fp, rel=1e-3) == 0.5555

    # Verifies F0.5 penalizes false positives roughly 2x more than false negatives!
    assert score_fn > score_fp


def test_evaluate_predictions_aggregation():
    gt = {
        "S1-001": {"S2-10", "S3-20"},
        "S1-002": set(),  # singleton
    }
    preds = {
        "S1-001": {"S2-10", "S3-20"},  # 1.0
        "S1-002": set(),  # 1.0
    }
    report = evaluate_resolution_predictions(gt, preds)
    assert report.macro_f05 == 1.0
    assert report.singleton_score == 1.0
    assert report.non_singleton_f05 == 1.0


def test_evaluate_blocking_candidates():
    gt = {"S1-001": {"S2-10", "S3-20"}}
    candidates = {"S1-001": {"S2-10", "S3-20", "S2-99"}}
    report = evaluate_blocking_candidates(gt, candidates, total_target_records=100)
    assert report.pair_completeness == 1.0
    assert report.total_candidates == 3
    assert report.avg_candidates_per_s1 == 3.0
