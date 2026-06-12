from __future__ import annotations

import numpy as np

from fraud_platform.metrics import (
    calibration_error,
    evaluate_decision_thresholds,
    evaluate_threshold_grid,
    expected_fraud_utility,
    recall_at_min_precision,
)


def test_recall_at_min_precision_returns_best_supported_recall() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.05, 0.90, 0.80, 0.40, 0.70])

    recall = recall_at_min_precision(y_true, scores, min_precision=0.75)

    assert recall == 1.0


def test_expected_fraud_utility_rewards_caught_fraud_and_penalizes_false_blocks() -> None:
    y_true = np.array([0, 1, 1, 0])
    decisions = np.array(["approve", "block", "review", "block"])

    utility = expected_fraud_utility(
        y_true,
        decisions,
        fraud_loss=100.0,
        review_cost=5.0,
        false_block_cost=25.0,
    )

    assert utility == 70.0


def test_evaluate_decision_thresholds_reports_policy_operating_metrics() -> None:
    y_true = np.array([0, 1, 1, 0, 0])
    probabilities = np.array([0.05, 0.25, 0.85, 0.90, 0.45])

    report = evaluate_decision_thresholds(
        y_true,
        probabilities,
        approve_threshold=0.10,
        block_threshold=0.80,
        fraud_loss=100.0,
        review_cost=5.0,
        false_block_cost=25.0,
    )

    assert report["approve_count"] == 1
    assert report["review_count"] == 2
    assert report["block_count"] == 2
    assert report["approve_rate"] == 0.2
    assert report["block_precision"] == 0.5
    assert report["block_recall"] == 0.5
    assert report["review_fraud_capture_rate"] == 0.5
    assert report["false_block_rate"] == 1 / 3
    assert report["expected_utility"] == 65.0


def test_evaluate_threshold_grid_sorts_by_expected_utility() -> None:
    y_true = np.array([0, 1, 1, 0])
    probabilities = np.array([0.35, 0.55, 0.95, 0.80])

    reports = evaluate_threshold_grid(
        y_true,
        probabilities,
        approve_thresholds=[0.10, 0.40],
        block_thresholds=[0.60, 0.90],
        fraud_loss=100.0,
        review_cost=5.0,
        false_block_cost=25.0,
    )

    assert reports[0]["approve_threshold"] == 0.4
    assert reports[0]["block_threshold"] == 0.9
    assert reports[0]["expected_utility"] == 90.0


def test_calibration_error_bins_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    error = calibration_error(y_true, probabilities, n_bins=2)

    assert error < 0.2
