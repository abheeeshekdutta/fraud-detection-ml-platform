from __future__ import annotations

import numpy as np

from fraud_platform.metrics import (
    calibration_error,
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


def test_calibration_error_bins_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    error = calibration_error(y_true, probabilities, n_bins=2)

    assert error < 0.2
