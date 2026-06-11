from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def recall_at_min_precision(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    supported = recall[precision >= min_precision]
    return float(supported.max()) if supported.size else 0.0


def expected_fraud_utility(
    y_true: np.ndarray,
    decisions: np.ndarray,
    fraud_loss: float,
    review_cost: float,
    false_block_cost: float,
) -> float:
    utility = 0.0
    for label, decision in zip(y_true, decisions, strict=True):
        if decision == "block" and label == 1:
            utility += fraud_loss
        elif decision == "block" and label == 0:
            utility -= false_block_cost
        elif decision == "review":
            utility -= review_cost
    return utility


def calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probabilities)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if not mask.any():
            continue
        bin_confidence = float(probabilities[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        error += (mask.sum() / total) * abs(bin_accuracy - bin_confidence)
    return float(error)
