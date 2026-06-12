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


def evaluate_decision_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    approve_threshold: float,
    block_threshold: float,
    fraud_loss: float,
    review_cost: float,
    false_block_cost: float,
) -> dict[str, float | int]:
    if approve_threshold >= block_threshold:
        raise ValueError("approve_threshold must be lower than block_threshold")
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities).astype(float)
    if labels.shape != scores.shape:
        raise ValueError("y_true and probabilities must have the same shape")

    decisions = np.full(scores.shape, "review", dtype=object)
    decisions[scores <= approve_threshold] = "approve"
    decisions[scores >= block_threshold] = "block"

    approve_mask = decisions == "approve"
    review_mask = decisions == "review"
    block_mask = decisions == "block"
    fraud_mask = labels == 1
    legit_mask = labels == 0
    total = len(labels)
    total_fraud = int(fraud_mask.sum())
    total_legit = int(legit_mask.sum())
    block_count = int(block_mask.sum())
    approve_count = int(approve_mask.sum())

    blocked_fraud = int((block_mask & fraud_mask).sum())
    blocked_legit = int((block_mask & legit_mask).sum())
    reviewed_fraud = int((review_mask & fraud_mask).sum())
    approved_fraud = int((approve_mask & fraud_mask).sum())

    return {
        "approve_threshold": float(approve_threshold),
        "block_threshold": float(block_threshold),
        "approve_count": approve_count,
        "review_count": int(review_mask.sum()),
        "block_count": block_count,
        "approve_rate": _safe_rate(approve_count, total),
        "review_rate": _safe_rate(int(review_mask.sum()), total),
        "block_rate": _safe_rate(block_count, total),
        "block_precision": _safe_rate(blocked_fraud, block_count),
        "block_recall": _safe_rate(blocked_fraud, total_fraud),
        "review_fraud_capture_rate": _safe_rate(reviewed_fraud, total_fraud),
        "approval_fraud_rate": _safe_rate(approved_fraud, approve_count),
        "false_block_rate": _safe_rate(blocked_legit, total_legit),
        "expected_utility": float(
            expected_fraud_utility(
                labels,
                decisions,
                fraud_loss=fraud_loss,
                review_cost=review_cost,
                false_block_cost=false_block_cost,
            )
        ),
    }


def evaluate_threshold_grid(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    approve_thresholds: list[float],
    block_thresholds: list[float],
    fraud_loss: float,
    review_cost: float,
    false_block_cost: float,
) -> list[dict[str, float | int]]:
    reports = [
        evaluate_decision_thresholds(
            y_true=y_true,
            probabilities=probabilities,
            approve_threshold=approve_threshold,
            block_threshold=block_threshold,
            fraud_loss=fraud_loss,
            review_cost=review_cost,
            false_block_cost=false_block_cost,
        )
        for approve_threshold in approve_thresholds
        for block_threshold in block_thresholds
        if approve_threshold < block_threshold
    ]
    return sorted(
        reports,
        key=lambda report: (
            float(report["expected_utility"]),
            float(report["block_recall"]),
            -float(report["false_block_rate"]),
        ),
        reverse=True,
    )


def select_constrained_thresholds(
    reports: list[dict[str, float | int]],
    max_false_block_rate: float | None = None,
    max_review_rate: float | None = None,
    min_block_precision: float | None = None,
) -> dict[str, float | int] | None:
    for report in reports:
        if max_false_block_rate is not None and report["false_block_rate"] > max_false_block_rate:
            continue
        if max_review_rate is not None and report["review_rate"] > max_review_rate:
            continue
        if min_block_precision is not None and report["block_precision"] < min_block_precision:
            continue
        return report
    return None


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


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
