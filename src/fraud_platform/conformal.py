from __future__ import annotations

import numpy as np


class SplitConformalClassifier:
    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.threshold_: float | None = None

    def fit(self, fraud_probabilities: np.ndarray, labels: np.ndarray) -> SplitConformalClassifier:
        true_class_probability = np.where(labels == 1, fraud_probabilities, 1 - fraud_probabilities)
        nonconformity = 1 - true_class_probability
        self.threshold_ = float(np.quantile(nonconformity, 1 - self.alpha, method="higher"))
        return self

    def predict_sets(self, fraud_probabilities: np.ndarray) -> list[list[str]]:
        if self.threshold_ is None:
            raise RuntimeError("conformal classifier must be fit before predict_sets")
        prediction_sets: list[list[str]] = []
        for probability in fraud_probabilities:
            labels: list[str] = []
            if 1 - probability >= 1 - self.threshold_:
                labels.append("legit")
            if probability >= 1 - self.threshold_:
                labels.append("fraud")
            prediction_sets.append(labels or ["legit", "fraud"])
        return prediction_sets
