from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
        if self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(raw_scores, labels)
        else:
            model = LogisticRegression()
            model.fit(raw_scores.reshape(-1, 1), labels)
        self._model = model
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("calibrator must be fit before predict")
        if self.method == "isotonic":
            return self._model.predict(raw_scores)
        return self._model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
