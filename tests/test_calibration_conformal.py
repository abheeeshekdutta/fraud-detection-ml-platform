from __future__ import annotations

import numpy as np

from fraud_platform.calibration import ProbabilityCalibrator
from fraud_platform.conformal import SplitConformalClassifier


def test_probability_calibrator_maps_scores_to_probabilities() -> None:
    raw_scores = np.array([0.05, 0.20, 0.80, 0.95])
    labels = np.array([0, 0, 1, 1])
    calibrator = ProbabilityCalibrator(method="isotonic")

    calibrator.fit(raw_scores, labels)
    probabilities = calibrator.predict(raw_scores)

    assert probabilities.min() >= 0
    assert probabilities.max() <= 1
    assert probabilities[-1] >= probabilities[0]


def test_split_conformal_classifier_returns_prediction_sets() -> None:
    probabilities = np.array([0.05, 0.95, 0.50, 0.60])
    labels = np.array([0, 1, 0, 1])
    conformal = SplitConformalClassifier(alpha=0.25)

    conformal.fit(probabilities, labels)
    prediction_sets = conformal.predict_sets(np.array([0.05, 0.95, 0.50]))

    assert prediction_sets[0] == ["legit"]
    assert prediction_sets[1] == ["fraud"]
    assert prediction_sets[2] == ["legit", "fraud"]
