from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_platform.artifacts import load_model_bundle


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


def save_conformal(conformal: SplitConformalClassifier, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(conformal, handle)


def load_conformal(path: str | Path) -> SplitConformalClassifier:
    with Path(path).open("rb") as handle:
        conformal = pickle.load(handle)
    if not isinstance(conformal, SplitConformalClassifier):
        raise TypeError("conformal artifact does not contain a SplitConformalClassifier")
    return conformal


def fit_conformal_artifact(
    processed_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    alpha: float = 0.1,
) -> dict[str, object]:
    processed_path = Path(processed_dir)
    calibration = pd.read_parquet(processed_path / "calibration.parquet")
    validation = pd.read_parquet(processed_path / "validation.parquet")
    bundle = load_model_bundle(model_dir)

    calibration_scores = np.array(bundle.predict_raw_probability(calibration))
    calibration_labels = calibration["isFraud"].astype(int).to_numpy()
    validation_scores = np.array(bundle.predict_raw_probability(validation))
    validation_labels = validation["isFraud"].astype(int).to_numpy()

    conformal = SplitConformalClassifier(alpha=alpha).fit(calibration_scores, calibration_labels)
    validation_sets = conformal.predict_sets(validation_scores)
    coverage = _coverage(validation_sets, validation_labels)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conformal_path = output / "conformal.pkl"
    save_conformal(conformal, conformal_path)
    summary = {
        "model_version": bundle.metadata.model_version,
        "model_type": bundle.metadata.model_type,
        "alpha": alpha,
        "calibration_rows": int(len(calibration)),
        "validation_rows": int(len(validation)),
        "conformal_path": str(conformal_path),
        "validation_conformal_coverage": coverage,
    }
    (output / "conformal_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _coverage(prediction_sets: list[list[str]], labels: np.ndarray) -> float:
    covered = []
    for prediction_set, label in zip(prediction_sets, labels, strict=True):
        label_name = "fraud" if int(label) == 1 else "legit"
        covered.append(label_name in prediction_set)
    return float(sum(covered) / len(covered)) if covered else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="artifacts/model/latest")
    parser.add_argument("--output-dir", default="artifacts/conformal/latest")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    fit_conformal_artifact(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
    )
