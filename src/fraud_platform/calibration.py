from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from fraud_platform.artifacts import load_model_bundle
from fraud_platform.metrics import calibration_error


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


def save_calibrator(calibrator: ProbabilityCalibrator, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(calibrator, handle)


def load_calibrator(path: str | Path) -> ProbabilityCalibrator:
    with Path(path).open("rb") as handle:
        calibrator = pickle.load(handle)
    if not isinstance(calibrator, ProbabilityCalibrator):
        raise TypeError("calibration artifact does not contain a ProbabilityCalibrator")
    return calibrator


def fit_calibrator_artifact(
    processed_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    method: str = "isotonic",
) -> dict[str, object]:
    processed_path = Path(processed_dir)
    calibration = pd.read_parquet(processed_path / "calibration.parquet")
    validation = pd.read_parquet(processed_path / "validation.parquet")
    bundle = load_model_bundle(model_dir)

    calibration_scores = np.array(bundle.predict_raw_probability(calibration))
    calibration_labels = calibration["isFraud"].astype(int).to_numpy()
    validation_scores = np.array(bundle.predict_raw_probability(validation))
    validation_labels = validation["isFraud"].astype(int).to_numpy()

    calibrator = ProbabilityCalibrator(method=method).fit(calibration_scores, calibration_labels)
    validation_calibrated = calibrator.predict(validation_scores)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibrator_path = output / "calibrator.pkl"
    save_calibrator(calibrator, calibrator_path)

    summary = {
        "model_version": bundle.metadata.model_version,
        "model_type": bundle.metadata.model_type,
        "method": method,
        "calibration_rows": int(len(calibration)),
        "validation_rows": int(len(validation)),
        "calibrator_path": str(calibrator_path),
        "validation_raw_brier_score": float(brier_score_loss(validation_labels, validation_scores)),
        "validation_calibrated_brier_score": float(
            brier_score_loss(validation_labels, validation_calibrated)
        ),
        "validation_raw_calibration_error": calibration_error(validation_labels, validation_scores),
        "validation_calibrated_calibration_error": calibration_error(
            validation_labels, validation_calibrated
        ),
    }
    (output / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="artifacts/model/latest")
    parser.add_argument("--output-dir", default="artifacts/calibration/latest")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    args = parser.parse_args()
    fit_calibrator_artifact(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        method=args.method,
    )
