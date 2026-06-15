from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from fraud_platform.artifacts import load_model_bundle
from fraud_platform.contracts import ReasonCode

RISK_FEATURE_ORDER = ["TransactionAmt", "card1", "ProductCD", "P_emaildomain", "DeviceType"]


def fallback_reason_codes(features: pd.DataFrame, max_reasons: int = 3) -> list[ReasonCode]:
    row = features.iloc[0]
    reason_codes: list[ReasonCode] = []
    for feature in RISK_FEATURE_ORDER:
        if feature not in row or pd.isna(row[feature]):
            continue
        if feature == "TransactionAmt" and float(row[feature]) <= 100:
            continue
        reason_codes.append(ReasonCode(feature=feature, direction="increases_risk"))
        if len(reason_codes) == max_reasons:
            break
    return reason_codes


def fit_explanation_artifact(
    processed_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    max_background_rows: int = 100,
    max_explain_rows: int = 100,
    top_k: int = 20,
) -> dict[str, object]:
    processed_path = Path(processed_dir)
    validation = pd.read_parquet(processed_path / "validation.parquet")
    features = validation.drop(columns=["isFraud"], errors="ignore")
    background = features.head(max_background_rows).copy()
    explain_rows = features.head(max_explain_rows).copy()
    bundle = load_model_bundle(model_dir)

    def predict_fraud_probability(values) -> np.ndarray:
        frame = _as_feature_frame(values, features.columns)
        return np.array(bundle.predict_raw_probability(frame))

    explainer = shap.KernelExplainer(predict_fraud_probability, background)
    shap_values = explainer.shap_values(explain_rows, nsamples=max(10, len(features.columns) * 2))
    importance = np.abs(np.asarray(shap_values)).mean(axis=0)
    ranked = sorted(
        (
            {"feature": str(feature), "mean_abs_shap": float(value)}
            for feature, value in zip(features.columns, importance, strict=True)
        ),
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_version": bundle.metadata.model_version,
        "model_type": bundle.metadata.model_type,
        "method": "shap_kernel",
        "background_rows": int(len(background)),
        "explained_rows": int(len(explain_rows)),
        "top_features": ranked[:top_k],
    }
    (output / "global_shap_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _as_feature_frame(values, columns: pd.Index) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    return pd.DataFrame(values, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="artifacts/model/latest")
    parser.add_argument("--output-dir", default="artifacts/explain/latest")
    parser.add_argument("--max-background-rows", type=int, default=100)
    parser.add_argument("--max-explain-rows", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    fit_explanation_artifact(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_background_rows=args.max_background_rows,
        max_explain_rows=args.max_explain_rows,
        top_k=args.top_k,
    )
