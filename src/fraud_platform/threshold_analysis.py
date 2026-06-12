from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_platform.artifacts import load_model_bundle
from fraud_platform.metrics import evaluate_threshold_grid


def run_threshold_analysis(
    processed_dir: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    approve_thresholds: list[float],
    block_thresholds: list[float],
    fraud_loss: float,
    review_cost: float,
    false_block_cost: float,
    top_k: int,
) -> dict[str, object]:
    processed_path = Path(processed_dir)
    validation = pd.read_parquet(processed_path / "validation.parquet")
    bundle = load_model_bundle(model_dir)
    probabilities = bundle.predict_raw_probability(validation)
    reports = evaluate_threshold_grid(
        y_true=validation["isFraud"].to_numpy(),
        probabilities=pd.Series(probabilities).to_numpy(),
        approve_thresholds=approve_thresholds,
        block_thresholds=block_thresholds,
        fraud_loss=fraud_loss,
        review_cost=review_cost,
        false_block_cost=false_block_cost,
    )
    selected_reports = reports[:top_k]
    report = {
        "model_version": bundle.metadata.model_version,
        "model_type": bundle.metadata.model_type,
        "validation_rows": int(len(validation)),
        "validation_fraud_rate": float(validation["isFraud"].mean()),
        "cost_assumptions": {
            "fraud_loss": float(fraud_loss),
            "review_cost": float(review_cost),
            "false_block_cost": float(false_block_cost),
        },
        "best_thresholds": selected_reports[0] if selected_reports else None,
        "threshold_reports": selected_reports,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def _parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="artifacts/model/latest")
    parser.add_argument("--output-path", default="reports/generated/threshold_analysis.json")
    parser.add_argument("--approve-thresholds", default="0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--block-thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--fraud-loss", type=float, default=100.0)
    parser.add_argument("--review-cost", type=float, default=5.0)
    parser.add_argument("--false-block-cost", type=float, default=25.0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    run_threshold_analysis(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        output_path=args.output_path,
        approve_thresholds=_parse_thresholds(args.approve_thresholds),
        block_thresholds=_parse_thresholds(args.block_thresholds),
        fraud_loss=args.fraud_loss,
        review_cost=args.review_cost,
        false_block_cost=args.false_block_cost,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
