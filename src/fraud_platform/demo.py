"""Prepare deterministic, credential-free local demonstration assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_platform.training import train_synthetic_model


def prepare_demo(output_dir: str | Path = "artifacts/demo", rows: int = 200) -> Path:
    if rows < 1:
        raise ValueError("rows must be positive")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_synthetic_model(target / "model")
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1, rows + 1),
            "TransactionDT": np.arange(rows) * 60.0,
            "TransactionAmt": rng.choice([15.0, 35.0, 75.0, 200.0, 500.0, 900.0], rows),
            "ProductCD": rng.choice(["W", "C", "R", "H"], rows),
            "card1": rng.choice([1001, 1002, 1003, 1004], rows),
            "addr1": rng.choice([100.0, 200.0, 300.0], rows),
            "P_emaildomain": rng.choice(["a.test", "b.test", "c.test"], rows),
            "DeviceType": rng.choice(["desktop", "mobile"], rows),
            "id_31": rng.choice(["chrome", "safari", "firefox"], rows),
        }
    )
    frame.to_parquet(target / "replay.parquet", index=False)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "data_source": "synthetic",
                "seed": 42,
                "rows": rows,
                "purpose": "Functional demonstration only; no model-quality claims.",
            },
            indent=2,
        )
        + "\n"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/demo")
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    print(f"Prepared synthetic demo assets in {prepare_demo(args.output_dir, args.rows)}")
