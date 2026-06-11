from __future__ import annotations

import pandas as pd

from fraud_platform.artifacts import ModelBundle, load_model_bundle
from fraud_platform.training import train_synthetic_model


def test_train_synthetic_model_writes_loadable_bundle(tmp_path) -> None:
    output_dir = tmp_path / "model"

    metadata = train_synthetic_model(output_dir)
    bundle = load_model_bundle(output_dir)

    assert metadata.model_version == "synthetic-fraud-model:1"
    assert isinstance(bundle, ModelBundle)
    assert bundle.feature_schema_version == "v1"


def test_loaded_bundle_predicts_probability(tmp_path) -> None:
    output_dir = tmp_path / "model"
    train_synthetic_model(output_dir)
    bundle = load_model_bundle(output_dir)
    rows = pd.DataFrame(
        {
            "TransactionAmt": [20.0],
            "ProductCD": ["W"],
            "card1": [1001],
            "addr1": [100.0],
            "P_emaildomain": ["a.test"],
            "DeviceType": ["desktop"],
            "id_31": ["chrome"],
        }
    )

    probability = bundle.predict_raw_probability(rows)[0]

    assert 0 <= probability <= 1
