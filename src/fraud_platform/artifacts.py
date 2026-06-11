from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    decision_policy_version: str = Field(min_length=1)
    model_type: str = Field(min_length=1)


@dataclass
class ModelBundle:
    model: Any
    metadata: ModelMetadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    @property
    def feature_schema_version(self) -> str:
        return self.metadata.feature_schema_version

    def predict_raw_probability(self, features: pd.DataFrame) -> list[float]:
        probabilities = self.model.predict_proba(features)[:, 1]
        return [float(value) for value in probabilities]


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metadata.json").write_text(bundle.metadata.model_dump_json(indent=2))
    with (target / "model.pkl").open("wb") as handle:
        pickle.dump(bundle.model, handle)


def load_model_bundle(path: str | Path) -> ModelBundle:
    target = Path(path)
    metadata = ModelMetadata.model_validate_json((target / "metadata.json").read_text())
    with (target / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    return ModelBundle(model=model, metadata=metadata)
