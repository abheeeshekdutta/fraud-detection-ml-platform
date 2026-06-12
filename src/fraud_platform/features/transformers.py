from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    raw_feature_columns = [
        "TransactionAmt",
        "TransactionDT",
        "ProductCD",
        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceType",
        "id_31",
    ]
    derived_numeric_columns = [
        "TransactionAmt_log1p",
        "TransactionAmt_cents",
        "has_identity",
        "missing_addr1",
        "missing_P_emaildomain",
        "missing_DeviceType",
        "transaction_day",
        "transaction_hour",
    ]
    output_columns = raw_feature_columns + derived_numeric_columns
    categorical_columns = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FraudFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X.reindex(columns=self.raw_feature_columns).copy()
        amount = pd.to_numeric(features["TransactionAmt"], errors="coerce")
        transaction_dt = pd.to_numeric(features["TransactionDT"], errors="coerce").fillna(0)

        features["TransactionAmt_log1p"] = np.log1p(amount.clip(lower=0))
        features["TransactionAmt_cents"] = ((amount.fillna(0) * 100).round().astype(int) % 100)
        features["has_identity"] = (
            features[["DeviceType", "id_31"]].notna().any(axis=1).astype(int)
        )
        features["missing_addr1"] = features["addr1"].isna().astype(int)
        features["missing_P_emaildomain"] = features["P_emaildomain"].isna().astype(int)
        features["missing_DeviceType"] = features["DeviceType"].isna().astype(int)
        features["transaction_day"] = (transaction_dt // 86_400).astype(int)
        features["transaction_hour"] = ((transaction_dt // 3_600) % 24).astype(int)

        for column in self.categorical_columns:
            features[column] = features[column].fillna("missing").astype("category")
        return features.reindex(columns=self.output_columns)
