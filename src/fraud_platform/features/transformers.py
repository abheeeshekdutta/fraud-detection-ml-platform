from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    feature_columns = [
        "TransactionAmt",
        "ProductCD",
        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceType",
        "id_31",
    ]
    categorical_columns = ["ProductCD", "P_emaildomain", "DeviceType", "id_31"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FraudFeatureTransformer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = X.reindex(columns=self.feature_columns).copy()
        for column in self.categorical_columns:
            features[column] = features[column].fillna("missing").astype("category")
        return features
