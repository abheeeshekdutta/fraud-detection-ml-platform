from __future__ import annotations

import pandas as pd

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
