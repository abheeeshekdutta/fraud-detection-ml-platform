from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [10, 20, 30, 40, 50, 60],
            "TransactionAmt": [20.0, 200.0, 35.0, 500.0, 75.0, 900.0],
            "ProductCD": ["W", "C", "W", "R", "H", "C"],
            "card1": [1001, 1002, 1001, 1003, 1004, 1002],
            "addr1": [100.0, 200.0, 100.0, None, 300.0, 200.0],
            "P_emaildomain": ["a.test", "b.test", None, "c.test", "a.test", "b.test"],
            "isFraud": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def synthetic_identity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 4],
            "DeviceType": ["desktop", "mobile", "mobile"],
            "id_31": ["chrome", "safari", "firefox"],
        }
    )
