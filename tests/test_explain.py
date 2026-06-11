from __future__ import annotations

import pandas as pd

from fraud_platform.explain import fallback_reason_codes


def test_fallback_reason_codes_are_stable_and_analyst_readable() -> None:
    rows = pd.DataFrame(
        {
            "TransactionAmt": [900.0],
            "ProductCD": ["C"],
            "card1": [1002],
        }
    )

    reason_codes = fallback_reason_codes(rows, max_reasons=2)

    assert reason_codes[0].feature == "TransactionAmt"
    assert reason_codes[0].direction == "increases_risk"
