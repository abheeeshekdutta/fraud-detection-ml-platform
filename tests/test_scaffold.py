from __future__ import annotations

import fraud_platform


def test_package_version() -> None:
    assert fraud_platform.__version__ == "0.1.0"
