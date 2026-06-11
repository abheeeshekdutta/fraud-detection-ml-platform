from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RawDataPaths:
    raw_dir: Path
    transaction_filename: str = "train_transaction.csv"
    identity_filename: str = "train_identity.csv"

    @property
    def transaction_path(self) -> Path:
        return self.raw_dir / self.transaction_filename

    @property
    def identity_path(self) -> Path:
        return self.raw_dir / self.identity_filename


@dataclass(frozen=True)
class DataProfile:
    row_count: int
    fraud_count: int
    fraud_rate: float
    amount_summary: dict[str, float]
    product_fraud_rates: pd.DataFrame
    identity_join_coverage: float
    missingness: pd.DataFrame
    categorical_cardinality: pd.DataFrame
    time_window_fraud_rates: pd.DataFrame
    target_leakage_columns: list[str]
    serving_incompatible_columns: list[str]


REQUIRED_TRANSACTION_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "isFraud",
}
TARGET_LEAKAGE_PATTERNS = ("fraud", "target", "label")
SERVING_INCOMPATIBLE_COLUMNS = {"isFraud"}


def check_raw_data_paths(paths: RawDataPaths) -> None:
    missing = [
        path
        for path in (paths.transaction_path, paths.identity_path)
        if not path.exists()
    ]
    if missing:
        expected = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing IEEE-CIS raw data files: {expected}. "
            f"Place train_transaction.csv and train_identity.csv in {paths.raw_dir}."
        )


def load_ieee_cis_data(paths: RawDataPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_raw_data_paths(paths)
    return pd.read_csv(paths.transaction_path), pd.read_csv(paths.identity_path)


def build_profile(
    transactions: pd.DataFrame,
    identity: pd.DataFrame | None,
    drift_windows: int = 6,
) -> DataProfile:
    _validate_transactions(transactions)
    joined = _join_identity(transactions, identity)
    row_count = len(transactions)
    fraud_count = int(transactions["isFraud"].sum())
    fraud_rate = _safe_rate(fraud_count, row_count)

    return DataProfile(
        row_count=row_count,
        fraud_count=fraud_count,
        fraud_rate=fraud_rate,
        amount_summary=_amount_summary(transactions["TransactionAmt"]),
        product_fraud_rates=_product_fraud_rates(transactions),
        identity_join_coverage=_identity_join_coverage(transactions, identity),
        missingness=_missingness(joined),
        categorical_cardinality=_categorical_cardinality(joined),
        time_window_fraud_rates=_time_window_fraud_rates(transactions, drift_windows),
        target_leakage_columns=_target_leakage_columns(transactions),
        serving_incompatible_columns=_serving_incompatible_columns(transactions),
    )


def write_profile_outputs(profile: DataProfile, reports_dir: Path, docs_path: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    profile.product_fraud_rates.to_csv(reports_dir / "product_fraud_rates.csv")
    profile.missingness.to_csv(reports_dir / "missingness.csv")
    profile.categorical_cardinality.to_csv(reports_dir / "categorical_cardinality.csv")
    profile.time_window_fraud_rates.to_csv(reports_dir / "time_window_fraud_rates.csv", index=False)
    (reports_dir / "product_fraud_rates.svg").write_text(
        _bar_chart_svg(profile.product_fraud_rates["fraud_rate"], "Product fraud rate"),
        encoding="utf-8",
    )
    (reports_dir / "time_window_fraud_rates.svg").write_text(
        _line_chart_svg(profile.time_window_fraud_rates["fraud_rate"], "Fraud-rate drift"),
        encoding="utf-8",
    )
    docs_path.write_text(render_markdown_report(profile), encoding="utf-8")


def render_markdown_report(profile: DataProfile) -> str:
    leakage = ", ".join(profile.target_leakage_columns) or "None flagged"
    serving_risks = ", ".join(profile.serving_incompatible_columns) or "None flagged"
    highest_product = (
        profile.product_fraud_rates.sort_values("fraud_rate", ascending=False).head(1).index[0]
        if not profile.product_fraud_rates.empty
        else "n/a"
    )
    return "\n".join(
        [
            "# IEEE-CIS Data Profile",
            "",
            "## Dataset Summary",
            "",
            f"- Rows profiled: {profile.row_count}",
            f"- Fraud labels: {profile.fraud_count}",
            f"- Fraud rate: {profile.fraud_rate:.4f}",
            f"- Identity join coverage: {profile.identity_join_coverage:.4f}",
            f"- Median transaction amount: {profile.amount_summary['p50']:.2f}",
            f"- Highest-risk ProductCD in this profile: {highest_product}",
            "",
            "## Leakage And Serving Checks",
            "",
            f"- Potential leakage columns: {leakage}",
            f"- Serving-incompatible columns: {serving_risks}",
            "",
            "## Modeling Implications",
            "",
            "- CatBoost should be a strong candidate because the profile preserves categorical "
            "cardinality and missingness instead of forcing early one-hot assumptions.",
            "- LightGBM remains useful for fast benchmarks, but high-cardinality categorical "
            "features should be encoded carefully based on the profile output.",
            "- Calibration should use later time windows because fraud-rate drift can make random "
            "splits overstate probability quality.",
            "- The decision policy should monitor false positives by ProductCD when product-level "
            "fraud rates differ meaningfully.",
            "",
            "## Generated Artifacts",
            "",
            "- `reports/eda/product_fraud_rates.csv`",
            "- `reports/eda/missingness.csv`",
            "- `reports/eda/categorical_cardinality.csv`",
            "- `reports/eda/time_window_fraud_rates.csv`",
            "- `reports/eda/product_fraud_rates.svg`",
            "- `reports/eda/time_window_fraud_rates.svg`",
            "",
        ]
    )


def _validate_transactions(transactions: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns))
    if missing:
        raise ValueError(f"missing required transaction columns: {missing}")
    if not set(transactions["isFraud"].dropna().unique()).issubset({0, 1}):
        raise ValueError("isFraud must contain only 0 and 1")


def _join_identity(transactions: pd.DataFrame, identity: pd.DataFrame | None) -> pd.DataFrame:
    if identity is None or identity.empty:
        return transactions.copy()
    return transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")


def _safe_rate(numerator: int | float, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _amount_summary(amounts: pd.Series) -> dict[str, float]:
    quantiles = amounts.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(amounts.mean()),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(amounts.max()),
    }


def _product_fraud_rates(transactions: pd.DataFrame) -> pd.DataFrame:
    grouped = transactions.groupby("ProductCD", dropna=False)["isFraud"].agg(
        ["count", "sum", "mean"]
    )
    return grouped.rename(columns={"sum": "fraud_count", "mean": "fraud_rate"}).sort_index()


def _identity_join_coverage(transactions: pd.DataFrame, identity: pd.DataFrame | None) -> float:
    if identity is None or identity.empty:
        return 0.0
    matched = transactions["TransactionID"].isin(identity["TransactionID"]).sum()
    return _safe_rate(int(matched), len(transactions))


def _missingness(frame: pd.DataFrame) -> pd.DataFrame:
    missing_counts = frame.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_rate": missing_counts / len(frame),
        }
    )
    return summary.sort_values(["missing_rate", "missing_count"], ascending=False)


def _categorical_cardinality(frame: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = frame.select_dtypes(include=["object", "category", "bool"]).columns
    summary = pd.DataFrame(
        {
            "unique_values": {
                column: int(frame[column].nunique(dropna=True)) for column in categorical_columns
            }
        }
    )
    return summary.sort_values("unique_values", ascending=False)


def _time_window_fraud_rates(transactions: pd.DataFrame, drift_windows: int) -> pd.DataFrame:
    ordered = transactions.sort_values("TransactionDT").reset_index(drop=True).copy()
    ordered["time_window"] = pd.qcut(
        ordered.index,
        q=min(drift_windows, len(ordered)),
        labels=False,
        duplicates="drop",
    )
    grouped = ordered.groupby("time_window", dropna=False).agg(
        row_count=("isFraud", "size"),
        fraud_count=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean"),
        transaction_dt_min=("TransactionDT", "min"),
        transaction_dt_max=("TransactionDT", "max"),
    )
    return grouped.reset_index()


def _target_leakage_columns(transactions: pd.DataFrame) -> list[str]:
    flagged: list[str] = []
    for column in transactions.columns:
        if column == "isFraud":
            continue
        normalized = column.lower()
        if any(pattern in normalized for pattern in TARGET_LEAKAGE_PATTERNS):
            flagged.append(column)
    return sorted(flagged)


def _serving_incompatible_columns(transactions: pd.DataFrame) -> list[str]:
    return sorted(
        column for column in transactions.columns if column in SERVING_INCOMPATIBLE_COLUMNS
    )


def _bar_chart_svg(values: pd.Series, title: str) -> str:
    width = 640
    bar_height = 24
    padding = 36
    height = padding * 2 + max(1, len(values)) * (bar_height + 12)
    max_value = max(float(values.max()) if not values.empty else 0.0, 0.001)
    bars: list[str] = []
    for idx, (label, value) in enumerate(values.items()):
        y = padding + idx * (bar_height + 12)
        bar_width = int((float(value) / max_value) * 420)
        bars.append(
            f'<text x="16" y="{y + 17}" font-size="12">{label}</text>'
            f'<rect x="150" y="{y}" width="{bar_width}" height="{bar_height}" fill="#2563eb" />'
            f'<text x="{160 + bar_width}" y="{y + 17}" font-size="12">{float(value):.4f}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="16" y="24" font-size="16" font-weight="bold">{title}</text>'
        f'{"".join(bars)}</svg>'
    )


def _line_chart_svg(values: pd.Series, title: str) -> str:
    width = 640
    height = 240
    if values.empty:
        points = ""
    else:
        max_value = max(float(values.max()), 0.001)
        step = 520 / max(1, len(values) - 1)
        coords = [
            f"{60 + idx * step},{190 - (float(value) / max_value) * 150}"
            for idx, value in enumerate(values)
        ]
        points = " ".join(coords)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="16" y="24" font-size="16" font-weight="bold">{title}</text>'
        '<line x1="60" y1="190" x2="600" y2="190" stroke="#334155" />'
        '<line x1="60" y1="40" x2="60" y2="190" stroke="#334155" />'
        f'<polyline points="{points}" fill="none" stroke="#dc2626" stroke-width="3" />'
        "</svg>"
    )
