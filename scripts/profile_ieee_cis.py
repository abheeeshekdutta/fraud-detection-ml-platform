from __future__ import annotations

import argparse
from pathlib import Path

from fraud_platform.data_profile import (
    RawDataPaths,
    build_profile,
    load_ieee_cis_data,
    write_profile_outputs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile IEEE-CIS fraud data.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--reports-dir", default="reports/eda")
    parser.add_argument("--docs-path", default="docs/data-profile.md")
    parser.add_argument("--drift-windows", type=int, default=6)
    args = parser.parse_args(argv)

    paths = RawDataPaths(raw_dir=Path(args.raw_dir))
    try:
        transactions, identity = load_ieee_cis_data(paths)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    profile = build_profile(transactions, identity, drift_windows=args.drift_windows)
    write_profile_outputs(profile, Path(args.reports_dir), Path(args.docs_path))


if __name__ == "__main__":
    main()
