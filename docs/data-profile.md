# IEEE-CIS Data Profile

## Dataset Summary

- Rows profiled: 590540
- Fraud labels: 20663
- Fraud rate: 0.0350
- Identity join coverage: 0.2442
- Median transaction amount: 68.77
- Highest-risk ProductCD in this profile: C

## Leakage And Serving Checks

- Potential leakage columns: None flagged
- Serving-incompatible columns: isFraud

## Modeling Implications

- CatBoost should be a strong candidate because the profile preserves categorical cardinality and missingness instead of forcing early one-hot assumptions.
- LightGBM remains useful for fast benchmarks, but high-cardinality categorical features should be encoded carefully based on the profile output.
- Calibration should use later time windows because fraud-rate drift can make random splits overstate probability quality.
- The decision policy should monitor false positives by ProductCD when product-level fraud rates differ meaningfully.

## Generated Artifacts

- `reports/eda/product_fraud_rates.csv`
- `reports/eda/missingness.csv`
- `reports/eda/categorical_cardinality.csv`
- `reports/eda/time_window_fraud_rates.csv`
- `reports/eda/product_fraud_rates.svg`
- `reports/eda/time_window_fraud_rates.svg`
