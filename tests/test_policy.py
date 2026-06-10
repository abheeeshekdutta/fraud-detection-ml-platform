from __future__ import annotations

from fraud_platform.policy import DecisionPolicy, PolicyConfig


def test_policy_approves_low_risk_legit_set() -> None:
    policy = DecisionPolicy(
        PolicyConfig(
            version="v1",
            approve_threshold=0.2,
            block_threshold=0.8,
            conformal_alpha=0.1,
        )
    )

    result = policy.decide(calibrated_probability=0.05, prediction_set=["legit"])

    assert result.decision == "approve"
    assert result.uncertainty == "low"


def test_policy_blocks_high_risk_fraud_set() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.95, prediction_set=["fraud"])

    assert result.decision == "block"
    assert result.uncertainty == "low"


def test_policy_reviews_ambiguous_conformal_set() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.5, prediction_set=["legit", "fraud"])

    assert result.decision == "review"
    assert result.uncertainty == "high"


def test_policy_reviews_empty_conformal_set() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.95, prediction_set=[])

    assert result.decision == "review"
    assert result.uncertainty == "high"


def test_policy_reviews_probability_threshold_mismatch() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.7, prediction_set=["fraud"])

    assert result.decision == "review"
    assert result.uncertainty == "medium"
