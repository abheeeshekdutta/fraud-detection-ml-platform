from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from fraud_platform.policy import DecisionPolicy, PolicyConfig, load_policy


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


def test_policy_approves_at_approve_threshold_boundary() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.2, prediction_set=["legit"])

    assert result.decision == "approve"
    assert result.uncertainty == "low"


def test_policy_blocks_at_block_threshold_boundary() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.8, prediction_set=["fraud"])

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


def test_policy_rejects_duplicate_prediction_labels() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    with pytest.raises(ValueError, match="duplicate"):
        policy.decide(calibrated_probability=0.95, prediction_set=["fraud", "fraud"])


def test_policy_rejects_invalid_prediction_labels() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    with pytest.raises(ValueError, match="prediction_set"):
        policy.decide(calibrated_probability=0.95, prediction_set=["chargeback"])


def test_policy_reviews_probability_threshold_mismatch() -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    result = policy.decide(calibrated_probability=0.7, prediction_set=["fraud"])

    assert result.decision == "review"
    assert result.uncertainty == "medium"


def test_policy_config_rejects_extra_yaml_keys(tmp_path) -> None:
    policy_path = tmp_path / "decision_policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "version: v1",
                "approve_threshold: 0.2",
                "block_treshold: 0.9",
            ]
        )
    )

    with pytest.raises(ValidationError):
        load_policy(policy_path)


def test_load_policy_rejects_yaml_list(tmp_path) -> None:
    policy_path = tmp_path / "decision_policy.yaml"
    policy_path.write_text("- version: v1\n")

    with pytest.raises(ValueError, match="policy YAML must contain a mapping"):
        load_policy(policy_path)


def test_load_policy_wraps_yaml_parser_errors(tmp_path) -> None:
    policy_path = tmp_path / "decision_policy.yaml"
    policy_path.write_text("version: [unterminated\n")

    with pytest.raises(ValueError, match=f"failed to parse policy YAML at {policy_path}"):
        load_policy(policy_path)


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan])
def test_policy_rejects_invalid_calibrated_probability(probability: float) -> None:
    policy = DecisionPolicy(PolicyConfig(version="v1", approve_threshold=0.2, block_threshold=0.8))

    with pytest.raises(ValueError, match="calibrated_probability"):
        policy.decide(calibrated_probability=probability, prediction_set=["legit"])
