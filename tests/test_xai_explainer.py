import pandas as pd
import pytest

from src.xai_explainer import explain_bid, DATASET_PATH


@pytest.fixture
def sample_bid():
    data = pd.read_csv(DATASET_PATH, nrows=1)
    return data.iloc[0]


def test_prediction_returns_valid_risk_level(sample_bid):
    result = explain_bid(sample_bid)

    assert result["predicted_risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_probabilities_are_valid(sample_bid):
    result = explain_bid(sample_bid)
    probabilities = result["prediction_probabilities"]

    assert set(probabilities) == {"LOW", "MEDIUM", "HIGH"}
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_shap_explanations_return_top_drivers(sample_bid):
    result = explain_bid(sample_bid)

    assert result["explanation_method"] == "SHAP_LinearExplainer"
    assert len(result["positive_drivers"]) <= 3
    assert len(result["negative_drivers"]) <= 3


def test_missing_feature_is_rejected(sample_bid):
    sample_bid = sample_bid.drop(labels=["total_requirements"])

    with pytest.raises(ValueError, match="Missing required model features"):
        explain_bid(sample_bid)


def test_invalid_top_n_is_rejected(sample_bid):
    with pytest.raises(ValueError, match="top_n"):
        explain_bid(sample_bid, top_n=0)