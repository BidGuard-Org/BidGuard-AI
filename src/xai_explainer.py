
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bidguard_risk_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "bidguard_risk_scaler.joblib"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "risk_dataset.csv"

FEATURE_DESCRIPTIONS = {
    "total_requirements": "Total tender requirements",
    "mandatory_requirements": "Number of mandatory requirements",
    "compliant_count": "Requirements marked compliant",
    "partially_compliant_count": "Partially compliant requirements",
    "non_compliant_count": "Non-compliant requirements",
    "insufficient_evidence_count": "Requirements with insufficient evidence",
    "compliance_percentage": "Overall compliance percentage",
    "missing_document_count": "Missing documents",
    "expired_document_count": "Expired documents",
    "invalid_document_count": "Invalid documents",
    "inconsistency_count": "Cross-document inconsistencies",
    "mandatory_failure_count": "Mandatory requirement failures",
    "financial_requirement_failure": "Financial requirement failure flag",
    "experience_requirement_failure": "Experience requirement failure flag",
    "registration_requirement_failure": "Registration requirement failure flag",
    "authorization_requirement_failure": "Authorization requirement failure flag",
    "evidence_confidence_average": "Average evidence confidence",
    "document_quality_score": "Document quality score",
}


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Risk model or scaler file is missing.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = list(getattr(model, "feature_names_in_", []))

    if not features or not hasattr(model, "coef_"):
        raise TypeError("Expected a trained linear model with feature names.")

    scaler_features = list(getattr(scaler, "feature_names_in_", []))
    if scaler_features and features != scaler_features:
        raise ValueError("Model and scaler feature orders do not match.")

    if len(features) != scaler.n_features_in_:
        raise ValueError("Model and scaler feature counts do not match.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Background dataset missing: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing features: {missing}")

    background = data[features].apply(pd.to_numeric, errors="raise")
    background = background.replace([np.inf, -np.inf], np.nan).dropna()

    if background.empty:
        raise ValueError("No valid rows available for SHAP background data.")

    # Use a representative sample to keep explanation efficient.
    background = background.head(100)
    background_scaled = pd.DataFrame(
        scaler.transform(background),
        columns=features,
    )

    explainer = shap.LinearExplainer(model, background_scaled)

    return model, scaler, features, explainer


def explain_bid(row, top_n=3):
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    model, scaler, features, explainer = load_artifacts()

    if isinstance(row, pd.Series):
        row = row.to_dict()

    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    values = pd.DataFrame(
        [[row[f] for f in features]],
        columns=features,
    ).apply(pd.to_numeric, errors="raise")

    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Model features must be finite numeric values.")

    scaled = pd.DataFrame(scaler.transform(values), columns=features)

    predicted_class = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    class_index = list(model.classes_).index(predicted_class)

    shap_result = explainer(scaled)
    shap_values = np.asarray(shap_result.values)

    # Multiclass SHAP output: samples x features x classes.
    if shap_values.ndim == 3:
        contributions = shap_values[0, :, class_index]
    elif shap_values.ndim == 2:
        contributions = shap_values[0]
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {shap_values.shape}"
        )

    explanation = pd.DataFrame({
        "feature": features,
        "contribution": contributions,
    })
    explanation["description"] = explanation["feature"].map(
        FEATURE_DESCRIPTIONS
    ).fillna(explanation["feature"])

    positive = (
        explanation[explanation["contribution"] > 0]
        .sort_values("contribution", ascending=False)
        .head(top_n)
    )
    negative = (
        explanation[explanation["contribution"] < 0]
        .sort_values("contribution")
        .head(top_n)
    )

    return {
        "predicted_risk_level": str(predicted_class),
        "prediction_probabilities": {
            str(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        },
        "positive_drivers": positive.to_dict(orient="records"),
        "negative_drivers": negative.to_dict(orient="records"),
        "explanation_method": "SHAP_LinearExplainer",
        "interpretation_note": (
            "SHAP values explain contributions to the model's class score; "
            "they are not causal effects or percentage-point changes "
            "in predicted probability."
        ),
    }


if __name__ == "__main__":
    dataset = pd.read_csv(DATASET_PATH, nrows=1)
    result = explain_bid(dataset.iloc[0])
    print(json.dumps(result, indent=2))
