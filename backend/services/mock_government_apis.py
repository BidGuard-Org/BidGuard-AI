"""Synthetic mock verification services for BidGuard AI.

These services do not connect to real government portals.
All results are simulated from the project's synthetic document fields.
"""

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELDS_PATH = PROJECT_ROOT / "data" / "processed" / "document_fields.csv"

SUPPORTED_REGISTRIES = {
    "GST": ("gstin", "synthetic_gstin"),
    "PAN": ("pan", "synthetic_pan"),
    "UDYAM": ("udyam", "synthetic_udyam"),
    "MSME": ("udyam", "synthetic_udyam"),
    "EPFO": ("epfo", "synthetic_epfo"),
    "ESIC": ("esic", "synthetic_esic"),
}


def _load_fields() -> pd.DataFrame:
    """Load extracted synthetic fields."""
    if not FIELDS_PATH.exists():
        raise FileNotFoundError(
            f"Document fields dataset not found: {FIELDS_PATH}"
        )

    return pd.read_csv(FIELDS_PATH, dtype=str).fillna("")


def verify_registration(
    registry: str,
    registration_number: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Simulate registration verification against extracted synthetic fields."""
    registry = registry.strip().upper()
    registration_number = registration_number.strip()

    if registry not in SUPPORTED_REGISTRIES:
        return {
            "registry": registry,
            "status": "UNSUPPORTED_REGISTRY",
            "verified": False,
            "synthetic": True,
        }

    if not registration_number:
        return {
            "registry": registry,
            "status": "INVALID_INPUT",
            "verified": False,
            "synthetic": True,
        }

    fields = _load_fields()
    accepted_names = SUPPORTED_REGISTRIES[registry]
    matches = fields[
        fields["field_name"].str.lower().isin(accepted_names)
        & fields["normalized_value"].str.casefold().eq(
            registration_number.casefold()
        )
    ]

    if document_id:
        matches = matches[matches["document_id"] == document_id]

    if matches.empty:
        return {
            "registry": registry,
            "status": "NOT_FOUND_IN_SYNTHETIC_DATA",
            "verified": False,
            "synthetic": True,
            "message": (
                "No matching registration was found in the local "
                "synthetic dataset. This is not a real-world validity check."
            ),
        }

    match = matches.iloc[0]

    return {
        "registry": registry,
        "status": "MATCHED_SYNTHETIC_RECORD",
        "verified": True,
        "synthetic": True,
        "document_id": match["document_id"],
        "field_name": match["field_name"],
        "matched_value": match["normalized_value"],
        "confidence": (
            float(match["confidence"])
            if match["confidence"]
            else None
        ),
        "message": (
            "Matched a synthetic extracted field only; "
            "not verified with a government authority."
        ),
    }


def verify_gst(registration_number: str, document_id: str | None = None):
    return verify_registration("GST", registration_number, document_id)


def verify_pan(registration_number: str, document_id: str | None = None):
    return verify_registration("PAN", registration_number, document_id)


def verify_udyam(registration_number: str, document_id: str | None = None):
    return verify_registration("UDYAM", registration_number, document_id)


def verify_epfo(registration_number: str, document_id: str | None = None):
    return verify_registration("EPFO", registration_number, document_id)


def verify_esic(registration_number: str, document_id: str | None = None):
    return verify_registration("ESIC", registration_number, document_id)