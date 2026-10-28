import pytest

from backend.services.mock_government_apis import (
    verify_registration,
    verify_gst,
    verify_pan,
)


def test_gst_matches_synthetic_record():
    result = verify_gst(
        "SYN-GST-001090-2371",
        "DOC-000001",
    )

    assert result["status"] == "MATCHED_SYNTHETIC_RECORD"
    assert result["verified"] is True
    assert result["synthetic"] is True


def test_pan_matches_synthetic_record():
    result = verify_pan("SYN-PAN-000214-211")

    assert result["status"] == "MATCHED_SYNTHETIC_RECORD"
    assert result["verified"] is True


def test_unknown_registration_is_not_found():
    result = verify_gst("NONEXISTENT-GST-ID")

    assert result["status"] == "NOT_FOUND_IN_SYNTHETIC_DATA"
    assert result["verified"] is False


def test_empty_registration_is_rejected():
    result = verify_gst(" ")

    assert result["status"] == "INVALID_INPUT"
    assert result["verified"] is False


def test_unsupported_registry_is_rejected():
    result = verify_registration("UNKNOWN", "ABC123")

    assert result["status"] == "UNSUPPORTED_REGISTRY"
    assert result["verified"] is False