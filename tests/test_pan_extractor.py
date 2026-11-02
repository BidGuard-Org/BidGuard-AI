import pytest
from src.pan_extractor import (
    is_valid_pan,
    get_pan_entity_type,
    compare_pan_with_gstin,
    extract_pan_details
)

def test_pan_syntax_validation():
    """is_valid_pan correctly identifies valid and malformed 10-character PANs."""
    assert is_valid_pan("ABCDE1234F") is True
    assert is_valid_pan("AAACI4798L") is True
    assert is_valid_pan("AAPFU0939F") is True

    # Invalid cases
    assert is_valid_pan("ABCDE12345") is False  # Last char must be alpha
    assert is_valid_pan("12345ABCDE") is False  # Leading chars must be alpha
    assert is_valid_pan("ABCD12345F") is False  # Wrong format
    assert is_valid_pan("ABCDE1234") is False   # Too short
    assert is_valid_pan("ABCDE1234FA") is False # Too long
    assert is_valid_pan(None) is False
    assert is_valid_pan("") is False

def test_pan_entity_type_derivation():
    """4th character of PAN indicates legal entity category."""
    assert get_pan_entity_type("AAACP1234F") == "Company"
    assert get_pan_entity_type("AAAPP1234F") == "Individual / Person"
    assert get_pan_entity_type("AAAFH1234F") == "Firm / Limited Liability Partnership (LLP)"
    assert get_pan_entity_type("AAATH1234F") == "Trust"
    assert get_pan_entity_type("INVALID") == "UNKNOWN"

def test_compare_pan_with_gstin_pass():
    """When standalone PAN matches characters 3 to 12 of GSTIN, check passes."""
    res = compare_pan_with_gstin("AAPFU0939F", "27AAPFU0939F1ZV")
    assert res["status"] == "PASS"
    assert res["passed"] is True
    assert res["standalone_pan"] == "AAPFU0939F"
    assert res["gstin_embedded_pan"] == "AAPFU0939F"
    assert res["reason"] is None

def test_compare_pan_with_gstin_mismatch_fails():
    """When standalone PAN does not match characters 3 to 12 of GSTIN, check fails."""
    res = compare_pan_with_gstin("BBBBB9999B", "27AAPFU0939F1ZV")
    assert res["status"] == "FAIL"
    assert res["passed"] is False
    assert "does not match" in res["reason"]

def test_compare_pan_with_gstin_missing_standalone():
    """Missing standalone PAN reports NOT_VERIFIED, not automatic fraud."""
    res = compare_pan_with_gstin(None, "27AAPFU0939F1ZV")
    assert res["status"] == "NOT_VERIFIED"
    assert res["passed"] is False
    assert "not uploaded" in res["reason"]

def test_extract_pan_details_from_ocr():
    """extract_pan_details isolates PAN and cardholder name without deriving from external GST data."""
    ocr_items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 300, 50], "conf": 0.98},
        {"text": "GOVT. OF INDIA", "bbox": [50, 55, 250, 80], "conf": 0.97},
        {"text": "APEX INFOTECH PVT LTD", "bbox": [50, 100, 350, 130], "conf": 0.92},
        {"text": "PERMANENT ACCOUNT NUMBER CARD", "bbox": [50, 140, 380, 165], "conf": 0.95},
        {"text": "AAPFU0939F", "bbox": [50, 180, 250, 210], "conf": 0.96}
    ]
    raw_text = "\n".join([item["text"] for item in ocr_items])

    details = extract_pan_details(ocr_items, raw_text)
    assert details["pan"] == "AAPFU0939F"
    assert details["is_valid_format"] is True
    assert details["entity_type"] == "Firm / Limited Liability Partnership (LLP)"
    assert details["cardholder_name"] == "APEX INFOTECH PVT LTD"
    assert details["pan_status"] == "VERIFIED"
    assert details["name_status"] == "VERIFIED"
    assert len(details["lineage"]) >= 2

def test_extract_pan_details_low_confidence_flags_pending_review():
    """Low confidence OCR results are marked PENDING_REVIEW rather than inventing values."""
    ocr_items = [
        {"text": "AAPFU0939F", "bbox": [50, 180, 250, 210], "conf": 0.50},
        {"text": "APEX INFOTECH", "bbox": [50, 100, 350, 130], "conf": 0.40}
    ]
    raw_text = "AAPFU0939F APEX INFOTECH"
    details = extract_pan_details(ocr_items, raw_text)
    assert details["pan_status"] == "PENDING_REVIEW"
    assert details["name_status"] == "PENDING_REVIEW"

def test_extract_pan_details_missing_name_not_copied_from_gst():
    """When PAN card name is missing from OCR, it remains NOT DETECTED and NOT_VERIFIED (never copied from GST)."""
    ocr_items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 300, 50], "conf": 0.98},
        {"text": "AAPFU0939F", "bbox": [50, 180, 250, 210], "conf": 0.96}
    ]
    raw_text = "INCOME TAX DEPARTMENT AAPFU0939F"
    details = extract_pan_details(ocr_items, raw_text)
    assert details["pan"] == "AAPFU0939F"
    assert details["cardholder_name"] == "NOT DETECTED"
    assert details["name_status"] == "NOT_VERIFIED"
