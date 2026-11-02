import pytest
from src.compliance_engine import ComplianceVerificationEngine

@pytest.fixture
def engine():
    return ComplianceVerificationEngine()

def test_historical_registration_date_passes(engine):
    """Statutory registration dates issued before or on the cut-off date must PASS."""
    res = engine.check_validity_date(
        detected_date_str="14/05/2024",
        tender_submission_date="01/01/2026",
        is_expiry=False
    )
    assert res["status"] == "PASS"
    assert res["passed"] is True
    assert res["reason"] is None

def test_future_registration_date_after_cutoff_fails(engine):
    """An entity registered after the cut-off date must FAIL registration validity."""
    res = engine.check_validity_date(
        detected_date_str="15/05/2026",
        tender_submission_date="01/01/2026",
        is_expiry=False
    )
    assert res["status"] == "FAIL"
    assert res["passed"] is False
    assert "after mandatory cut-off date" in res["reason"]

def test_document_expiry_date_behavior(engine):
    """When is_expiry=True, dates on or after reference date pass, and expired dates fail."""
    valid_res = engine.check_validity_date(
        detected_date_str="15/05/2027",
        tender_submission_date="01/01/2026",
        is_expiry=True
    )
    assert valid_res["status"] == "PASS"
    assert valid_res["passed"] is True

    expired_res = engine.check_validity_date(
        detected_date_str="15/05/2024",
        tender_submission_date="01/01/2026",
        is_expiry=True
    )
    assert expired_res["status"] == "FAIL"
    assert expired_res["passed"] is False
    assert "expired on" in expired_res["reason"]

def test_evaluate_bid_compliance_integration(engine):
    """Complete evaluation pipeline passes valid payload with historical registration date."""
    ocr_payload = {
        "document_id": "DOC-TEST-001",
        "extracted_fields": {
            "gstin": "07AAAAA0000A1Z5",
            "detected_date": "10/08/2023",
            "turnover": 100.0
        }
    }
    tender_rules = {
        "expected_gstin": "07AAAAA0000A1Z5",
        "min_turnover": 75.0,
        "tender_submission_date": "31/12/2026",
        "is_expiry": False
    }
    report = engine.evaluate_bid_compliance(ocr_payload, tender_rules)
    assert report["all_rules_passed"] is True
    assert report["failure_count"] == 0
