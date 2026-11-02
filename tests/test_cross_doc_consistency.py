import pytest
from src.cross_doc_consistency import CrossDocumentConsistencyEngine

@pytest.fixture
def consistency_engine():
    return CrossDocumentConsistencyEngine(turnover_tolerance_percentage=5.0)

def test_optional_balance_sheet_omitted_is_not_flagged_as_inconsistency(consistency_engine):
    """When the optional balance sheet is omitted, consistency check must pass with 0 inconsistencies."""
    dossier_no_bs = {
        "pan_card": {"pan_number": "AAAAA0000A", "legal_name": "APEX INFOTECH PVT LTD"},
        "gst_certificate": {"gstin": "07AAAAA0000A1Z5", "legal_name": "APEX INFOTECH PVT LTD", "annual_turnover": 85.0},
        "balance_sheet": {"audited_turnover": None}
    }
    report = consistency_engine.evaluate_dossier_consistency(dossier_no_bs)
    assert report["is_fully_consistent"] is True
    assert report["inconsistency_count"] == 0

    bs_check = next(c for c in report["audit_trail"] if "Turnover Consistency" in c["check"])
    assert bs_check["status"] == "NOT PROVIDED"
    assert bs_check["passed"] is True

def test_turnover_mismatch_detected_when_balance_sheet_provided(consistency_engine):
    """When balance sheet is provided and turnover differs > 5%, an inconsistency must be flagged."""
    dossier_mismatch = {
        "pan_card": {"pan_number": "AAAAA0000A", "legal_name": "APEX INFOTECH PVT LTD"},
        "gst_certificate": {"gstin": "07AAAAA0000A1Z5", "legal_name": "APEX INFOTECH PVT LTD", "annual_turnover": 100.0},
        "balance_sheet": {"audited_turnover": 40.0}
    }
    report = consistency_engine.evaluate_dossier_consistency(dossier_mismatch)
    assert report["is_fully_consistent"] is False
    assert report["inconsistency_count"] == 1

def test_pan_mismatch_in_gstin(consistency_engine):
    """When standalone PAN does not match embedded PAN in GSTIN, an inconsistency is flagged."""
    res = consistency_engine.check_pan_in_gstin("BBBBB9999B", "07AAAAA0000A1Z5")
    assert res["status"] == "FAIL"
    assert res["passed"] is False

def test_entity_name_mismatch(consistency_engine):
    """When entity legal names mismatch, check fails."""
    res = consistency_engine.check_entity_name_consistency(
        "ALPHA ENTERPRISES", "BETA LOGISTICS", "GST", "PAN"
    )
    assert res["status"] == "FAIL"
    assert res["passed"] is False
