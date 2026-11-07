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
    """When standalone PAN does not match embedded PAN in GSTIN, a structured CRITICAL inconsistency is flagged."""
    res = consistency_engine.check_pan_in_gstin("BBBBB9999B", "07AAAAA0000A1Z5")
    assert res["status"] == "FAIL"
    assert res["passed"] is False
    assert res["flag"] == "PAN_MISMATCH_IN_GSTIN"
    assert res["severity"] == "CRITICAL"
    assert res["discrepancy_detected"] is True
    assert "CRITICAL FORENSIC MISMATCH" in res["reason"]

    # When matching, verify benign status
    match_res = consistency_engine.check_pan_in_gstin("AAAAA0000A", "07AAAAA0000A1Z5")
    assert match_res["status"] == "PASS"
    assert match_res["passed"] is True
    assert match_res["flag"] == "PAN_MATCH_VERIFIED"
    assert match_res["severity"] == "NONE"
    assert match_res["discrepancy_detected"] is False

def test_structured_discrepancy_collection(consistency_engine):
    """evaluate_dossier_consistency collects structured discrepancies and counts critical ones."""
    dossier = {
        "pan_card": {"pan_number": "BBBBB9999B", "legal_name": "APEX INFOTECH PVT LTD"},
        "gst_certificate": {"gstin": "07AAAAA0000A1Z5", "legal_name": "APEX INFOTECH PVT LTD", "annual_turnover": 100.0},
        "balance_sheet": {"audited_turnover": 40.0}
    }
    report = consistency_engine.evaluate_dossier_consistency(dossier)
    assert report["is_fully_consistent"] is False
    assert report["inconsistency_count"] == 2
    assert report["critical_discrepancy_count"] == 1
    flags = [d["flag"] for d in report["discrepancies"]]
    assert "PAN_MISMATCH_IN_GSTIN" in flags
    assert "TURNOVER_VARIANCE_EXCEEDED" in flags

def test_entity_name_mismatch(consistency_engine):
    """When entity legal names mismatch, check fails."""
    res = consistency_engine.check_entity_name_consistency(
        "ALPHA ENTERPRISES", "BETA LOGISTICS", "GST", "PAN"
    )
    assert res["status"] == "FAIL"
    assert res["passed"] is False

def test_borderline_fuzzy_name_variation_review(consistency_engine):
    """When entity names have minor abbreviation or legal suffix variation (70% - 90% match),
    status is marked as REVIEW, severity as MEDIUM, and an active discrepancy note is flagged."""
    res = consistency_engine.check_entity_name_consistency(
        "Apex Buildcon Pvt Ltd", "APEX BUILDCON PRIVATE LIMITED", "GST Certificate", "PAN Card"
    )
    assert res["status"] == "REVIEW"
    assert res["severity"] == "MEDIUM"
    assert res["passed"] is False
    assert res["discrepancy_detected"] is True
    assert 70.0 <= res["similarity_score"] < 90.0
    assert "Minor name variation detected" in res["reason"]

    # Verify inclusion in evaluate_dossier_consistency discrepancies
    dossier = {
        "pan_card": {"pan_number": "AABCA5678B", "legal_name": "APEX BUILDCON PRIVATE LIMITED"},
        "gst_certificate": {"gstin": "08AABCA5678B1ZT", "legal_name": "Apex Buildcon Pvt Ltd", "annual_turnover": 10.22},
        "balance_sheet": {"audited_turnover": 9.80}
    }
    report = consistency_engine.evaluate_dossier_consistency(dossier)
    assert report["is_fully_consistent"] is False
    assert report["inconsistency_count"] == 1
    assert len(report["discrepancies"]) == 1
    assert report["discrepancies"][0]["flag"] == "NAME_VARIATION_REVIEW"
    assert report["discrepancies"][0]["severity"] == "MEDIUM"
