import pytest
import re
from src.ocr_pipeline import (
    DocumentOCRPipeline,
    is_valid_gstin,
    compute_gstin_checksum,
    validate_gstin_checksum
)

# Benchmark Test Fixture: Valid synthetic GSTIN generated with mathematically correct check digit
# State: 07 (Delhi), PAN: AAAAA0000A, Entity: 1, Default: Z, Check digit: 4
VALID_SYNTHETIC_GSTIN = "07AAAAA0000A1Z4"
# Sample document value with unverified/invalid check digit '5'
LEGACY_SYNTHETIC_GSTIN = "07AAAAA0000A1Z5"

@pytest.fixture
def ocr_pipeline():
    return DocumentOCRPipeline.__new__(DocumentOCRPipeline)

def test_gstin_checksum_calculation():
    """ISO 7064 Mod 36,10 (Luhn Mod 36) correctly calculates check digits for diverse real-world Indian GSTINs."""
    test_cases = [
        ("27AAPFU0939F1Z", "V"),
        ("29AAACI4798L1Z", "U"),
        ("27AAACI4798L1Z", "Y"),
        ("09AAACI4798L1Z", "W"),
        ("27AAACT2727Q1Z", "W"),
        ("27AAACR4995L1Z", "Q"),
        ("27AAACB2902M1Z", "T"),
        ("29AAACW0387R2Z", "I"),
        ("27AAACA0006A1Z", "S"),
        ("07AAAAA0000A1Z", "4"),
    ]
    for gstin14, expected_check in test_cases:
        assert compute_gstin_checksum(gstin14) == expected_check

def test_valid_gstin_verification():
    """Valid GSTIN patterns pass verification with check digit verification enabled."""
    assert is_valid_gstin("27AAPFU0939F1ZV", verify_checksum=True) is True
    assert is_valid_gstin("27AAACA0006A1ZS", verify_checksum=True) is True
    assert is_valid_gstin("29AAACI4798L1ZU", verify_checksum=True) is True
    # Correctly calculated synthetic benchmark fixture
    assert is_valid_gstin(VALID_SYNTHETIC_GSTIN, verify_checksum=True) is True

def test_regression_hardcoded_bypass_removed_07aaaaa0000a1z5_rejected():
    """
    Regression test: 07AAAAA0000A1Z5 has computed check digit '4'.
    The hardcoded bypass has been removed: check digit '5' must be strictly REJECTED.
    """
    assert compute_gstin_checksum("07AAAAA0000A1Z") == "4"
    assert is_valid_gstin(LEGACY_SYNTHETIC_GSTIN, verify_checksum=True) is False

def test_invalid_gstin_fails_checksum():
    """GSTIN with an invalid check digit fails verification."""
    # 27AAPFU0939F1Z check digit is 'V', '9' is invalid
    assert is_valid_gstin("27AAPFU0939F1Z9", verify_checksum=True) is False
    # 27AAACA0006A1Z check digit is 'S', '5' is invalid
    assert is_valid_gstin("27AAACA0006A1Z5", verify_checksum=True) is False

def test_invalid_state_code_rejected():
    """State codes outside statutory range (01-38, 97, 99) are rejected."""
    assert is_valid_gstin("00AAACA0006A1ZS", verify_checksum=True) is False
    assert is_valid_gstin("39AAACA0006A1ZS", verify_checksum=True) is False

def test_safe_slot_13_repair(ocr_pipeline):
    """Slot 13 OCR misreads ('2', '7', 'C', '€') are repaired to 'Z' without corrupting other slots."""
    misread_2 = list("07AAAAA0000A124")
    repaired_2 = ocr_pipeline.repair_gstin_slot(misread_2)
    assert repaired_2[13] == "Z"

    misread_c = list("07AAAAA0000A1C4")
    repaired_c = ocr_pipeline.repair_gstin_slot(misread_c)
    assert repaired_c[13] == "Z"

def test_slot_14_preserves_check_digit_s(ocr_pipeline):
    """OCR repair does not blindly overwrite check digit 'S' with '5'."""
    cand = list("27AAACA0006A1ZS")
    repaired = ocr_pipeline.repair_gstin_slot(cand)
    assert repaired[14] == "S"

def test_repair_gstin_with_checksum_permutations(ocr_pipeline):
    """Tier 1 & Tier 2 confusable repair resolves low-DPI scan confusions."""
    # 27AAACA0006A1Z has check digit 'S'. Candidate has '5' (common OCR confusion for S)
    repaired = ocr_pipeline.repair_gstin_with_checksum_permutations("27AAACA0006A1Z5")
    assert repaired == "27AAACA0006A1ZS"
    assert is_valid_gstin(repaired, verify_checksum=True) is True

    # Check that already valid GSTIN passes through unchanged
    assert ocr_pipeline.repair_gstin_with_checksum_permutations("27AAACA0006A1ZS") == "27AAACA0006A1ZS"

    # Non-destructive fallback returns structurally valid string even if checksum cannot be matched
    fallback = ocr_pipeline.repair_gstin_with_checksum_permutations("07AAAAA0000A1Z5")
    assert fallback == "07AAAAA0000A1Z5"
    assert is_valid_gstin(fallback, verify_checksum=False) is True

def test_find_gstin_preserves_exact_extracted_value(ocr_pipeline):
    """find_gstin extracts valid GSTIN verbatim without silent modifications."""
    sample_text = "INVOICE TAX REGISTRATION GSTIN: 27AAPFU0939F1ZV DATE: 12/04/2024"
    extracted = ocr_pipeline.find_gstin(sample_text)
    assert extracted == "27AAPFU0939F1ZV"

def test_find_gstin_never_silently_replaces_invalid_checksum(ocr_pipeline):
    """find_gstin never replaces an invalid check digit with a generated one; it rejects the string."""
    sample_invalid = f"INVOICE TAX REGISTRATION GSTIN: {LEGACY_SYNTHETIC_GSTIN} DATE: 12/04/2024"
    extracted = ocr_pipeline.find_gstin(sample_invalid, verify_checksum=True)
    assert extracted is None

def test_checksum_reporting_not_presented_as_government_failure():
    """
    Checksum verification results must report exact mathematical details and explicitly
    state that verification is an offline algorithmic check, not a government portal rejection.
    """
    res_failed = validate_gstin_checksum(LEGACY_SYNTHETIC_GSTIN)
    assert res_failed["is_valid"] is False
    assert res_failed["status"] == "CHECKSUM_FAILED"
    assert res_failed["expected_check_digit"] == "4"
    assert res_failed["actual_check_digit"] == "5"
    assert res_failed["is_government_verified"] is False
    assert "not a government portal rejection" in res_failed["reason"]

    res_valid = validate_gstin_checksum(VALID_SYNTHETIC_GSTIN)
    assert res_valid["is_valid"] is True
    assert res_valid["status"] == "VALID"
    assert res_valid["expected_check_digit"] == "4"
    assert res_valid["actual_check_digit"] == "4"
    assert res_valid["is_government_verified"] is False

def test_dashboard_extraction_checksum_parity():
    """Validates that dashboard regex extraction rules align with is_valid_gstin checksum verification."""
    def extract_dashboard_gstin(text: str) -> str:
        clean_t = text.upper()
        gstin_match = re.search(r'\b([0-9O]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z|2][0-9A-Z]{1})\b', clean_t)
        if gstin_match:
            cand = list(gstin_match.group(1))
            if cand[0] in ['O', 'D']: cand[0] = '0'
            if cand[1] in ['O', 'D']: cand[1] = '0'
            if cand[13] in ['2', '7', 'C', 'c', 'z', '€', 'Γé¼']:
                cand[13] = 'Z'
            cand_str = "".join(cand)
            if is_valid_gstin(cand_str, verify_checksum=True):
                return cand_str
        return "NOT DETECTED"

    assert extract_dashboard_gstin(f"GSTIN: {LEGACY_SYNTHETIC_GSTIN}") == "NOT DETECTED"
    assert extract_dashboard_gstin(f"GSTIN: {VALID_SYNTHETIC_GSTIN}") == VALID_SYNTHETIC_GSTIN
    assert extract_dashboard_gstin("GSTIN: 27AAPFU0939F1ZV") == "27AAPFU0939F1ZV"
    assert extract_dashboard_gstin("GSTIN: 27AAACA0006A1ZS") == "27AAACA0006A1ZS"
