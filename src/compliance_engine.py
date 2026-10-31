import datetime
from typing import Dict, Any, List

class ComplianceVerificationEngine:
    def __init__(self):
        """
        Phase 7: Compliance Verification Engine
        Evaluates extracted document entities against tender statutory rules.
        """
        print("[INFO] Initializing Compliance Verification Engine...")

    def check_threshold(self, extracted_value: float, required_value: float, rule_name: str = "Turnover Check") -> Dict[str, Any]:
        """
        THRESHOLD_CHECK: Verifies if extracted numerical values (e.g. Turnover, Net Worth)
        meet or exceed the mandatory tender threshold.
        """
        if extracted_value is None:
            return {
                "rule": rule_name,
                "type": "THRESHOLD_CHECK",
                "status": "FAIL",
                "reason": "Missing extracted numerical value.",
                "passed": False
            }
        
        passed = extracted_value >= required_value
        return {
            "rule": rule_name,
            "type": "THRESHOLD_CHECK",
            "status": "PASS" if passed else "FAIL",
            "extracted_value": extracted_value,
            "required_value": required_value,
            "passed": passed,
            "reason": None if passed else f"Value {extracted_value} is below mandatory threshold {required_value}."
        }

    def check_validity_date(self, detected_date_str: str, tender_submission_date: str = None, rule_name: str = "Certificate Validity") -> Dict[str, Any]:
        """
        DATE_CHECK: Verifies if certificate expiry date is after the tender submission date.
        """
        if not detected_date_str:
            return {
                "rule": rule_name,
                "type": "DATE_CHECK",
                "status": "FAIL",
                "reason": "No valid date found on document.",
                "passed": False
            }

        date_formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        doc_date = None
        for fmt in date_formats:
            try:
                doc_date = datetime.datetime.strptime(detected_date_str, fmt).date()
                break
            except ValueError:
                continue

        if not doc_date:
            return {
                "rule": rule_name,
                "type": "DATE_CHECK",
                "status": "FAIL",
                "reason": f"Unable to parse date string: {detected_date_str}",
                "passed": False
            }

        ref_date = datetime.date.today()
        if tender_submission_date:
            for fmt in date_formats:
                try:
                    ref_date = datetime.datetime.strptime(tender_submission_date, fmt).date()
                    break
                except ValueError:
                    pass

        is_valid = doc_date >= ref_date
        return {
            "rule": rule_name,
            "type": "DATE_CHECK",
            "status": "PASS" if is_valid else "FAIL",
            "document_date": str(doc_date),
            "reference_date": str(ref_date),
            "passed": is_valid,
            "reason": None if is_valid else f"Certificate expired on {doc_date} (Reference: {ref_date})."
        }

    def check_field_match(self, extracted_value: str, reference_value: str, rule_name: str = "Identity/GST Match") -> Dict[str, Any]:
        """
        FIELD_MATCH: Verifies exact or normalized match between statutory ID in OCR
        and master registration record.
        """
        if not extracted_value or not reference_value:
            return {
                "rule": rule_name,
                "type": "FIELD_MATCH",
                "status": "FAIL",
                "reason": "Missing value for comparison.",
                "passed": False
            }

        norm_extracted = extracted_value.strip().upper()
        norm_ref = reference_value.strip().upper()

        match = norm_extracted == norm_ref
        return {
            "rule": rule_name,
            "type": "FIELD_MATCH",
            "status": "PASS" if match else "FAIL",
            "extracted": norm_extracted,
            "reference": norm_ref,
            "passed": match,
            "reason": None if match else f"Extracted value '{norm_extracted}' does not match registered '{norm_ref}'."
        }

    def evaluate_bid_compliance(self, ocr_payload: Dict[str, Any], tender_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a complete document OCR payload against all tender statutory requirements.
        """
        extracted_fields = ocr_payload.get("extracted_fields", {})
        results: List[Dict[str, Any]] = []

        # 1. GST/Entity Match
        if "expected_gstin" in tender_requirements:
            gst_res = self.check_field_match(
                extracted_fields.get("gstin"),
                tender_requirements["expected_gstin"],
                rule_name="GSTIN Verification"
            )
            results.append(gst_res)

        # 2. Date/Expiry Check
        if "detected_date" in extracted_fields:
            date_res = self.check_validity_date(
                extracted_fields.get("detected_date"),
                tender_requirements.get("tender_submission_date"),
                rule_name="Document Validity Check"
            )
            results.append(date_res)

        # 3. Threshold Check (e.g. Turnover if available)
        if "min_turnover" in tender_requirements:
            turnover_res = self.check_threshold(
                extracted_fields.get("turnover"),
                tender_requirements["min_turnover"],
                rule_name="Minimum Turnover Check"
            )
            results.append(turnover_res)

        # Summary
        all_passed = all(r["passed"] for r in results) if results else False
        failure_count = sum(1 for r in results if not r["passed"])

        return {
            "document_id": ocr_payload.get("document_id"),
            "all_rules_passed": all_passed,
            "total_rules_evaluated": len(results),
            "failure_count": failure_count,
            "rule_details": results
        }


if __name__ == "__main__":
    print("[RUNNING] Testing Phase 7 Compliance Verification Engine...")
    engine = ComplianceVerificationEngine()

    # Mock test payload from Phase 6 OCR output
    mock_ocr = {
        "document_id": "DOC-0001",
        "extracted_fields": {
            "gstin": "07AAAAA0000A1Z5",
            "detected_date": "12/04/2027",
            "turnover": 6.5
        }
    }

    mock_rules = {
        "expected_gstin": "07AAAAA0000A1Z5",
        "min_turnover": 5.0,
        "tender_submission_date": "01/01/2026"
    }

    report = engine.evaluate_bid_compliance(mock_ocr, mock_rules)
    print("\n--- Compliance Evaluation Report ---")
    print(f"Overall Passed: {report['all_rules_passed']}")
    print(f"Failures: {report['failure_count']}")
    for r in report["rule_details"]:
        print(f"  - [{r['status']}] {r['rule']}: {r.get('reason') or 'Matched Successfully'}")
    print("\n[SUCCESS] Phase 7 Engine verified successfully!")