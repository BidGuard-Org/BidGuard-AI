from typing import Dict, Any, List

class CrossDocumentConsistencyEngine:
    def __init__(self, turnover_tolerance_percentage: float = 5.0):
        """
        Phase 9: Cross-Document Consistency Engine
        Detects inter-document discrepancies across statutory records 
        (e.g., GST vs PAN vs Audited Balance Sheet).
        """
        print("[INFO] Initializing Cross-Document Consistency Engine...")
        self.tolerance_pct = turnover_tolerance_percentage

    def check_entity_name_consistency(self, name_doc_a: str, name_doc_b: str, doc_a_label: str, doc_b_label: str) -> Dict[str, Any]:
        """
        Checks if bidder entity legal name matches across two different documents.
        """
        if not name_doc_a or not name_doc_b:
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "FAIL",
                "passed": False,
                "reason": f"Missing name field in one or both documents ({doc_a_label}: '{name_doc_a}', {doc_b_label}: '{name_doc_b}')"
            }

        clean_a = name_doc_a.strip().upper()
        clean_b = name_doc_b.strip().upper()

        match = clean_a == clean_b
        return {
            "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
            "status": "PASS" if match else "FAIL",
            "passed": match,
            "val_a": clean_a,
            "val_b": clean_b,
            "reason": None if match else f"Conflict detected: '{clean_a}' in {doc_a_label} does not match '{clean_b}' in {doc_b_label}"
        }

    def check_turnover_consistency(self, turnover_gst: float, turnover_balance_sheet: float) -> Dict[str, Any]:
        """
        Checks if declared GST turnover matches Audited Balance Sheet turnover within tolerance.
        """
        if turnover_gst is None or turnover_balance_sheet is None:
            return {
                "check": "Turnover Consistency: GST vs Balance Sheet",
                "status": "FAIL",
                "passed": False,
                "reason": "Missing numerical turnover in one of the statutory documents."
            }

        diff = abs(turnover_gst - turnover_balance_sheet)
        base = max(turnover_gst, turnover_balance_sheet, 1.0)
        actual_pct_diff = (diff / base) * 100.0

        is_consistent = actual_pct_diff <= self.tolerance_pct

        return {
            "check": "Turnover Consistency: GST vs Balance Sheet",
            "status": "PASS" if is_consistent else "FAIL",
            "passed": is_consistent,
            "gst_turnover": turnover_gst,
            "bs_turnover": turnover_balance_sheet,
            "discrepancy_percentage": round(actual_pct_diff, 2),
            "reason": None if is_consistent else f"Discrepancy of {round(actual_pct_diff, 2)}% exceeds allowed tolerance {self.tolerance_pct}%"
        }

    def check_pan_in_gstin(self, pan_number: str, gstin_number: str) -> Dict[str, Any]:
        """
        Verifies if characters 3 to 13 of GSTIN match the 10-character PAN.
        Format: 2 digits State Code + 10 chars PAN + 1 char entity + 1 char 'Z' + 1 checksum char.
        """
        if not pan_number or not gstin_number or len(gstin_number) < 13:
            return {
                "check": "PAN Embedded in GSTIN Check",
                "status": "FAIL",
                "passed": False,
                "reason": "Invalid or missing PAN/GSTIN string length."
            }

        embedded_pan = gstin_number.strip().upper()[2:12]
        clean_pan = pan_number.strip().upper()

        match = embedded_pan == clean_pan

        return {
            "check": "PAN Embedded in GSTIN Check",
            "status": "PASS" if match else "FAIL",
            "passed": match,
            "pan": clean_pan,
            "gstin_extracted_pan": embedded_pan,
            "reason": None if match else f"Embedded PAN '{embedded_pan}' in GSTIN does not match standalone PAN '{clean_pan}'"
        }

    def evaluate_dossier_consistency(self, dossier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates cross-document validation across all submitted records.
        """
        checks: List[Dict[str, Any]] = []

        # 1. PAN vs GSTIN validation
        pan = dossier.get("pan_card", {}).get("pan_number")
        gstin = dossier.get("gst_certificate", {}).get("gstin")
        if pan and gstin:
            checks.append(self.check_pan_in_gstin(pan, gstin))

        # 2. Entity Legal Name across docs
        name_gst = dossier.get("gst_certificate", {}).get("legal_name")
        name_pan = dossier.get("pan_card", {}).get("legal_name")
        if name_gst and name_pan:
            checks.append(self.check_entity_name_consistency(name_gst, name_pan, "GST Certificate", "PAN Card"))

        # 3. Turnover reconciliation
        turnover_gst = dossier.get("gst_certificate", {}).get("annual_turnover")
        turnover_bs = dossier.get("balance_sheet", {}).get("audited_turnover")
        if turnover_gst is not None and turnover_bs is not None:
            checks.append(self.check_turnover_consistency(turnover_gst, turnover_bs))

        inconsistency_count = sum(1 for c in checks if not c["passed"])
        is_fully_consistent = (inconsistency_count == 0) and len(checks) > 0

        return {
            "is_fully_consistent": is_fully_consistent,
            "total_checks_run": len(checks),
            "inconsistency_count": inconsistency_count,
            "audit_trail": checks
        }


if __name__ == "__main__":
    print("[RUNNING] Testing Phase 9 Cross-Document Consistency Engine...")
    engine = CrossDocumentConsistencyEngine(turnover_tolerance_percentage=5.0)

    # Test Case 1: Perfectly consistent dossier
    consistent_dossier = {
        "pan_card": {"pan_number": "AAAAA0000A", "legal_name": "Apex Infotech Pvt Ltd"},
        "gst_certificate": {"gstin": "07AAAAA0000A1Z5", "legal_name": "Apex Infotech Pvt Ltd", "annual_turnover": 5.0},
        "balance_sheet": {"audited_turnover": 5.1}
    }

    report = engine.evaluate_dossier_consistency(consistent_dossier)
    print("\n--- Consistency Evaluation (Clean Dossier) ---")
    print(f"Consistent: {report['is_fully_consistent']} | Inconsistencies: {report['inconsistency_count']}")
    for item in report["audit_trail"]:
        print(f"  - [{item['status']}] {item['check']}: {item.get('reason') or 'Passed without conflict'}")

    # Test Case 2: Inconsistent dossier (PAN mismatch + Turnover mismatch)
    fraudulent_dossier = {
        "pan_card": {"pan_number": "BBBBB9999B", "legal_name": "Apex Infotech Pvt Ltd"},
        "gst_certificate": {"gstin": "07AAAAA0000A1Z5", "legal_name": "Apex Infotech Pvt Ltd", "annual_turnover": 10.0},
        "balance_sheet": {"audited_turnover": 4.0}
    }

    fraud_report = engine.evaluate_dossier_consistency(fraudulent_dossier)
    print("\n--- Consistency Evaluation (Conflicting Dossier) ---")
    print(f"Consistent: {fraud_report['is_fully_consistent']} | Inconsistencies: {fraud_report['inconsistency_count']}")
    for item in fraud_report["audit_trail"]:
        print(f"  - [{item['status']}] {item['check']}: {item.get('reason') or 'Passed without conflict'}")

    print("\n[SUCCESS] Phase 9 Consistency Engine is fully operational!")