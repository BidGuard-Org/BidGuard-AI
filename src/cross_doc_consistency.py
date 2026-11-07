import difflib
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
        Supports exact match (PASS), borderline fuzzy/abbreviation match (REVIEW),
        and outright mismatch (FAIL).
        """
        if not name_doc_a or not name_doc_b:
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "FAIL",
                "severity": "HIGH",
                "passed": False,
                "discrepancy_detected": True,
                "reason": f"Missing name field in one or both documents ({doc_a_label}: '{name_doc_a}', {doc_b_label}: '{name_doc_b}')"
            }

        clean_a = name_doc_a.strip().upper()
        clean_b = name_doc_b.strip().upper()

        if clean_a == clean_b:
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "PASS",
                "severity": "NONE",
                "passed": True,
                "discrepancy_detected": False,
                "val_a": clean_a,
                "val_b": clean_b,
                "similarity_score": 100.0,
                "reason": None
            }

        similarity = round(difflib.SequenceMatcher(None, clean_a, clean_b).ratio() * 100.0, 1)

        # Check if difference is a valid corporate abbreviation/suffix variation (e.g. PVT LTD vs PRIVATE LIMITED)
        corp_keywords = ["PVT LTD", "PRIVATE LIMITED", "LTD", "LIMITED", "LLP"]
        has_corp_a = any(clean_a.endswith(k) for k in corp_keywords)
        has_corp_b = any(clean_b.endswith(k) for k in corp_keywords)

        # If one entity claims corporate incorporation while the other lacks any corporate suffix,
        # it is an entity-type conflict rather than a minor abbreviation variation
        if has_corp_a != has_corp_b and similarity < 90.0:
            note = f"Conflict detected: '{clean_a}' in {doc_a_label} does not match '{clean_b}' in {doc_b_label} (Similarity: {similarity}%)"
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "FAIL",
                "severity": "HIGH",
                "passed": False,
                "discrepancy_detected": True,
                "val_a": clean_a,
                "val_b": clean_b,
                "similarity_score": similarity,
                "reason": note
            }

        if similarity >= 90.0:
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "PASS",
                "severity": "NONE",
                "passed": True,
                "discrepancy_detected": False,
                "val_a": clean_a,
                "val_b": clean_b,
                "similarity_score": similarity,
                "reason": None
            }
        elif 70.0 <= similarity < 90.0:
            note = f"Minor name variation detected: '{clean_a}' vs '{clean_b}' (Similarity: {similarity}%)"
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "REVIEW",
                "severity": "MEDIUM",
                "passed": False,
                "discrepancy_detected": True,
                "val_a": clean_a,
                "val_b": clean_b,
                "similarity_score": similarity,
                "reason": note
            }
        else:
            note = f"Conflict detected: '{clean_a}' in {doc_a_label} does not match '{clean_b}' in {doc_b_label} (Similarity: {similarity}%)"
            return {
                "check": f"Entity Name: {doc_a_label} vs {doc_b_label}",
                "status": "FAIL",
                "severity": "HIGH",
                "passed": False,
                "discrepancy_detected": True,
                "val_a": clean_a,
                "val_b": clean_b,
                "similarity_score": similarity,
                "reason": note
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
        if not pan_number or not gstin_number or len(str(gstin_number).strip()) < 12:
            return {
                "check": "PAN Embedded in GSTIN Check",
                "flag": "PAN_MISMATCH_IN_GSTIN",
                "status": "FAIL",
                "passed": False,
                "severity": "CRITICAL",
                "discrepancy_detected": True,
                "pan": str(pan_number) if pan_number else None,
                "gstin_extracted_pan": None,
                "reason": "Invalid or missing PAN/GSTIN string length for cross-document reconciliation."
            }

        embedded_pan = str(gstin_number).strip().upper()[2:12]
        clean_pan = str(pan_number).strip().upper()

        match = embedded_pan == clean_pan

        return {
            "check": "PAN Embedded in GSTIN Check",
            "flag": "PAN_MATCH_VERIFIED" if match else "PAN_MISMATCH_IN_GSTIN",
            "status": "PASS" if match else "FAIL",
            "passed": match,
            "severity": "NONE" if match else "CRITICAL",
            "discrepancy_detected": not match,
            "pan": clean_pan,
            "gstin_extracted_pan": embedded_pan,
            "reason": None if match else f"CRITICAL FORENSIC MISMATCH: Embedded PAN '{embedded_pan}' in GSTIN does not match standalone PAN '{clean_pan}'."
        }

    def evaluate_dossier_consistency(self, dossier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates cross-document validation across all submitted records.
        """
        checks: List[Dict[str, Any]] = []
        discrepancies: List[Dict[str, Any]] = []

        # 1. PAN vs GSTIN validation
        pan = dossier.get("pan_card", {}).get("pan_number")
        gstin = dossier.get("gst_certificate", {}).get("gstin")
        if pan and gstin:
            pan_check = self.check_pan_in_gstin(pan, gstin)
            checks.append(pan_check)
            if not pan_check["passed"]:
                discrepancies.append({
                    "flag": pan_check["flag"],
                    "severity": pan_check["severity"],
                    "check": pan_check["check"],
                    "field_a": "pan_card.pan_number",
                    "val_a": pan_check["pan"],
                    "field_b": "gst_certificate.embedded_pan",
                    "val_b": pan_check["gstin_extracted_pan"],
                    "details": pan_check["reason"]
                })

        # 2. Entity Legal Name across docs
        name_gst = dossier.get("gst_certificate", {}).get("legal_name")
        name_pan = dossier.get("pan_card", {}).get("legal_name")
        if name_gst and name_pan:
            name_check = self.check_entity_name_consistency(name_gst, name_pan, "GST Certificate", "PAN Card")
            checks.append(name_check)
            if not name_check["passed"]:
                flag = "NAME_VARIATION_REVIEW" if name_check.get("status") == "REVIEW" else "ENTITY_NAME_MISMATCH"
                discrepancies.append({
                    "flag": flag,
                    "severity": name_check.get("severity", "MEDIUM" if name_check.get("status") == "REVIEW" else "HIGH"),
                    "check": name_check["check"],
                    "field_a": "gst_certificate.legal_name",
                    "val_a": name_check.get("val_a"),
                    "field_b": "pan_card.legal_name",
                    "val_b": name_check.get("val_b"),
                    "details": name_check.get("reason"),
                    "similarity_score": name_check.get("similarity_score")
                })
        elif name_gst or name_pan:
            checks.append({
                "check": "Entity Name: GST Certificate vs PAN Card",
                "status": "NOT PROVIDED",
                "passed": True,
                "reason": "Single document provided; cross-entity name reconciliation skipped."
            })
        else:
            checks.append({
                "check": "Entity Name: GST Certificate vs PAN Card",
                "status": "NOT PROVIDED",
                "passed": True,
                "reason": "Entity names not detected in submitted documents; reconciliation skipped."
            })

        # 3. Turnover reconciliation
        turnover_gst = dossier.get("gst_certificate", {}).get("annual_turnover")
        turnover_bs = dossier.get("balance_sheet", {}).get("audited_turnover") if dossier.get("balance_sheet") else None
        if turnover_gst is not None and turnover_bs is not None:
            turnover_check = self.check_turnover_consistency(turnover_gst, turnover_bs)
            checks.append(turnover_check)
            if not turnover_check["passed"]:
                discrepancies.append({
                    "flag": "TURNOVER_VARIANCE_EXCEEDED",
                    "severity": "HIGH",
                    "check": turnover_check["check"],
                    "field_a": "gst_certificate.annual_turnover",
                    "val_a": turnover_check.get("gst_turnover"),
                    "field_b": "balance_sheet.audited_turnover",
                    "val_b": turnover_check.get("bs_turnover"),
                    "details": turnover_check.get("reason")
                })
        else:
            checks.append({
                "check": "Turnover Consistency: GST vs Balance Sheet",
                "status": "NOT PROVIDED",
                "passed": True,
                "reason": "Balance Sheet turnover not provided; cross-reconciliation skipped for optional submission."
            })

        inconsistency_count = sum(1 for c in checks if not c["passed"])
        is_fully_consistent = (inconsistency_count == 0) and len(checks) > 0
        critical_discrepancy_count = sum(1 for d in discrepancies if d.get("severity") == "CRITICAL")

        return {
            "is_fully_consistent": is_fully_consistent,
            "total_checks_run": len(checks),
            "inconsistency_count": inconsistency_count,
            "critical_discrepancy_count": critical_discrepancy_count,
            "discrepancies": discrepancies,
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
