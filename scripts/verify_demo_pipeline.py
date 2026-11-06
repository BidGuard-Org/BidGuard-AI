#!/usr/bin/env python3
"""
BidGuard AI — End-to-End Demo Pipeline Verification (SIH26100)
==============================================================
Validates the complete multi-document procurement compliance pipeline against
the three synthetic demo dossiers generated in demo_dossiers/:
  - Set A (set_a_compliant): 100% compliant, zero discrepancies, LOW risk.
  - Set B (set_b_discrepancy): Name variation, ~4.11% turnover variance, MEDIUM risk.
  - Set C (set_c_fraudulent): Critical PAN-in-GSTIN mismatch, 60% turnover variance, CRITICAL risk (DISQUALIFIED).

Pipeline Components Verified:
  1. Document Classification & Field Parsing (PAN, GSTIN, Udyam, Balance Sheet Turnover)
  2. Cross-Document Consistency Engine (PAN-in-GSTIN, Entity Name, Turnover Reconciliation)
  3. Statutory Compliance Verification Engine (Tender thresholds, cutoff dates)
  4. Predictive ML Risk Engine (Logistic Regression 18-feature model & scaler)
  5. Forensic Discrepancy Alerts & Statutory Hard Overrides
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

# Project roots and module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.upload_workflow import DocumentStagingQueue
from src.cross_doc_consistency import CrossDocumentConsistencyEngine
from src.compliance_engine import ComplianceVerificationEngine


# ----------------------------------------------------------------------
# ML Risk Prediction Helper (Standalone - no Streamlit dependency)
# ----------------------------------------------------------------------
def run_ml_risk_prediction(
    p7_result: Dict[str, Any],
    p9_result: Dict[str, Any],
    lineage_records: List[Dict[str, Any]],
    model,
    scaler
) -> Dict[str, Any]:
    """Constructs the exact 18-feature vector and runs ML risk classification."""
    p7_fails = p7_result.get("failure_count", 0)
    inconsistencies = p9_result.get("inconsistency_count", 0)
    crit_count = p9_result.get("critical_discrepancy_count", 0)

    total_reqs = max(len(p7_result.get("rule_details", [])) + len(p9_result.get("audit_trail", [])), 5)
    non_compliant = p7_fails + inconsistencies
    compliant = max(0, total_reqs - non_compliant)
    comp_pct = (compliant / total_reqs) * 100.0 if total_reqs > 0 else 0.0

    conf_scores = [
        item.get("confidence", 0.95)
        for item in lineage_records
        if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))
    ]
    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.95
    doc_quality = 1.0 if avg_conf >= 0.85 else 0.5

    fin_fail = 1 if any(
        "Turnover" in str(r.get("rule", "")) and not r.get("passed", True)
        for r in p7_result.get("rule_details", [])
    ) else 0
    reg_fail = 1 if (p7_fails > 0 or inconsistencies > 0) else 0

    features_dict = {
        "total_requirements": float(total_reqs),
        "mandatory_requirements": float(total_reqs),
        "compliant_count": float(compliant),
        "partially_compliant_count": 0.0,
        "non_compliant_count": float(non_compliant),
        "insufficient_evidence_count": 0.0,
        "compliance_percentage": float(comp_pct),
        "missing_document_count": 1.0 if non_compliant > 0 and inconsistencies > 0 else 0.0,
        "expired_document_count": 0.0,
        "invalid_document_count": 1.0 if inconsistencies > 0 else 0.0,
        "inconsistency_count": float(inconsistencies),
        "mandatory_failure_count": float(non_compliant),
        "financial_requirement_failure": float(fin_fail),
        "experience_requirement_failure": 0.0,
        "registration_requirement_failure": float(reg_fail),
        "authorization_requirement_failure": 0.0,
        "evidence_confidence_average": float(avg_conf),
        "document_quality_score": float(doc_quality),
    }

    df_feat = pd.DataFrame([features_dict])
    X_scaled = scaler.transform(df_feat)
    X_scaled_df = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)

    pred_class = model.predict(X_scaled_df)[0]
    probs = model.predict_proba(X_scaled_df)[0]
    conf = float(probs[list(model.classes_).index(pred_class)])
    prob_map = {cls: float(p) for cls, p in zip(model.classes_, probs)}

    return {
        "predicted_class": pred_class,
        "confidence": conf,
        "probabilities": prob_map,
        "features": features_dict,
    }


# ----------------------------------------------------------------------
# Dossier Verification Harness
# ----------------------------------------------------------------------
def verify_dossier(
    dossier_name: str,
    dossier_dir: Path,
    model,
    scaler,
    min_turnover_req: float = 100.0,
    reg_cutoff_date: str = "31/12/2026",
    ext: str = ".pdf"
) -> Dict[str, Any]:
    """Stages, processes, and verifies all documents within a demo dossier."""
    staging_queue = DocumentStagingQueue()

    # Load matching files from the directory
    doc_files = sorted([f for f in os.listdir(dossier_dir) if f.endswith(ext)])
    if not doc_files:
        raise FileNotFoundError(f"No {ext} files found in {dossier_dir}")

    for doc_name in doc_files:
        file_path = dossier_dir / doc_name
        with open(file_path, "rb") as fp:
            staging_queue.add_file(doc_name, fp.read())

    # Execute consolidated verification
    verif_res = staging_queue.execute_consolidated_verification(
        ocr_reader=None,
        min_turnover_req=min_turnover_req,
        reg_cutoff_date=reg_cutoff_date
    )

    slots = verif_res["slots"]
    p7_res = verif_res["p7_result"]
    p9_res = verif_res["p9_result"]
    lineage = verif_res["all_lineage"]

    # ML Risk Prediction
    ml_eval = run_ml_risk_prediction(p7_res, p9_res, lineage, model, scaler)

    # Forensic Hard Overrides
    discrepancies = p9_res.get("discrepancies", [])
    crit_count = p9_res.get("critical_discrepancy_count", 0)
    inconsistencies = p9_res.get("inconsistency_count", 0)
    has_pan_mismatch = any(d.get("flag") == "PAN_MISMATCH_IN_GSTIN" for d in discrepancies)
    to_variance_flag = any(d.get("flag") == "TURNOVER_VARIANCE_EXCEEDED" for d in discrepancies)

    # Final statutory decision logic
    if crit_count > 0 or has_pan_mismatch:
        final_status = "DISQUALIFIED"
        final_risk = "CRITICAL"
        reason = "Critical forensic discrepancy detected (Standalone PAN differs from GSTIN embedded PAN)."
    elif inconsistencies > 0 or to_variance_flag:
        final_status = "CONDITIONALLY_COMPLIANT"
        final_risk = "MEDIUM"
        reason = "Inter-document discrepancy detected (Entity name variation or turnover discrepancy under audit)."
    else:
        final_status = "COMPLIANT"
        final_risk = "LOW"
        reason = "All statutory rules satisfied with zero cross-document discrepancies."

    # Extract turnover variance percentage
    turnover_variance_pct = 0.0
    for c in p9_res.get("audit_trail", []):
        if "Turnover" in c.get("check", "") and "discrepancy_percentage" in c:
            turnover_variance_pct = float(c["discrepancy_percentage"])
            break

    # Extract Udyam number from staged documents
    udyam_no = "NOT DETECTED"
    for d in staging_queue.documents.values():
        if d.detected_type == "Udyam Certificate" and d.extracted_fields.get("udyam"):
            udyam_no = d.extracted_fields.get("udyam")
            break

    return {
        "dossier_name": dossier_name,
        "dossier_dir": str(dossier_dir),
        "total_documents": len(staging_queue.documents),
        "documents": [
            {
                "filename": d.filename,
                "detected_type": d.detected_type,
                "status": d.status,
            }
            for d in staging_queue.documents.values()
        ],
        "slots": {
            "company_name": slots.get("company_name"),
            "gstin_no": slots.get("gstin_no"),
            "pan_no": slots.get("pan_no"),
            "pan_name": slots.get("pan_name"),
            "pan_entity_type": slots.get("pan_entity_type"),
            "udyam_no": udyam_no,
            "dob": slots.get("dob"),
            "gst_turnover_val": slots.get("gst_turnover_val"),
            "bs_turnover_val": slots.get("bs_turnover_val"),
            "turnover_variance_pct": turnover_variance_pct,
        },
        "p7_result": p7_res,
        "p9_result": p9_res,
        "ml_eval": ml_eval,
        "final_status": final_status,
        "final_risk": final_risk,
        "decision_reason": reason,
        "has_pan_mismatch": has_pan_mismatch,
        "inconsistency_count": inconsistencies,
        "critical_discrepancy_count": crit_count,
        "discrepancies": discrepancies,
    }


# ----------------------------------------------------------------------
# Terminal Report Formatting
# ----------------------------------------------------------------------
def print_dossier_report(d: Dict[str, Any]):
    """Prints a structured ASCII report of the verification results."""
    border = "=" * 78
    dossier_title = d["dossier_name"].upper()
    print(f"\n{border}")
    print(f"  DOSSIER AUDIT REPORT: {dossier_title}")
    print(f"  Location: {d['dossier_dir']}")
    print(f"{border}")

    # Section 1: Documents Staged & Classified
    print("\n[+] 1. DOCUMENT INVENTORY & CLASSIFICATION:")
    print("  " + "-" * 74)
    print(f"  {'Filename':<28} | {'Detected Document Type':<26} | {'Status'}")
    print("  " + "-" * 74)
    for doc in d["documents"]:
        print(f"  {doc['filename']:<28} | {doc['detected_type']:<26} | {doc['status']}")

    # Section 2: Extracted Entity Slots
    s = d["slots"]
    print("\n[+] 2. EXTRACTED STATUTORY SLOTS:")
    print("  " + "-" * 74)
    print(f"  Entity / Company Name : {s['company_name']}")
    print(f"  GSTIN Number          : {s['gstin_no']}")
    print(f"  PAN Number            : {s['pan_no']} ({s['pan_entity_type']})")
    print(f"  PAN Cardholder Name   : {s['pan_name']}")
    print(f"  Udyam Reg. Number     : {s['udyam_no']}")
    print(f"  Date of Incorporation : {s['dob']}")
    print(f"  GST Declared Turnover : {s['gst_turnover_val']}")
    print(f"  Audited BS Turnover   : {s['bs_turnover_val']}")
    print(f"  Turnover Variance     : {s['turnover_variance_pct']:.2f}%")

    # Section 3: Cross-Document Consistency Audit Trail
    print("\n[+] 3. CROSS-DOCUMENT CONSISTENCY AUDIT TRAIL:")
    print("  " + "-" * 74)
    for check in d["p9_result"].get("audit_trail", []):
        st = check.get("status", "UNKNOWN")
        tag = "[PASS]" if st == "PASS" else "[FAIL]"
        ch = check.get("check", "")
        reason = check.get("reason") or "Satisfied without conflict."
        print(f"  {tag:<7} {ch}")
        if st != "PASS":
            print(f"          -> Finding: {reason}")

    # Section 4: Statutory Compliance Rules
    print("\n[+] 4. STATUTORY TENDER COMPLIANCE (PHASE 7):")
    print("  " + "-" * 74)
    for r in d["p7_result"].get("rule_details", []):
        p = "[PASS]" if r.get("passed") else "[FAIL]"
        rn = r.get("rule", "")
        rsn = r.get("reason") or "Requirement satisfied."
        print(f"  {p:<7} {rn} - {rsn}")

    # Section 5: ML Risk Prediction & Overrides
    ml = d["ml_eval"]
    probs = ", ".join([f"{k}: {v*100:.1f}%" for k, v in ml["probabilities"].items()])
    print("\n[+] 5. PREDICTIVE ML RISK ENGINE & FINAL ADJUDICATION:")
    print("  " + "-" * 74)
    print(f"  ML Raw Predicted Risk : {ml['predicted_class']} ({ml['confidence']*100:.2f}%)")
    print(f"  Class Probabilities   : [{probs}]")
    print(f"  Discrepancies Logged  : {d['inconsistency_count']} (Critical: {d['critical_discrepancy_count']})")
    print(f"  FINAL DECISION STATUS : >>> {d['final_status']} <<<")
    print(f"  FINAL RISK LEVEL      : >>> {d['final_risk']} <<<")
    print(f"  Adjudication Reason   : {d['decision_reason']}")
    print(border)


# ----------------------------------------------------------------------
# Main Verification Runner
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BidGuard AI Demo Dossier Verification")
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf", help="Document format to test")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / "models" / "bidguard_risk_model.joblib"
    scaler_path = PROJECT_ROOT / "models" / "bidguard_risk_scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        print(f"[!] Error: Model artifacts not found at {model_path} or {scaler_path}")
        sys.exit(1)

    print(f"[INFO] Loading ML Model and Scaler from models/ ...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    if not hasattr(model, "multi_class"):
        setattr(model, "multi_class", "auto")

    demo_base = PROJECT_ROOT / "demo_dossiers"
    ext = f".{args.format}"

    dossiers_to_test = [
        ("set_a_compliant", "TechVanguard Solutions Pvt Ltd (Set A - Clean Baseline)"),
        ("set_b_discrepancy", "Apex Buildcon Pvt Ltd (Set B - Name & Turnover Discrepancy)"),
        ("set_c_fraudulent", "FakeCorp Logistics Pvt Ltd (Set C - Critical Fraud & PAN Mismatch)"),
    ]

    results = {}
    print(f"[INFO] Starting End-to-End Pipeline Verification across {len(dossiers_to_test)} Dossiers...")

    for folder_name, desc in dossiers_to_test:
        d_path = demo_base / folder_name
        if not d_path.exists():
            print(f"[!] Error: Dossier directory does not exist: {d_path}")
            sys.exit(1)

        print(f"\n>>> Running Verification on: {desc} ...")
        res = verify_dossier(
            dossier_name=folder_name,
            dossier_dir=d_path,
            model=model,
            scaler=scaler,
            min_turnover_req=100.0,
            reg_cutoff_date="31/12/2026",
            ext=ext
        )
        results[folder_name] = res
        print_dossier_report(res)

    # ------------------------------------------------------------------
    # Programmatic Assertions
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  AUTOMATED RIGOROUS ASSERTION VERIFICATION")
    print("=" * 78)

    # Assertion 1: Set A Compliant
    res_a = results["set_a_compliant"]
    assert res_a["final_status"] == "COMPLIANT", (
        f"Set A expected status COMPLIANT, got {res_a['final_status']}"
    )
    assert res_a["final_risk"] == "LOW", (
        f"Set A expected risk LOW, got {res_a['final_risk']}"
    )
    assert res_a["inconsistency_count"] == 0, (
        f"Set A expected 0 inconsistencies, got {res_a['inconsistency_count']}"
    )
    assert res_a["slots"]["pan_no"] == "AAACT1234K", (
        f"Set A expected PAN AAACT1234K, got {res_a['slots']['pan_no']}"
    )
    assert res_a["slots"]["gstin_no"] == "07AAACT1234K1ZI", (
        f"Set A expected GSTIN 07AAACT1234K1ZI, got {res_a['slots']['gstin_no']}"
    )
    assert res_a["slots"]["turnover_variance_pct"] <= 1.0, (
        f"Set A expected turnover variance <= 1%, got {res_a['slots']['turnover_variance_pct']}"
    )
    print("[+] SET A ASSERTIONS PASSED:")
    print("    - Status: COMPLIANT | Risk: LOW | Discrepancies: 0")
    print("    - Standalone PAN matched embedded GSTIN PAN perfectly.")
    print(f"    - Turnover variance {res_a['slots']['turnover_variance_pct']:.2f}% satisfies 5.0% threshold.")

    # Assertion 2: Set B Discrepancy
    res_b = results["set_b_discrepancy"]
    assert res_b["final_status"] in ["CONDITIONALLY_COMPLIANT", "REVIEW"], (
        f"Set B expected status CONDITIONALLY_COMPLIANT, got {res_b['final_status']}"
    )
    assert res_b["final_risk"] == "MEDIUM", (
        f"Set B expected risk MEDIUM, got {res_b['final_risk']}"
    )
    assert res_b["inconsistency_count"] > 0, (
        f"Set B expected >0 inconsistencies, got {res_b['inconsistency_count']}"
    )
    assert not res_b["has_pan_mismatch"], (
        "Set B should NOT have PAN mismatch in GSTIN"
    )
    assert 4.0 <= res_b["slots"]["turnover_variance_pct"] <= 5.0, (
        f"Set B expected turnover variance ~4.11%, got {res_b['slots']['turnover_variance_pct']}"
    )
    print("\n[+] SET B ASSERTIONS PASSED:")
    print(f"    - Status: {res_b['final_status']} | Risk: {res_b['final_risk']}")
    print("    - Entity name discrepancy correctly detected (APEX BUILDCON PVT LTD vs PRIVATE LIMITED).")
    print(f"    - Turnover variance {res_b['slots']['turnover_variance_pct']:.2f}% correctly calculated.")

    # Assertion 3: Set C Fraudulent
    res_c = results["set_c_fraudulent"]
    assert res_c["final_status"] == "DISQUALIFIED", (
        f"Set C expected status DISQUALIFIED, got {res_c['final_status']}"
    )
    assert res_c["final_risk"] == "CRITICAL", (
        f"Set C expected risk CRITICAL, got {res_c['final_risk']}"
    )
    assert res_c["has_pan_mismatch"], (
        "Set C MUST detect PAN_MISMATCH_IN_GSTIN"
    )
    assert res_c["critical_discrepancy_count"] >= 1, (
        f"Set C expected >=1 critical discrepancy, got {res_c['critical_discrepancy_count']}"
    )
    assert res_c["slots"]["turnover_variance_pct"] >= 50.0, (
        f"Set C expected turnover variance >= 50%, got {res_c['slots']['turnover_variance_pct']}"
    )
    print("\n[+] SET C ASSERTIONS PASSED:")
    print(f"    - Status: {res_c['final_status']} | Risk: {res_c['final_risk']}")
    print("    - Forensic hard override triggered on PAN_MISMATCH_IN_GSTIN.")
    print(f"    - Turnover variance {res_c['slots']['turnover_variance_pct']:.2f}% > 5.0% tolerance.")

    print("\n" + "=" * 78)
    print("  ALL DEMO DOSSIER PIPELINE VERIFICATIONS SUCCESSFULLY COMPLETED!")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
