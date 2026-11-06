import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import streamlit as st
import pandas as pd
import json
import os
import re
import uuid
import numpy as np
import joblib
import torch
torch.set_num_threads(1)
from PIL import Image
from datetime import datetime
from pypdf import PdfReader
import easyocr

# =====================================================================
# BACKEND ENGINES INTEGRATION (MEMBER 2: PHASE 7 & 9)
# =====================================================================
from src.compliance_engine import ComplianceVerificationEngine
from src.cross_doc_consistency import CrossDocumentConsistencyEngine
from src.ocr_pipeline import is_valid_gstin
from src.document_classifier import (
    classify_document_text,
    DOC_TYPE_PAN,
    DOC_TYPE_GST,
    DOC_TYPE_BALANCE_SHEET,
    DOC_TYPE_ITR,
    DOC_TYPE_UDYAM,
    DOC_TYPE_UNKNOWN
)
from src.pan_extractor import is_valid_pan, get_pan_entity_type, compare_pan_with_gstin, extract_pan_details, mask_pan, mask_dob, mask_aadhaar
from src.upload_workflow import DocumentStagingQueue, format_file_size
from src.report_generator import generate_gem_verification_pdf

# Phase 13 Audit Logger Helper
AUDIT_LOG_FILE = "audit_ledger.json"
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

def check_fastapi_health() -> bool:
    """Pings FastAPI /health endpoint to verify REST microservice status."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{FASTAPI_BASE_URL}/health", headers={"User-Agent": "BidGuard-Dashboard"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "ok"
    except Exception:
        pass
    return False

def call_fastapi_predict_risk(feature_dict: dict) -> dict | None:
    """Delegates ML risk scoring to FastAPI POST /predict-risk."""
    try:
        import urllib.request
        req_data = json.dumps({"features": feature_dict}).encode("utf-8")
        req = urllib.request.Request(
            f"{FASTAPI_BASE_URL}/predict-risk",
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "BidGuard-Dashboard"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("success"):
                    return payload.get("prediction")
    except Exception as e:
        print(f"[WARN] FastAPI predict-risk request failed: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

# =====================================================================
# PHASE 3: ML RISK MODEL & SCALER LOADER (MEMBER 1 INTEGRATION)
# =====================================================================
@st.cache_resource(show_spinner="Loading Predictive ML Risk Engine...")
def load_risk_ml_models():
    """Load serialized tabular ML risk classifier and baseline feature scaler."""
    model_path = os.path.join("models", "bidguard_risk_model.joblib")
    scaler_path = os.path.join("models", "bidguard_risk_scaler.joblib")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            m = joblib.load(model_path)
            s = joblib.load(scaler_path)
            return m, s
        except Exception as e:
            print(f"[WARN] Error loading ML models: {e}")
            return None, None
    return None, None

risk_model, risk_scaler = load_risk_ml_models()

def predict_bidder_risk(p7_result, p9_result, lineage_records, use_fastapi: bool = False):
    """
    Constructs the exact 18-feature vector required by Member 1's trained
    LogisticRegression model and returns (predicted_class, confidence_score, badge_mode).
    """
    p7_fails = p7_result.get("failure_count", 0)
    inconsistencies = p9_result.get("inconsistency_count", 0)

    # Total requirements across both Phase 7 & Phase 9
    total_reqs = max(len(p7_result.get("rule_details", [])) + len(p9_result.get("audit_trail", [])), 5)

    # Combined non-compliant count
    non_compliant = p7_fails + inconsistencies
    compliant = max(0, total_reqs - non_compliant)
    comp_pct = (compliant / total_reqs) * 100.0 if total_reqs > 0 else 0.0

    # Vision confidence extraction from Phase 6 lineage
    conf_scores = [item.get("confidence", 0.95) for item in lineage_records if isinstance(item, dict) and "confidence" in item]
    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.95
    doc_quality = 1.0 if avg_conf >= 0.85 else 0.5

    # Specific failure indicators
    fin_fail = 1 if any("Turnover" in str(r.get("rule", "")) and not r.get("passed", True) for r in p7_result.get("rule_details", [])) else 0
    reg_fail = 1 if (p7_fails > 0 or inconsistencies > 0) else 0

    # Exact 18 features required by scaler.feature_names_in_
    features_dict = {
        'total_requirements': float(total_reqs),
        'mandatory_requirements': float(total_reqs),
        'compliant_count': float(compliant),
        'partially_compliant_count': 0.0,
        'non_compliant_count': float(non_compliant),
        'insufficient_evidence_count': 0.0,
        'compliance_percentage': float(comp_pct),
        'missing_document_count': 1.0 if non_compliant > 0 and inconsistencies > 0 else 0.0,
        'expired_document_count': 0.0,
        'invalid_document_count': 1.0 if inconsistencies > 0 else 0.0,
        'inconsistency_count': float(inconsistencies),
        'mandatory_failure_count': float(non_compliant),
        'financial_requirement_failure': float(fin_fail),
        'experience_requirement_failure': 0.0,
        'registration_requirement_failure': float(reg_fail),
        'authorization_requirement_failure': 0.0,
        'evidence_confidence_average': float(avg_conf),
        'document_quality_score': float(doc_quality)
    }

    if use_fastapi:
        api_prediction = call_fastapi_predict_risk(features_dict)
        if api_prediction:
            pred_class = api_prediction.get("predicted_risk_level", "LOW")
            probs = api_prediction.get("prediction_probabilities", {})
            confidence = float(probs.get(pred_class, 0.95))
            return pred_class, confidence, "FASTAPI"

    if risk_model is not None and risk_scaler is not None:
        try:
            # Compatibility patch for cross-version unpickling (1.9.0 model on 1.7.2 environment)
            if not hasattr(risk_model, 'multi_class'):
                setattr(risk_model, 'multi_class', 'auto')

            df_feat = pd.DataFrame([features_dict])
            X_scaled = risk_scaler.transform(df_feat)

            # Pass DataFrame to match feature names
            X_scaled_df = pd.DataFrame(X_scaled, columns=risk_scaler.feature_names_in_)

            pred_class = risk_model.predict(X_scaled_df)[0]
            probs = risk_model.predict_proba(X_scaled_df)[0]
            class_idx = list(risk_model.classes_).index(pred_class)
            confidence = float(probs[class_idx])

            # Terminal verification log
            print("\n" + "="*50)
            print(">>> LIVE ML MODEL INFERENCE EXECUTED <<<")
            print(f"Model Learned Classes: {risk_model.classes_}")
            print(f"Calculated Probabilities: {dict(zip(risk_model.classes_, probs))}")
            print(f"Final Decision: {pred_class} with Confidence: {confidence:.4f}")
            print("="*50 + "\n")

            return pred_class, confidence, True
        except Exception as e:
            print(f"[WARN] Inference exception: {e}")

    # Fallback heuristic if model files are unreadable
    fallback_class = "LOW" if (non_compliant + inconsistencies) == 0 else "HIGH"
    return fallback_class, 0.95, False

def safe_log_decision(tender_id, bidder_id, action, remarks, lineage_data):
    """Phase 13: Immutable Local Adjudication Logger"""
    event_id = f"EVT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    new_record = {
        "event_id": event_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "adjudication": {
            "decision": action,
            "remarks": remarks
        },
        "lineage_evidence": lineage_data
    }

    logs = []
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(new_record)
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

    return event_id

def safe_get_all_logs():
    """Phase 13: Fetch historical audit ledger"""
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def extract_slots_and_lineage_from_ocr(ocr_results, source_doc_name):
    """
    Universal Slot Extractor: Robust against noise tokens ('TTAA'), Hindi labels,
    and document variations across GST, PAN, and Balance Sheets.
    """
    data = {
        "entity_name": "NOT DETECTED",
        "gstin": "NOT DETECTED",
        "pan": "NOT DETECTED",
        "issue_date": "NOT DETECTED",
        "turnover": 0.0,
        "turnover_str": "Rs. 0.0 Lakhs",
        "bs_turnover": 0.0,
        "bs_turnover_str": "Rs. 0.0 Lakhs"
    }
    lineage = []

    extracted_items = []
    for bbox, text, conf in ocr_results:
        clean_t = text.strip()
        if not clean_t:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        box_coords = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        extracted_items.append({"text": clean_t, "bbox": box_coords, "conf": float(conf)})

    full_text_lines = [item["text"] for item in extracted_items]

    # -------------------------------------------------------------
    # 1. GSTIN EXTRACTION
    # -------------------------------------------------------------
    for item in extracted_items:
        clean_t = item["text"].upper()
        gstin_match = re.search(r'\b([0-9O]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z|2][0-9A-Z]{1})\b', clean_t)
        if gstin_match and data["gstin"] == "NOT DETECTED":
            cand = list(gstin_match.group(1))
            if cand[0] in ['O', 'D']: cand[0] = '0'
            if cand[1] in ['O', 'D']: cand[1] = '0'
            if cand[13] in ['2', '7', 'C', 'c', 'z', '€', 'Γé¼']:
                cand[13] = 'Z'
            cand_str = "".join(cand)
            if is_valid_gstin(cand_str, verify_checksum=True):
                data["gstin"] = cand_str
                lineage.append({
                    "document": source_doc_name,
                    "field": "gstin",
                    "value": data["gstin"],
                    "bbox": item["bbox"],
                    "confidence": round(item["conf"], 3)
                })
                break

    # Backup GSTIN search by keyword
    if data["gstin"] == "NOT DETECTED":
        for idx, item in enumerate(extracted_items):
            if "GSTIN" in item["text"].upper():
                val = item["text"].split(":")[-1].strip() if ":" in item["text"] else ""
                if not val and idx + 1 < len(extracted_items):
                    val = extracted_items[idx + 1]["text"].strip()
                cleaned_val = re.sub(r'[^A-Za-z0-9]', '', val).upper()
                if len(cleaned_val) >= 15:
                    cand_backup = cleaned_val[:15]
                    cand_list = list(cand_backup)
                    if cand_list[0] in ['O', 'D']: cand_list[0] = '0'
                    if cand_list[1] in ['O', 'D']: cand_list[1] = '0'
                    if cand_list[13] in ['2', '7', 'C', 'c', 'z', '€', 'Γé¼']:
                        cand_list[13] = 'Z'
                    cand_backup_str = "".join(cand_list)
                    if is_valid_gstin(cand_backup_str, verify_checksum=True):
                        data["gstin"] = cand_backup_str
                        lineage.append({
                            "document": source_doc_name,
                            "field": "gstin",
                            "value": data["gstin"],
                            "bbox": item["bbox"],
                            "confidence": round(item["conf"], 3)
                        })
                        break

    # -------------------------------------------------------------
    # 2. STANDALONE PAN EXTRACTION (FORENSIC PRIORITY)
    # -------------------------------------------------------------
    for item in extracted_items:
        clean_t = item["text"].upper()
        pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b', clean_t)
        if pan_match:
            cand_pan = pan_match.group(1)
            raw_no_space = clean_t.replace(" ", "").replace(":", "")
            if len(raw_no_space) == 10 or "PAN" in clean_t or "PERMANENT" in clean_t or "ACCOUNT" in clean_t:
                data["pan"] = cand_pan
                lineage.append({
                    "document": source_doc_name,
                    "field": "pan",
                    "value": cand_pan,
                    "bbox": item["bbox"],
                    "confidence": round(item["conf"], 3)
                })
                break


    # -------------------------------------------------------------
    # 3. ENTITY / COMPANY / LEGAL NAME EXTRACTION
    # -------------------------------------------------------------
    business_keywords = ["PVT LTD", "PRIVATE LIMITED", "LTD", "LIMITED", "ENTERPRISES", "SOLUTIONS", "INFRA", "LOGISTICS", "INDUSTRIES", "CORP", "INC", "LLP"]

    for item in extracted_items:
        t_upper = item["text"].upper()
        if any(b_word in t_upper for b_word in business_keywords):
            if not any(hdr in t_upper for hdr in ["GOVERNMENT", "DEPARTMENT", "INCOME TAX", "CERTIFICATE", "REPORT", "AUDITOR"]):
                c_name = item["text"].replace(":", "").replace(";", "").replace("\u20ac", "&").strip()
                c_name = re.sub(r'[\u0900-\u097F]', '', c_name).strip()
                if "ENTITY" in c_name.upper():
                    c_name = c_name.split(":")[-1].strip()
                if len(c_name) > 4:
                    data["entity_name"] = c_name
                    lineage.append({
                        "document": source_doc_name,
                        "field": "entity_name",
                        "value": c_name,
                        "bbox": item["bbox"],
                        "confidence": round(item["conf"], 3)
                    })
                    break

    if data["entity_name"] == "NOT DETECTED":
        for idx, item in enumerate(extracted_items):
            t_upper = item["text"].upper()
            if any(lbl in t_upper for lbl in ["ENTITY NAME", "LEGAL NAME", "NAME OF HOLDER", "NAME OF BIDDER"]):
                cand_name = ""
                if ":" in item["text"]:
                    parts = item["text"].split(":", 1)[-1].strip()
                    if len(parts) > 4 and not any(k in parts for k in ["\u0915\u093e\u0928\u0942\u0928\u0940", "\u0928\u093e\u092e", "\u0935\u094d\u092f\u093e\u092a\u093e\u0930"]):
                        cand_name = parts
                if not cand_name:
                    for offset in range(1, 4):
                        if idx + offset < len(extracted_items):
                            nxt = extracted_items[idx + offset]["text"].strip()
                            if any(k in nxt for k in ["\u0915\u093e\u0928\u0942\u0928\u0940", "\u0928\u093e\u092e", "\u0935\u094d\u092f\u093e\u092a\u093e\u0930", "Trade", "GSTIN", "PAN"]):
                                continue
                            clean_cand = nxt.replace(":", "").strip()
                            if len(clean_cand) > 4 and not re.search(r'\d{2}[A-Z]{5}', clean_cand):
                                cand_name = clean_cand
                                break
                if cand_name:
                    data["entity_name"] = cand_name
                    lineage.append({
                        "document": source_doc_name,
                        "field": "entity_name",
                        "value": cand_name,
                        "bbox": item["bbox"],
                        "confidence": round(item["conf"], 3)
                    })
                    break

    # -------------------------------------------------------------
    # 4. DATE DETECTION
    # -------------------------------------------------------------
    for item in extracted_items:
        date_match = re.search(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b', item["text"])
        if date_match and data["issue_date"] == "NOT DETECTED":
            data["issue_date"] = date_match.group(1)
            lineage.append({
                "document": source_doc_name,
                "field": "issue_date",
                "value": data["issue_date"],
                "bbox": item["bbox"],
                "confidence": round(item["conf"], 3)
            })

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 5. SOURCE-SPECIFIC TURNOVER EXTRACTION (VALUES IN LAKHS)
    # -------------------------------------------------------------
    is_gst = source_doc_name == "GST_CERTIFICATE"
    is_balance_sheet = source_doc_name == "BALANCE_SHEET"

    if is_gst or is_balance_sheet:
        target_key = "turnover" if is_gst else "bs_turnover"
        target_str_key = "turnover_str" if is_gst else "bs_turnover_str"

        for item in extracted_items:
            clean_t = item["text"]
            t_lower = clean_t.lower()

            if any(word in t_lower for word in [
                "gstin", "pan", "udin", "date", "issue", "liability", "status"
            ]):
                continue

            match = re.search(
                r'[:\-]?\s*(?:Rs\.?\s*)?(\d+(?:\.\d+)?)\s*(Lakhs|Lakh|Cr|Crore)',
                clean_t,
                re.IGNORECASE
            )

            if match:
                try:
                    num_val = float(match.group(1))
                    unit = match.group(2).lower()

                    if unit in ("cr", "crore"):
                        num_val *= 100

                    if num_val > 0:
                        data[target_key] = num_val
                        data[target_str_key] = f"Rs. {num_val:g} Lakhs"
                        lineage.append({
                            "document": source_doc_name,
                            "field": target_key,
                            "value": f"Rs. {num_val:g} Lakhs",
                            "bbox": item["bbox"],
                            "confidence": round(item["conf"], 3)
                        })
                        break
                except (ValueError, TypeError):
                    pass

        if data[target_key] == 0.0:
            for idx, item in enumerate(extracted_items):
                if "TURNOVER" in item["text"].upper():
                    for offset in range(1, 3):
                        if idx + offset >= len(extracted_items):
                            continue

                        next_item = extracted_items[idx + offset]
                        num_match = re.search(
                            r'[:\-]?\s*(?:Rs\.?\s*)?(\d+(?:\.\d+)?)\s*(Lakhs|Lakh|Cr|Crore)?',
                            next_item["text"],
                            re.IGNORECASE
                        )

                        if num_match:
                            try:
                                num_val = float(num_match.group(1))
                                unit = (num_match.group(2) or "Lakhs").lower()

                                if unit in ("cr", "crore"):
                                    num_val *= 100

                                if num_val > 0:
                                    data[target_key] = num_val
                                    data[target_str_key] = f"Rs. {num_val:g} Lakhs"
                                    lineage.append({
                                        "document": source_doc_name,
                                        "field": target_key,
                                        "value": f"Rs. {num_val:g} Lakhs",
                                        "bbox": next_item["bbox"],
                                        "confidence": round(next_item["conf"], 3)
                                    })
                                    break
                            except (ValueError, TypeError):
                                pass

                    if data[target_key] > 0:
                        break

    return data, lineage, "\n".join(full_text_lines)

# =====================================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="BidGuard-AI | Officer Procurement Portal",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("\U0001F6E1 BidGuard-AI: Officer Audit & Statutory Verification Portal")
st.caption("Integrated: Vision OCR (Phase 6) + Statutory Engine (Phase 7) + Predictive ML Risk (Phase 3) + Forensic Audit (Phase 9) + Ledger (Phase 13)")
st.divider()

# =====================================================================
# SIDEBAR CONTROLS (MULTI-DOCUMENT DOSSIER)
# =====================================================================
st.sidebar.header("🌐 System Architecture Mode")
engine_mode = st.sidebar.radio(
    "Processing Engine",
    ["Direct In-Process (Standalone)", "FastAPI Microservice (REST Client)"],
    help="Toggle between direct module execution and decoupled HTTP REST microservice."
)

use_fastapi_client = False
if engine_mode == "FastAPI Microservice (REST Client)":
    if check_fastapi_health():
        st.sidebar.success(f"🟢 Connected to FastAPI ({FASTAPI_BASE_URL})")
        use_fastapi_client = True
    else:
        st.sidebar.warning(f"🟡 FastAPI offline at {FASTAPI_BASE_URL}. Falling back cleanly to In-Process Engine.")
        use_fastapi_client = False

st.sidebar.divider()
st.sidebar.header("\U0001F4CB Tender Specifications (NIT)")
selected_tender = st.sidebar.selectbox(
    "Active Tender Reference",
    ["TND-2026-001 (Oil & Gas Infrastructure Pipeline)", "TND-2026-002 (Civil Works Expansion)"]
)

min_turnover_req = st.sidebar.number_input(
    "Mandatory Minimum Turnover (Rs. Lakhs)",
    min_value=10.0,
    max_value=1000.0,
    value=75.0,
    step=5.0
)

reg_cutoff_date = st.sidebar.date_input(
    "Mandatory Registration Cut-off Date",
    value=pd.to_datetime("2026-12-31")
)

st.sidebar.divider()
st.sidebar.header("\U0001F4C1 Vendor Document Dossier")

# Initialize persistent multi-document upload staging queue in session state
if "staging_queue" not in st.session_state:
    st.session_state["staging_queue"] = DocumentStagingQueue()
staging_queue: DocumentStagingQueue = st.session_state["staging_queue"]

# -------------------------------------------------------------
# QUICK DEMO LOADER (Instant Multi-Document Benchmark Dossiers)
# -------------------------------------------------------------
st.sidebar.markdown("#### ⚡ Quick Demo Loader")
st.sidebar.caption("Load complete 4-document dossiers directly into the staging queue:")

def load_demo_dossier_to_staging(set_folder: str, set_label: str):
    staging_queue.clear()
    folder_path = os.path.join("demo_dossiers", set_folder)
    if os.path.exists(folder_path):
        count = 0
        for f in sorted(os.listdir(folder_path)):
            if f.endswith(".pdf"):
                fpath = os.path.join(folder_path, f)
                with open(fpath, "rb") as fp:
                    staging_queue.add_file(f, fp.read())
                count += 1
        st.session_state["upload_mode_radio"] = "Live Multi-Document Upload (Phase 6 Integration)"
        st.sidebar.success(f"⚡ Loaded {count} documents from {set_label}!")
        st.rerun()

if st.sidebar.button("⚡ Load Compliant Bidder (Set A)", use_container_width=True, help="Loads Set A (Clean Baseline): PAN, GST, Udyam, Audited Balance Sheet. All criteria fully compliant."):
    load_demo_dossier_to_staging("set_a_compliant", "Set A (Compliant)")

if st.sidebar.button("⚡ Load Discrepancy Bidder (Set B)", use_container_width=True, help="Loads Set B: Contains entity name spelling variation and turnover variance (~4.11%)."):
    load_demo_dossier_to_staging("set_b_discrepancy", "Set B (Discrepancy)")

if st.sidebar.button("⚡ Load Fraudulent Bidder (Set C)", use_container_width=True, help="Loads Set C: Contains fraudulent PAN mismatch in GSTIN and 60% turnover variance."):
    load_demo_dossier_to_staging("set_c_fraudulent", "Set C (Fraudulent)")

st.sidebar.markdown("---")

upload_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Live Multi-Document Upload (Phase 6 Integration)", "Pre-loaded Benchmark Dossiers"],
    key="upload_mode_radio"
)

bidder_selection = None
current_lineage = []
all_extracted_texts = {}
pan_name = "NOT DETECTED"
pan_name_status = "NOT_VERIFIED"
slots = {}

if upload_mode == "Live Multi-Document Upload (Phase 6 Integration)":
    st.sidebar.markdown("**Upload Tender Documents (PDF, JPG, PNG):**")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Vendor Dossier",
        type=["pdf", "jpeg", "jpg", "png"],
        accept_multiple_files=True,
        key="staged_multi_file_uploader",
        help="Upload multiple documents. Uploading files does NOT automatically run OCR. Verification only runs when requested."
    )

    # Sync uploaded files to persistent staging queue without triggering OCR
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in staging_queue.documents:
                staging_queue.add_file(uf.name, uf.getvalue())

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Upload Staging Queue")
    st.sidebar.caption(f"**{len(staging_queue.documents)}** document(s) staged in queue.")

    if staging_queue.documents:
        for doc_name, doc_obj in list(staging_queue.documents.items()):
            col_q1, col_q2 = st.sidebar.columns([3, 1])
            status_badge = "✅ Verified" if doc_obj.status == "Verified" else "⏳ Queued"
            col_q1.markdown(f"**{doc_name}**<br><small>{doc_obj.size_formatted} | {doc_obj.detected_type} | {status_badge}</small>", unsafe_allow_html=True)
            if col_q2.button("✖", key=f"del_{doc_name}", help=f"Remove {doc_name} from queue"):
                staging_queue.remove_file(doc_name)
                st.rerun()

        if st.sidebar.button("🗑️ Clear All Files", use_container_width=True):
            staging_queue.clear()
            st.rerun()

        st.sidebar.markdown("---")
        verify_clicked = st.sidebar.button(
            "🔍 Verify All Documents",
            type="primary",
            use_container_width=True,
            help="Executes Vision OCR, pattern classification, slot extraction, and statutory compliance checks."
        )

        if verify_clicked:
            with st.spinner("\U0001F50D Running Multi-Document Vision Pipelines & Forensic Lineage..."):
                ocr_reader = load_ocr_reader()
                staging_queue.execute_consolidated_verification(
                    ocr_reader=ocr_reader,
                    min_turnover_req=float(min_turnover_req),
                    reg_cutoff_date=str(reg_cutoff_date.strftime("%d/%m/%Y"))
                )
            st.rerun()

        if not staging_queue.is_verified:
            st.info("👈 Staged documents are ready. Click **'🔍 Verify All Documents'** in the sidebar to run analysis.")
            st.markdown("### 📋 Upload Staging Queue")
            df_queue = pd.DataFrame(staging_queue.get_queue_summary())
            st.dataframe(df_queue, use_container_width=True, hide_index=True)
            st.stop()

        # Retrieve cached results without repeating OCR
        res = staging_queue.verification_results
        current_lineage = res["all_lineage"]
        all_extracted_texts = res["all_extracted_texts"]
        audit_data = res["audit_data"]
        compliance_status = res["compliance_status"]
        inconsistencies_count = res["discrepancies_count"]
        slots = res["slots"]
        company_name = slots["company_name"]
        gstin_no = slots["gstin_no"]
        pan_no = slots["pan_no"]
        pan_name = slots["pan_name"]
        pan_name_status = slots["pan_name_status"]
        pan_entity_type = slots.get("pan_entity_type", "UNKNOWN")
        father_name = slots.get("father_name")
        father_name_status = slots.get("father_name_status", "NOT_VERIFIED")
        dob = slots.get("dob")
        dob_status = slots.get("dob_status", "NOT_VERIFIED")
        masked_pan = slots.get("masked_pan") or mask_pan(pan_no)
        masked_dob = slots.get("masked_dob") or (mask_dob(dob) if dob else "NOT DETECTED")
        gst_turnover_val = slots["gst_turnover_val"]
        bs_turnover_val = slots["bs_turnover_val"]
        doc_date = slots["doc_date"]
        bidder_id = f"BID-{uuid.uuid4().hex[:5].upper()}"

        # Execute Predictive ML Risk Classifier
        pred_risk, risk_prob, is_ml = predict_bidder_risk(
            res["p7_result"], res["p9_result"], current_lineage, use_fastapi=use_fastapi_client
        )
        if is_ml == "FASTAPI":
            badge_type = "🌐 FastAPI REST Client"
            risk_badge = f"{risk_prob*100:.1f}% (FastAPI Microservice)"
        elif is_ml is True:
            badge_type = "\U0001F916 In-Process ML Model"
            risk_badge = f"{risk_prob*100:.1f}% (ML Model)"
        else:
            badge_type = "Deterministic"
            risk_badge = f"{risk_prob*100:.1f}% (Deterministic)"
        risk_rating = pred_risk
    else:
        st.info("👈 Please upload vendor documents (GST Certificate, PAN Card, Balance Sheet, etc.) in the sidebar to begin.")
        st.stop()
elif upload_mode == "Pre-loaded Benchmark Dossiers":
    is_clean = "Clean" in (bidder_selection or "")
    bidder_id = "BID-101" if is_clean else "BID-102"
    company_name = "Apex Infotech Pvt Ltd" if is_clean else "Apex Enterprises"
    gstin_no = "07AAAAA0000A1Z5"
    pan_no = "AAAAA0000A" if is_clean else "BBBBB9999B"
    pan_entity_type = "Company" if is_clean else "Individual / Person"
    father_name = None if is_clean else "Suresh Kumar"
    father_name_status = "NOT_APPLICABLE" if is_clean else "VERIFIED"
    dob = "15/05/2018" if is_clean else "12/08/1985"
    dob_status = "VERIFIED"
    masked_pan = mask_pan(pan_no)
    masked_dob = mask_dob(dob)
    gst_turnover_val = "Rs. 85.0 Lakhs" if is_clean else "Rs. 100.0 Lakhs"
    bs_turnover_val = "Rs. 85.0 Lakhs" if is_clean else "Rs. 40.0 Lakhs"
    doc_date = "15/05/2026"

    inconsistencies_count = 0 if is_clean else 2
    compliance_status = "PASSED" if is_clean else "FLAGGED"

    audit_data = [
        {"Stage": "Phase 7 Statutory", "Rule Checked": "GSTIN Identification", "Status": "PASS", "Finding Details": "Entity GSTIN registered in database."},
        {"Stage": "Phase 7 Statutory", "Rule Checked": "Turnover Threshold", "Status": "PASS", "Finding Details": "Declared turnover meets tender limit."},
        {"Stage": "Phase 7 Statutory", "Rule Checked": "Document Validity", "Status": "PASS", "Finding Details": "Certificate issue date is valid."},
        {"Stage": "Phase 9 Forensic", "Rule Checked": "PAN Embedded inside GSTIN", "Status": "PASS" if is_clean else "FAIL", "Finding Details": f"GSTIN[2:12] vs PAN '{pan_no}'" + (" matched." if is_clean else " MISMATCH DETECTED!")},
        {"Stage": "Phase 9 Forensic", "Rule Checked": "Turnover Variance (GST vs Balance Sheet)", "Status": "PASS" if is_clean else "FAIL", "Finding Details": "Discrepancy 0.0% within tolerance." if is_clean else "Variance 60.0% violates 5.0% tolerance limit."}
    ]

    current_lineage = [
        {"document": "GST_CERTIFICATE", "field": "gstin", "value": gstin_no, "bbox": [110, 160, 390, 200], "confidence": 0.972},
        {"document": "PAN_CARD", "field": "pan", "value": pan_no, "bbox": [130, 150, 360, 190], "confidence": 0.958}
    ]

    benchmark_p7 = {
        "failure_count": 0,
        "rule_details": [
            {"rule": "GSTIN Identification", "passed": True},
            {"rule": "Turnover Threshold", "passed": True},
            {"rule": "Document Validity", "passed": True}
        ]
    }
    benchmark_p9 = {
        "inconsistency_count": 0 if is_clean else 2,
        "audit_trail": [
            {"check": "PAN Embedded inside GSTIN", "status": "PASS" if is_clean else "FAIL"},
            {"check": "Turnover Variance (GST vs Balance Sheet)", "status": "PASS" if is_clean else "FAIL"}
        ]
    }

    # Execute actual ML Model inference for benchmark dossiers
    pred_risk, risk_prob, is_ml = predict_bidder_risk(
        benchmark_p7, benchmark_p9, current_lineage, use_fastapi=use_fastapi_client
    )
    if is_ml == "FASTAPI":
        badge_type = "🌐 FastAPI REST Client"
        risk_badge = f"{risk_prob*100:.1f}% (FastAPI Microservice)"
    elif is_ml is True:
        badge_type = "\U0001F916 In-Process ML Model"
        risk_badge = f"{risk_prob*100:.1f}% (ML Model)"
    else:
        badge_type = "Deterministic"
        risk_badge = f"{risk_prob*100:.1f}% (Deterministic)"
    risk_rating = f"{pred_risk} RISK ({risk_prob*100:.1f}%)"
    risk_badge = f"{risk_prob*100:.1f}% ({'ML Model' if is_ml else 'Deterministic'})"

else:
    st.info("\U0001F448 Please upload the primary **GST Certificate** in the sidebar to begin automated lineage verification.")
    st.stop()

# =====================================================================
# UI LAYOUT
# =====================================================================
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.subheader("\U0001F4C4 Document Inspection & Spatial Lineage (Phase 6)")

    if upload_mode == "Live Multi-Document Upload (Phase 6 Integration)":
        doc_names = list(staging_queue.documents.keys())
        if doc_names:
            tab_labels = [f"{staging_queue.documents[dn].detected_type}: {dn}" for dn in doc_names]
            tabs = st.tabs(tab_labels)
            for idx, dn in enumerate(doc_names):
                doc_obj = staging_queue.documents[dn]
                with tabs[idx]:
                    st.caption(f"**Document Type:** {doc_obj.detected_type} (Confidence: {doc_obj.classification_confidence:.1%}) | **Size:** {doc_obj.size_formatted} | **Status:** {doc_obj.status}")
                    f_path = os.path.join("temp_uploads", dn)
                    if dn.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(f_path):
                        st.image(f_path, caption=f"{doc_obj.detected_type}: {dn}", use_container_width=True)
                    else:
                        st.success(f"\U0001F4C1 Document File: {dn} ({doc_obj.size_formatted})")
                    if doc_obj.extracted_fields:
                        st.json(doc_obj.extracted_fields)


        with st.expander("\U0001F50E View Raw Extracted OCR Streams", expanded=False):
            for doc_k, doc_v in all_extracted_texts.items():
                st.text_area(f"{doc_k} Stream", mask_aadhaar(doc_v), height=120)

    pan_extra = f"[{pan_entity_type}] &bull; Masked: <code>{masked_pan}</code>" if pan_no != "NOT FOUND" else ""
    father_display = father_name if father_name else ("Not Applicable (Corporate Entity)" if pan_entity_type in ["Company", "Firm / Limited Liability Partnership (LLP)", "Trust"] else "NOT DETECTED")
    dob_display = f"{dob} (Masked: {masked_dob})" if dob and dob != "NOT DETECTED" else "NOT DETECTED"

    st.markdown(
        f"""
        <div style="border: 1px solid #333; padding: 16px; border-radius: 8px; background-color: #111418; color: #f0f0f0;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #444; padding-bottom: 8px; margin-bottom: 12px;">
                <strong>EXTRACTION SLOT SUMMARY</strong>
                <span style="color: #4caf50;">EasyOCR: Active & Running</span>
            </div>
            <p><strong>Legal Entity Name:</strong> <mark style="background-color: #2e7d32; color: white; padding: 2px 6px; border-radius: 4px;">{company_name}</mark></p>
            <p><strong>GSTIN Number:</strong> <mark style="background-color: #1565c0; color: white; padding: 2px 6px; border-radius: 4px;">{gstin_no}</mark></p>
            <p><strong>Permanent Account Number (PAN):</strong> <mark style="background-color: #e65100; color: white; padding: 2px 6px; border-radius: 4px;">{pan_no}</mark> <span style="font-size: 0.85em; color: #bbb;">{pan_extra}</span></p>
            <p><strong>PAN Cardholder / Legal Name:</strong> <mark style="background-color: #4a148c; color: white; padding: 2px 6px; border-radius: 4px;">{pan_name}</mark> <span style="font-size: 0.85em; color: #bbb;">[{pan_name_status}]</span></p>
            <p><strong>Father's Name (Individual PAN):</strong> <strong>{father_display}</strong></p>
            <p><strong>Date of Birth / Incorporation:</strong> <strong>{dob_display}</strong></p>
            <p><strong>Declared GST Turnover:</strong> <strong>{gst_turnover_val}</strong></p>
            <p><strong>Audited Balance Sheet Turnover:</strong> <strong>{bs_turnover_val}</strong></p>
            <p><strong>Issue / Certification Date:</strong> {doc_date}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if slots.get("multi_pan_count", 0) > 1:
        st.markdown(f"##### 👥 Associated Partner / Signatory PAN Records ({slots.get('multi_pan_count')})")
        secondary_pans = []
        for p in slots.get("all_pan_extractions", []):
            p_fields = p.get("fields", {})
            secondary_pans.append({
                "Document": p.get("filename"),
                "PAN": mask_pan(p_fields.get("pan")),
                "Entity Type": p_fields.get("entity_type", "UNKNOWN"),
                "Holder Name": p_fields.get("cardholder_name", "NOT DETECTED"),
                "Status": p_fields.get("name_status", "NOT_VERIFIED")
            })
        st.dataframe(pd.DataFrame(secondary_pans), use_container_width=True, hide_index=True)

with col_right:
    st.subheader("\U0001F50D Automated Forensic & Compliance Scorecard")

    k1, k2, k3 = st.columns(3)
    k1.metric("Compliance Status", compliance_status)
    k2.metric("Discrepancies", f"{inconsistencies_count} Found")
    k3.metric("Assigned Risk", f"{risk_rating} RISK", delta=risk_badge)

    st.markdown("#### Lineage Verification Audit Trail")
    df_audit = pd.DataFrame(audit_data)
    st.dataframe(df_audit, use_container_width=True, hide_index=True)

    st.divider()

    # Officer Final Adjudication Panel (Phase 13)
    st.subheader("\u2696\uFE0F Officer Adjudication Panel (Phase 13)")
    officer_remarks = st.text_input("Officer Audit Notes / Remarks", value="Verified against statutory procurement criteria.")

    btn1, btn2, btn3 = st.columns(3)
    tender_id_ref = selected_tender.split()[0]

    if btn1.button("\u2705 Approve Tender Bid", use_container_width=True):
        if inconsistencies_count > 0:
            st.error(f"\u274C REJECTED BY SYSTEM: Cannot approve bid for {company_name}. Forensic cross-document conflicts detected!")
        else:
            evt_id = safe_log_decision(
                tender_id=tender_id_ref,
                bidder_id=bidder_id,
                action="APPROVED",
                remarks=officer_remarks,
                lineage_data=current_lineage
            )
            st.success(f"\u2714 APPROVED: Bid from {company_name} meets all checks! Event ID: {evt_id}")
            st.rerun()

    if btn2.button("\u26A0\uFE0F Request Clarification", use_container_width=True):
        evt_id = safe_log_decision(
            tender_id=tender_id_ref,
            bidder_id=bidder_id,
            action="CLARIFICATION_REQUESTED",
            remarks=officer_remarks,
            lineage_data=current_lineage
        )
        st.warning(f"Clarification memo logged for {company_name}. Event ID: {evt_id}")
        st.rerun()

    if btn3.button("\u274C Reject Submission", use_container_width=True):
        evt_id = safe_log_decision(
            tender_id=tender_id_ref,
            bidder_id=bidder_id,
            action="REJECTED",
            remarks=officer_remarks,
            lineage_data=current_lineage
        )
        st.error(f"Submission rejected under statutory rules. Event ID: {evt_id}")
        st.rerun()

    st.markdown("---")
    # Download GeM Verification Report (PDF)
    try:
        report_pdf = generate_gem_verification_pdf(
            tender_id=selected_tender,
            bidder_id=bidder_id,
            company_name=company_name,
            gstin_no=gstin_no,
            pan_no=masked_pan or pan_no,
            pan_entity_type=pan_entity_type,
            turnover_gst=gst_turnover_val,
            turnover_bs=bs_turnover_val,
            compliance_status=compliance_status,
            risk_rating=risk_rating,
            risk_badge=risk_badge,
            inconsistencies_count=inconsistencies_count,
            audit_data=audit_data,
            officer_remarks=officer_remarks
        )
        st.download_button(
            label="\U0001F4C4 Download GeM Verification Report (PDF)",
            data=report_pdf,
            file_name=f"GeM_Verification_Report_{bidder_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Exports official single-page GeM Statutory Compliance & Forensic Verification Summary PDF."
        )
    except Exception as e:
        st.warning(f"Could not generate GeM PDF Report: {e}")

# =====================================================================
# HISTORICAL AUDIT LEDGER (PHASE 13)
# =====================================================================
st.divider()
with st.expander("\U0001F4DC Historical Adjudication Ledger & Audit Lineage (Phase 13)", expanded=False):
    logs = safe_get_all_logs()
    if logs:
        display_records = []
        for l in reversed(logs):
            display_records.append({
                "Event ID": l.get("event_id"),
                "Timestamp": l.get("timestamp"),
                "Tender ID": l.get("tender_id"),
                "Bidder ID": l.get("bidder_id"),
                "Decision": l.get("adjudication", {}).get("decision"),
                "Remarks": l.get("adjudication", {}).get("remarks"),
                "Lineage Evidence Fields": len(l.get("lineage_evidence", []))
            })
        st.dataframe(pd.DataFrame(display_records), use_container_width=True, hide_index=True)

        st.download_button(
            label="\U0001F4E5 Download Full Audit Ledger (JSON)",
            data=json.dumps(logs, indent=4),
            file_name="bidguard_adjudication_ledger.json",
            mime="application/json"
        )
    else:
        st.info("No adjudication events logged yet.")
