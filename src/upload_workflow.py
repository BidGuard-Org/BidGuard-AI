"""Multi-Document Staging & Verification Workflow Engine for BidGuard AI.

Provides queue management, pre-verification staging, duplicate rerun prevention,
and consolidated multi-document compliance analysis.
"""

import os
import re
import uuid
from typing import Dict, Any, List, Optional
from pypdf import PdfReader
import io
import cv2

from src.document_classifier import (
    classify_document_text,
    DOC_TYPE_PAN,
    DOC_TYPE_GST,
    DOC_TYPE_BALANCE_SHEET,
    DOC_TYPE_ITR,
    DOC_TYPE_UDYAM,
    DOC_TYPE_UNKNOWN
)
from src.pan_extractor import (
    extract_pan_details,
    compare_pan_with_gstin,
    is_valid_pan,
    mask_aadhaar,
    mask_pan,
    mask_dob
)
from src.pan_preprocessor import DocumentPreprocessor
from src.ocr_pipeline import is_valid_gstin, validate_gstin_checksum
from src.compliance_engine import ComplianceVerificationEngine
from src.cross_doc_consistency import CrossDocumentConsistencyEngine


def format_file_size(size_bytes: int) -> str:
    """Formats bytes into human-readable KB or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


class StagedDocument:
    """Represents an uploaded document in the staging queue before or after verification."""
    def __init__(self, filename: str, file_bytes: bytes):
        self.doc_id = str(uuid.uuid4())[:8]
        self.filename = filename
        self.file_bytes = file_bytes
        self.size_bytes = len(file_bytes)
        self.size_formatted = format_file_size(self.size_bytes)
        self.detected_type: str = DOC_TYPE_UNKNOWN
        self.classification_confidence: float = 0.0
        self.status: str = "Queued"  # "Queued", "Processing", "Verified", "Error"
        self.extracted_fields: Dict[str, Any] = {}
        self.raw_text: str = ""
        self.lineage: List[Dict[str, Any]] = []
        self.error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "size": self.size_formatted,
            "detected_type": self.detected_type,
            "status": self.status,
            "extracted_fields": self.extracted_fields
        }


class DocumentStagingQueue:
    """
    Manages multi-document staging, removal, duplicate rerun prevention,
    and consolidated compliance execution.
    """
    def __init__(self):
        self.documents: Dict[str, StagedDocument] = {}
        self.is_verified: bool = False
        self.verification_results: Optional[Dict[str, Any]] = None

    def add_file(self, filename: str, file_bytes: bytes) -> StagedDocument:
        """Adds a document to the upload queue without running OCR or compliance."""
        doc = StagedDocument(filename, file_bytes)

        # Fast preview classification for digital PDFs only (takes < 0.005s, no vision OCR)
        if filename.lower().endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                preview_text = "\n".join([p.extract_text() or "" for p in reader.pages[:3]])
                if len(preview_text.strip()) > 30:
                    cls_res = classify_document_text(preview_text)
                    doc.detected_type = cls_res["document_type"]
                    doc.classification_confidence = cls_res["confidence"]
                    doc.raw_text = mask_aadhaar(preview_text)
            except Exception:
                pass

        self.documents[filename] = doc
        self.is_verified = False
        self.verification_results = None
        return doc

    def remove_file(self, filename: str) -> bool:
        """Removes a document from the staging queue."""
        if filename in self.documents:
            del self.documents[filename]
            self.is_verified = False
            self.verification_results = None
            return True
        return False

    def clear(self):
        """Clears all queued documents."""
        self.documents.clear()
        self.is_verified = False
        self.verification_results = None

    def get_queue_summary(self) -> List[Dict[str, Any]]:
        """Returns metadata list of all staged documents for UI display."""
        summary = []
        for doc in self.documents.values():
            summary.append({
                "Filename": doc.filename,
                "File Size": doc.size_formatted,
                "Detected Document Type": doc.detected_type,
                "Status": doc.status
            })
        return summary

    def process_document_ocr(self, doc: StagedDocument, ocr_reader, temp_dir: str = "temp_uploads") -> None:
        """Runs text extraction, classification, and slot parsing for a staged document."""
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, doc.filename)
        with open(file_path, "wb") as f:
            f.write(doc.file_bytes)

        is_pdf = doc.filename.lower().endswith(".pdf")
        ocr_items = []
        full_text = ""

        if is_pdf:
            full_text_pages = []
            try:
                reader = PdfReader(file_path)
                num_pages = len(reader.pages)
            except Exception:
                reader = None
                num_pages = 0

            # If PdfReader fails to get page count, check PyMuPDF safely
            if num_pages == 0:
                try:
                    import fitz
                    with fitz.open(file_path) as pdf_doc:
                        num_pages = len(pdf_doc)
                except Exception:
                    num_pages = 0

            for p_idx in range(num_pages):
                page_num = p_idx + 1
                page_digital_text = ""
                if reader and p_idx < len(reader.pages):
                    try:
                        page_digital_text = (reader.pages[p_idx].extract_text() or "").strip()
                    except Exception:
                        page_digital_text = ""

                if len(page_digital_text) >= 30:
                    # Usable digital text exists: use digital extraction without fake OCR confidence
                    lines = [line.strip() for line in page_digital_text.splitlines() if line.strip()]
                    for line in lines:
                        ocr_items.append({
                            "text": line,
                            "bbox": None,
                            "conf": None,
                            "extraction_method": "DIGITAL_TEXT_EXTRACTION",
                            "page": page_num
                        })
                    full_text_pages.append(page_digital_text)
                elif ocr_reader is not None:
                    # Scanned page (or digital text < 30 chars): render via PyMuPDF and perform Vision OCR
                    try:
                        page_img = DocumentPreprocessor.render_pdf_page_to_image(file_path, page_idx=p_idx, dpi=300)
                        if page_img is not None:
                            preprocessed = DocumentPreprocessor.preprocess_image_pipeline(page_img)
                            page_ocr = ocr_reader.readtext(preprocessed)
                            page_lines = []
                            for bbox, text, conf in page_ocr:
                                t_clean = text.strip()
                                if not t_clean:
                                    continue
                                xs = [pt[0] for pt in bbox]
                                ys = [pt[1] for pt in bbox]
                                box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                                ocr_items.append({
                                    "text": t_clean,
                                    "bbox": box,
                                    "conf": float(conf),
                                    "extraction_method": "VISION_OCR",
                                    "page": page_num
                                })
                                page_lines.append(t_clean)
                            if page_lines:
                                full_text_pages.append("\n".join(page_lines))
                    except Exception as e:
                        doc.error_message = f"Error reading scanned PDF page {page_num}: {e}"
                elif page_digital_text:
                    full_text_pages.append(page_digital_text)

            full_text = "\n".join(full_text_pages)
        else:
            if ocr_reader is not None:
                try:
                    raw_img = DocumentPreprocessor.load_image_with_exif(file_path)
                    if raw_img is not None:
                        preprocessed = DocumentPreprocessor.preprocess_image_pipeline(raw_img)
                        results = ocr_reader.readtext(preprocessed)
                    else:
                        results = ocr_reader.readtext(file_path)
                    for bbox, text, conf in results:
                        t_clean = text.strip()
                        if not t_clean:
                            continue
                        xs = [pt[0] for pt in bbox]
                        ys = [pt[1] for pt in bbox]
                        box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                        ocr_items.append({
                            "text": t_clean,
                            "bbox": box,
                            "conf": float(conf),
                            "extraction_method": "VISION_OCR",
                            "page": 1
                        })
                    full_text = "\n".join([item["text"] for item in ocr_items])
                except Exception as e:
                    doc.error_message = f"OCR error: {e}"

        # Mask Aadhaar numbers from full_text to prevent privacy leaks
        doc.raw_text = mask_aadhaar(full_text)

        # Classify document type from extracted text
        cls_res = classify_document_text(full_text)
        doc.detected_type = cls_res["document_type"]
        doc.classification_confidence = cls_res["confidence"]

        # Category-Specific Field Extraction
        extracted: Dict[str, Any] = {}
        lineage: List[Dict[str, Any]] = []

        if doc.detected_type == DOC_TYPE_PAN:
            pan_details = extract_pan_details(ocr_items, full_text, source_doc_name="PAN_CARD")
            extracted["pan"] = pan_details["pan"]
            extracted["pan_status"] = pan_details["pan_status"]
            extracted["entity_type"] = pan_details["entity_type"]
            extracted["cardholder_name"] = pan_details["cardholder_name"]
            extracted["name_status"] = pan_details["name_status"]
            extracted["father_name"] = pan_details.get("father_name")
            extracted["father_name_status"] = pan_details.get("father_name_status")
            extracted["dob"] = pan_details.get("dob")
            extracted["dob_status"] = pan_details.get("dob_status")
            extracted["masked_pan"] = pan_details.get("masked_pan")
            extracted["masked_dob"] = pan_details.get("masked_dob")
            lineage.extend(pan_details["lineage"])

        elif doc.detected_type == DOC_TYPE_GST:
            # GSTIN extraction with checksum verification
            extracted_gstin = "NOT DETECTED"
            for item in ocr_items:
                t_up = item["text"].upper()
                m = re.search(r'\b([0-9O]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z|2][0-9A-Z]{1})\b', t_up)
                if m:
                    cand = list(m.group(1))
                    if cand[0] in ['O', 'D']: cand[0] = '0'
                    if cand[1] in ['O', 'D']: cand[1] = '0'
                    if cand[13] in ['2', '7', 'C', 'c', 'z', '€', 'Γé¼']: cand[13] = 'Z'
                    cand_str = "".join(cand)
                    if is_valid_gstin(cand_str, verify_checksum=True):
                        extracted_gstin = cand_str
                        lineage.append({
                            "document": "GST_CERTIFICATE",
                            "field": "gstin",
                            "value": extracted_gstin,
                            "bbox": item["bbox"],
                            "confidence": round(float(item["conf"]), 3)
                        })
                        break
            extracted["gstin"] = extracted_gstin

            # Legal name extraction
            biz_kws = ["PVT LTD", "PRIVATE LIMITED", "LTD", "LIMITED", "ENTERPRISES", "SOLUTIONS", "INFRA", "LOGISTICS", "INDUSTRIES", "CORP", "LLP"]
            gst_name = "NOT DETECTED"
            for item in ocr_items:
                t_up = item["text"].upper()
                if any(k in t_up for k in biz_kws) and not any(h in t_up for h in ["GOVERNMENT", "DEPARTMENT", "INCOME TAX", "CERTIFICATE"]):
                    cleaned = re.sub(r'[\u0900-\u097F]', '', item["text"]).strip()
                    if len(cleaned) > 4:
                        gst_name = cleaned
                        lineage.append({
                            "document": "GST_CERTIFICATE",
                            "field": "entity_name",
                            "value": gst_name,
                            "bbox": item["bbox"],
                            "confidence": round(float(item["conf"]), 3)
                        })
                        break
            extracted["entity_name"] = gst_name

            # Date extraction
            date_match = re.search(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b', full_text)
            extracted["issue_date"] = date_match.group(1) if date_match else "NOT DETECTED"

            # Turnover
            to_match = re.search(r'[:\-]?\s*(?:Rs\.?\s*)?(\d+(?:\.\d+)?)\s*(Lakhs|Lakh|Cr|Crore)', full_text, re.IGNORECASE)
            if to_match:
                val = float(to_match.group(1))
                if to_match.group(2).lower() in ("cr", "crore"):
                    val *= 100
                extracted["turnover"] = val
                extracted["turnover_str"] = f"Rs. {val:g} Lakhs"
            else:
                extracted["turnover"] = 0.0
                extracted["turnover_str"] = "Rs. 0.0 Lakhs"

        elif doc.detected_type == DOC_TYPE_BALANCE_SHEET:
            to_match = re.search(r'[:\-]?\s*(?:Rs\.?\s*)?(\d+(?:\.\d+)?)\s*(Lakhs|Lakh|Cr|Crore)', full_text, re.IGNORECASE)
            if to_match:
                val = float(to_match.group(1))
                if to_match.group(2).lower() in ("cr", "crore"):
                    val *= 100
                extracted["bs_turnover"] = val
                extracted["bs_turnover_str"] = f"Rs. {val:g} Lakhs"
            else:
                extracted["bs_turnover"] = None
                extracted["bs_turnover_str"] = "NOT EXTRACTED"

        doc.extracted_fields = extracted
        doc.lineage = lineage
        doc.status = "Verified" if not doc.error_message else "Error"

    def execute_consolidated_verification(
        self,
        ocr_reader,
        min_turnover_req: float = 75.0,
        reg_cutoff_date: str = "31/12/2026"
    ) -> Dict[str, Any]:
        """
        Executes staged processing of all queued documents and runs
        statutory and forensic cross-document compliance verification.
        """
        if self.is_verified and self.verification_results is not None:
            # Prevent duplicate processing on normal Streamlit reruns
            return self.verification_results

        all_lineage = []
        all_extracted_texts = {}

        for doc in self.documents.values():
            if doc.status != "Verified" or not doc.extracted_fields:
                self.process_document_ocr(doc, ocr_reader)
            all_lineage.extend(doc.lineage)
            all_extracted_texts[f"{doc.filename} ({doc.detected_type})"] = doc.raw_text

        # Aggregate Dossier Slots across detected document categories
        gst_docs = [d for d in self.documents.values() if d.detected_type == DOC_TYPE_GST]
        pan_docs = [d for d in self.documents.values() if d.detected_type == DOC_TYPE_PAN]
        bs_docs = [d for d in self.documents.values() if d.detected_type == DOC_TYPE_BALANCE_SHEET]

        # Multi-document extraction collections (preserve separate evidence for each document)
        all_pan_extractions = []
        for p_doc in pan_docs:
            all_pan_extractions.append({
                "filename": p_doc.filename,
                "fields": p_doc.extracted_fields,
                "lineage": p_doc.lineage
            })

        all_gst_extractions = []
        for g_doc in gst_docs:
            all_gst_extractions.append({
                "filename": g_doc.filename,
                "fields": g_doc.extracted_fields,
                "lineage": g_doc.lineage
            })

        gst_info = gst_docs[0].extracted_fields if gst_docs else {}
        bs_info = bs_docs[0].extracted_fields if bs_docs else {}

        gstin_no = gst_info.get("gstin", "NOT DETECTED")
        company_name = gst_info.get("entity_name", "NOT DETECTED")
        gst_turnover = gst_info.get("turnover", 0.0)
        gst_turnover_str = gst_info.get("turnover_str", "Rs. 0.0 Lakhs")
        doc_date = gst_info.get("issue_date", "NOT DETECTED")

        # Multi-PAN matching logic: identify primary matching entity PAN against GSTIN
        primary_pan_doc = None
        if pan_docs and gstin_no != "NOT DETECTED" and len(gstin_no) >= 12:
            gst_pan = gstin_no[2:12]
            for p_doc in pan_docs:
                if p_doc.extracted_fields.get("pan") == gst_pan:
                    primary_pan_doc = p_doc
                    break
        if primary_pan_doc is None and pan_docs:
            primary_pan_doc = pan_docs[0]

        pan_info = primary_pan_doc.extracted_fields if primary_pan_doc else {}

        standalone_pan = pan_info.get("pan", "NOT DETECTED")
        pan_name = pan_info.get("cardholder_name", "NOT DETECTED")
        pan_name_status = pan_info.get("name_status", "NOT_VERIFIED")
        pan_entity_type = pan_info.get("entity_type", "UNKNOWN")
        father_name = pan_info.get("father_name")
        father_name_status = pan_info.get("father_name_status", "NOT_VERIFIED")
        dob = pan_info.get("dob")
        dob_status = pan_info.get("dob_status", "NOT_VERIFIED")
        masked_pan = pan_info.get("masked_pan")
        masked_dob = pan_info.get("masked_dob")

        bs_turnover = bs_info.get("bs_turnover")
        bs_turnover_str = bs_info.get("bs_turnover_str", "NOT PROVIDED") if bs_docs else "NOT PROVIDED"

        # Cross-Document Consistency Engine Integration
        norm_gst_name = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", company_name.upper())).strip()
        norm_pan_name = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", pan_name.upper())).strip()

        # Embedded PAN comparison
        pan_gstin_check = compare_pan_with_gstin(standalone_pan, gstin_no)

        consistency_engine = CrossDocumentConsistencyEngine(turnover_tolerance_percentage=5.0)
        dossier = {
            "pan_card": {
                "pan_number": standalone_pan,
                "legal_name": norm_pan_name
            },
            "gst_certificate": {
                "gstin": gstin_no,
                "legal_name": norm_gst_name,
                "annual_turnover": gst_turnover
            },
            "balance_sheet": {
                "audited_turnover": bs_turnover
            }
        }
        p9_result = consistency_engine.evaluate_dossier_consistency(dossier)

        # Statutory Engine Integration
        compliance_engine = ComplianceVerificationEngine()
        ocr_payload = {
            "document_id": gst_docs[0].filename if gst_docs else "MULTI-DOC-UPLOAD",
            "extracted_fields": {
                "gstin": gstin_no,
                "entity_name": company_name,
                "turnover": gst_turnover,
                "detected_date": doc_date
            }
        }
        tender_rules = {
            "min_turnover": float(min_turnover_req),
            "tender_submission_date": reg_cutoff_date
        }
        p7_result = compliance_engine.evaluate_bid_compliance(ocr_payload, tender_rules)

        # Audit Trail Assembly
        audit_data = []

        # 1. Statutory Rules
        for r in p7_result.get("rule_details", []):
            audit_data.append({
                "Stage": "Phase 7 Statutory",
                "Rule Checked": r.get("rule"),
                "Status": "PASS" if r.get("passed") else "FAIL",
                "Finding Details": r.get("reason") or "Rule requirement satisfied."
            })

        # 2. PAN Forensic Cross-Check(s)
        audit_data.append({
            "Stage": "Phase 9 Forensic",
            "Rule Checked": pan_gstin_check["check"],
            "Status": pan_gstin_check["status"],
            "Finding Details": pan_gstin_check.get("reason") or f"Primary entity PAN '{standalone_pan}' matches GSTIN embedded PAN."
        })
        if primary_pan_doc:
            audit_data.append({
                "Stage": "Phase 9 Forensic",
                "Rule Checked": "PAN Cardholder Name Verification",
                "Status": pan_name_status,
                "Finding Details": f"Cardholder Name '{pan_name}' (Status: {pan_name_status}). Missing name is never derived from GST."
            })
            if pan_entity_type != "UNKNOWN":
                audit_data.append({
                    "Stage": "Phase 9 Forensic",
                    "Rule Checked": "PAN Entity Classification",
                    "Status": "PASS",
                    "Finding Details": f"Entity Category: {pan_entity_type} (derived from statutory 4th character)."
                })
            if dob:
                audit_data.append({
                    "Stage": "Phase 9 Forensic",
                    "Rule Checked": "PAN Date of Birth / Incorporation",
                    "Status": dob_status,
                    "Finding Details": f"Detected Date: {masked_dob or dob} (Status: {dob_status})."
                })

        # Secondary / Signatory PAN documents
        if len(pan_docs) > 1:
            for p_doc in pan_docs:
                if p_doc == primary_pan_doc:
                    continue
                p_pan = p_doc.extracted_fields.get("pan", "NOT DETECTED")
                p_name = p_doc.extracted_fields.get("cardholder_name", "NOT DETECTED")
                p_entity = p_doc.extracted_fields.get("entity_type", "UNKNOWN")
                p_name_status = p_doc.extracted_fields.get("name_status", "NOT_VERIFIED")
                audit_data.append({
                    "Stage": "Phase 9 Forensic",
                    "Rule Checked": f"Secondary / Partner PAN ({p_doc.filename}) Verification",
                    "Status": "PASS" if is_valid_pan(p_pan) else "FLAGGED",
                    "Finding Details": f"Secondary PAN '{p_pan}' ({p_entity}) logged for partner/signatory '{p_name}' (Status: {p_name_status})."
                })

        # 3. Turnover Consistency (if Balance Sheet provided)
        for c in p9_result.get("audit_trail", []):
            if "Turnover" in c.get("check", ""):
                audit_data.append({
                    "Stage": "Phase 9 Forensic",
                    "Rule Checked": c["check"],
                    "Status": c["status"],
                    "Finding Details": c.get("reason") or "Turnover variance within allowable threshold."
                })

        # Missing optional documents do not automatically cause total failure
        hard_failures = [a for a in audit_data if a["Status"] == "FAIL"]
        pending_reviews = [a for a in audit_data if a["Status"] in ["PENDING_REVIEW", "NOT_VERIFIED"]]

        if len(hard_failures) > 0:
            compliance_status = "FLAGGED"
        elif len(pending_reviews) > 0:
            compliance_status = "PENDING_REVIEW"
        else:
            compliance_status = "PASSED"

        results = {
            "is_verified": True,
            "compliance_status": compliance_status,
            "discrepancies_count": len(hard_failures),
            "pending_review_count": len(pending_reviews),
            "audit_data": audit_data,
            "all_lineage": all_lineage,
            "all_extracted_texts": all_extracted_texts,
            "p7_result": p7_result,
            "p9_result": p9_result,
            "slots": {
                "company_name": company_name,
                "gstin_no": gstin_no,
                "pan_no": standalone_pan if standalone_pan != "NOT DETECTED" else (gstin_no[2:12] if len(gstin_no) >= 12 else "NOT FOUND"),
                "pan_name": pan_name,
                "pan_name_status": pan_name_status,
                "pan_entity_type": pan_entity_type,
                "father_name": father_name,
                "father_name_status": father_name_status,
                "dob": dob,
                "dob_status": dob_status,
                "masked_pan": masked_pan,
                "masked_dob": masked_dob,
                "gst_turnover_val": gst_turnover_str,
                "bs_turnover_val": bs_turnover_str,
                "doc_date": doc_date,
                "gst_docs_count": len(gst_docs),
                "pan_docs_count": len(pan_docs),
                "bs_docs_count": len(bs_docs),
                "total_docs": len(self.documents),
                "all_pan_extractions": all_pan_extractions,
                "all_gst_extractions": all_gst_extractions,
                "multi_pan_count": len(pan_docs),
                "multi_gst_count": len(gst_docs)
            }
        }

        self.is_verified = True
        self.verification_results = results
        return results
