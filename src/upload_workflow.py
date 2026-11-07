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
            # GSTIN candidate extraction: search full text, individual items, and token windows
            # Tolerates OCR whitespace/dashes between components and normalizes O->0, I->1, 2->Z
            extracted_gstin = "NOT DETECTED"
            gstin_status = "NOT_DETECTED"
            gstin_checksum_valid = False
            gstin_chk_details = None

            def try_parse_gstin(cand_raw: str) -> Optional[str]:
                clean = re.sub(r'[^A-Z0-9]', '', cand_raw.upper())
                if len(clean) == 14 and clean[0] in '123456789':
                    clean = '0' + clean
                elif len(clean) == 14 and clean[:2].isdigit() and clean[13] in 'Z27%':
                    from src.ocr_pipeline import compute_gstin_checksum
                    try:
                        clean = clean + compute_gstin_checksum(clean[:14])
                    except Exception:
                        pass
                if len(clean) != 15:
                    return None
                st = clean[:2].replace('O', '0').replace('D', '0').replace('Q', '0')
                p1 = clean[2:7].replace('0', 'O').replace('8', 'B').replace('1', 'I')
                p2 = clean[7:11].replace('O', '0').replace('D', '0').replace('I', '1').replace('L', '1')
                p3 = clean[11].replace('0', 'O').replace('8', 'B').replace('1', 'I')
                ent = '1' if clean[12] in ['I', 'L', '|'] else clean[12]
                z = 'Z'
                chk = clean[14]
                cand = f"{st}{p1}{p2}{p3}{ent}{z}{chk}"
                if is_valid_gstin(cand, verify_checksum=False):
                    return cand
                return None

            sources_to_search = [it["text"] for it in ocr_items]
            n_items = len(ocr_items)
            for w in range(2, 6):
                for i in range(max(0, n_items - w + 1)):
                    sources_to_search.append(" ".join([ocr_items[j]["text"] for j in range(i, i + w)]))
            sources_to_search.append(full_text)

            gstin_regex = r'([0-9ODQo]{1,2})[\s\-]*([A-Z0-9\s\-]{10,16})[\s\-]*([1-9A-ZIl|]{1})[\s\-]*([Z2%7sS\=]{1})[\s\-]*([0-9A-Z]{1})'

            for src in sources_to_search:
                if not src:
                    continue
                # Try regex matching
                for m in re.finditer(gstin_regex, src.upper()):
                    parsed = try_parse_gstin(m.group(0))
                    if parsed:
                        extracted_gstin = parsed
                        break
                if extracted_gstin != "NOT DETECTED":
                    break

                # Try scanning clean alphanumeric substrings of length 14-17
                clean_src = re.sub(r'[^A-Z0-9]', '', src.upper())
                for i in range(len(clean_src) - 13):
                    for sublen in (14, 15):
                        if i + sublen <= len(clean_src):
                            parsed = try_parse_gstin(clean_src[i:i+sublen])
                            if parsed:
                                extracted_gstin = parsed
                                break
                    if extracted_gstin != "NOT DETECTED":
                        break
                if extracted_gstin != "NOT DETECTED":
                    break

            # Separately printed PAN on GST Certificate (and comparison with embedded PAN)
            embedded_pan = extracted_gstin[2:12] if (extracted_gstin != "NOT DETECTED" and len(extracted_gstin) == 15) else None
            separate_pan = None
            fallback_pan = None
            pan_bbox = None
            pan_conf = None

            def try_parse_pan(raw_candidate: str) -> Optional[str]:
                if not raw_candidate:
                    return None
                for m in re.finditer(r'([A-Za-z0-9]{5})[\s\-]*([A-Za-z0-9]{4})[\s\-]*([A-Za-z0-9]{1})', raw_candidate):
                    p1 = m.group(1).upper().replace('0', 'O').replace('8', 'B').replace('1', 'I').replace('5', 'S').replace('2', 'Z')
                    p2 = m.group(2).upper().replace('O', '0').replace('D', '0').replace('Q', '0').replace('I', '1').replace('L', '1').replace('Z', '2').replace('S', '5')
                    p3 = m.group(3).upper().replace('0', 'O').replace('8', 'B').replace('1', 'I')
                    cand = f"{p1}{p2}{p3}"
                    if is_valid_pan(cand):
                        return cand
                return None

            # 1. Search full_text with flexible pattern following "PERMANENT ACCOUNT NUMBER" or "PAN"
            m_pan_lbl = re.search(r'(?:PERMANENT\s+ACCOUNT\s+NUMBER|PAN\s+NUMBER|PAN)[\s\:\.\-\w]{0,35}?([A-Z0-9\s\-]{10,22})', full_text.upper())
            if m_pan_lbl:
                parsed_lbl = try_parse_pan(m_pan_lbl.group(1))
                if parsed_lbl:
                    if embedded_pan and parsed_lbl != embedded_pan:
                        separate_pan = parsed_lbl
                    elif not fallback_pan:
                        fallback_pan = parsed_lbl

            # 2. Search label-anchored items in ocr_items
            if not separate_pan:
                for idx, item in enumerate(ocr_items):
                    t_up = item["text"].upper()
                    if any(k in t_up for k in ["PERMANENT", "PAN", "ACCOUNT NUMBER"]):
                        window_text = " ".join([ocr_items[k]["text"] for k in range(idx, min(idx + 6, len(ocr_items)))])
                        m_cand = re.search(r'([A-Z0-9\s\-]{10,20})', window_text.upper())
                        if m_cand:
                            parsed_cand = try_parse_pan(m_cand.group(1))
                            if parsed_cand:
                                if embedded_pan and parsed_cand != embedded_pan:
                                    separate_pan = parsed_cand
                                    pan_bbox = item.get("bbox")
                                    pan_conf = item.get("conf")
                                    break
                                elif not fallback_pan:
                                    fallback_pan = parsed_cand
                                    pan_bbox = item.get("bbox")
                                    pan_conf = item.get("conf")

            # 3. Sliding window search across all ocr_items (tolerates split tokens like 'ABCDE' + '1234F')
            if not separate_pan:
                for w in range(1, 4):
                    for i in range(max(0, n_items - w + 1)):
                        joined = " ".join([ocr_items[j]["text"] for j in range(i, i + w)])
                        clean_joined = re.sub(r'[^A-Z0-9]', '', joined.upper())
                        if len(clean_joined) >= 15 and clean_joined.startswith(('0', '1', '2', '3', '9')):
                            continue
                        for start_idx in range(len(clean_joined) - 9):
                            cand_slice = clean_joined[start_idx:start_idx + 10]
                            parsed_cand = try_parse_pan(cand_slice)
                            if parsed_cand:
                                if embedded_pan and parsed_cand != embedded_pan:
                                    separate_pan = parsed_cand
                                    pan_bbox = ocr_items[i].get("bbox")
                                    pan_conf = ocr_items[i].get("conf")
                                    break
                                elif not fallback_pan and any(k in joined.upper() for k in ["PAN", "PERMANENT", "ACCOUNT", "NUMBER"]):
                                    fallback_pan = parsed_cand
                                    pan_bbox = ocr_items[i].get("bbox")
                                    pan_conf = ocr_items[i].get("conf")
                        if separate_pan:
                            break
                    if separate_pan:
                        break

            # 4. Standalone items fallback
            if not separate_pan:
                for item in ocr_items:
                    parsed_cand = try_parse_pan(item["text"])
                    if parsed_cand:
                        if embedded_pan and parsed_cand != embedded_pan:
                            separate_pan = parsed_cand
                            pan_bbox = item.get("bbox")
                            pan_conf = item.get("conf")
                            break
                        elif not fallback_pan and any(k in item["text"].upper() for k in ["PAN", "PERMANENT", "ACCOUNT", "NUMBER"]):
                            fallback_pan = parsed_cand
                            pan_bbox = item.get("bbox")
                            pan_conf = item.get("conf")

            if not separate_pan and fallback_pan:
                separate_pan = fallback_pan

            # If GSTIN was not detected but a PAN was detected, look for adjacent state code & entity code
            if extracted_gstin == "NOT DETECTED" and separate_pan:
                # Search full_text and ocr_items around separate_pan
                m_st_ent = re.search(r'([0-9ODQo]{1,2})[\s\-]*' + re.escape(separate_pan) + r'[\s\-]*([1-9A-ZIl|]{1})[\s\-]*([Z2%7sS\=]{0,1})[\s\-]*([0-9A-Z]{1})', full_text.upper())
                if m_st_ent:
                    st_val = m_st_ent.group(1).replace('O', '0').replace('D', '0')
                    if len(st_val) == 1:
                        st_val = '0' + st_val
                    ent_val = '1' if m_st_ent.group(2) in ['I', 'L', '|'] else m_st_ent.group(2)
                    chk_val = m_st_ent.group(4)
                    cand_reconstructed = f"{st_val}{separate_pan}{ent_val}Z{chk_val}"
                    if is_valid_gstin(cand_reconstructed, verify_checksum=False):
                        extracted_gstin = cand_reconstructed
                else:
                    # Look across ocr_items around the item containing separate_pan
                    for idx, it in enumerate(ocr_items):
                        if separate_pan in it["text"].upper():
                            # Check window [idx-2 : idx+3]
                            win_text = " ".join([ocr_items[k]["text"] for k in range(max(0, idx - 2), min(len(ocr_items), idx + 3))])
                            cand_parsed = try_parse_gstin(win_text)
                            if cand_parsed:
                                extracted_gstin = cand_parsed
                                break

            # Recalculate embedded PAN if GSTIN was newly extracted
            if extracted_gstin != "NOT DETECTED":
                embedded_pan = extracted_gstin[2:12]
                gstin_chk_details = validate_gstin_checksum(extracted_gstin)
                gstin_checksum_valid = gstin_chk_details.get("is_valid", False)
                gstin_status = "VALID_FORMAT_CHECKSUM_PASS" if gstin_checksum_valid else "CHECKSUM_FAILED"
                lineage.append({
                    "document": "GST_CERTIFICATE",
                    "field": "gstin",
                    "value": extracted_gstin,
                    "bbox": None,
                    "confidence": "NOT_AVAILABLE",
                    "page_number": 1
                })

            extracted["gstin"] = extracted_gstin
            extracted["gstin_status"] = gstin_status
            extracted["gstin_checksum_valid"] = gstin_checksum_valid
            extracted["gstin_checksum_details"] = gstin_chk_details

            final_pan = separate_pan or embedded_pan
            extracted["pan"] = final_pan
            extracted["embedded_gstin_pan"] = embedded_pan
            extracted["separate_printed_pan"] = separate_pan
            if final_pan:
                lineage.append({
                    "document": "GST_CERTIFICATE",
                    "field": "pan",
                    "value": final_pan,
                    "bbox": pan_bbox,
                    "confidence": round(float(pan_conf), 3) if pan_conf is not None else "NOT_AVAILABLE",
                    "page_number": 1
                })

            if separate_pan and embedded_pan:
                extracted["pan_consistency"] = (separate_pan == embedded_pan)
                if separate_pan != embedded_pan:
                    extracted["pan_discrepancy_note"] = f"Separately printed PAN '{separate_pan}' does not match GSTIN embedded PAN '{embedded_pan}'."
            else:
                extracted["pan_consistency"] = "NOT_VERIFIED"

            # Legal name extraction
            biz_kws = ["PVT LTD", "PRIVATE LIMITED", "LTD", "LIMITED", "ENTERPRISES", "SOLUTIONS", "INFRA", "LOGISTICS", "INDUSTRIES", "CORP", "LLP"]
            gst_name = "NOT DETECTED"
            for item in ocr_items:
                t_up = item["text"].upper()
                if any(k in t_up for k in biz_kws) and not any(h in t_up for h in ["GOVERNMENT", "DEPARTMENT", "INCOME TAX", "CERTIFICATE"]):
                    cleaned = re.sub(r'[\u0900-\u097F]', '', item["text"]).strip()
                    cleaned = re.sub(r'^(?:\d+[\.\)]\s*)?(?:LEGAL\s+NAME|TRADE\s+NAME|NAME|ENTITY\s+NAME)\s*[:\-]?\s*', '', cleaned, flags=re.IGNORECASE).strip()
                    if len(cleaned) > 4:
                        gst_name = cleaned
                        lineage.append({
                            "document": "GST_CERTIFICATE",
                            "field": "entity_name",
                            "value": gst_name,
                            "bbox": item["bbox"],
                            "confidence": round(float(item["conf"]), 3) if item.get("conf") is not None else "NOT_AVAILABLE",
                            "page_number": item.get("page", 1)
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

        elif doc.detected_type == DOC_TYPE_UDYAM:
            extracted_udyam = "NOT DETECTED"
            # Flexible pattern across full text and items
            udyam_regex = r'UDYAM[\s\-_:\.]*([A-Za-z]{2})[\s\-_:\.]*([0-9OIl|]{2})[\s\-_:\.]*([0-9OIl|DQ\s]{7,12})'
            m_udyam = re.search(udyam_regex, full_text.upper())
            if not m_udyam:
                for item in ocr_items:
                    m_item = re.search(udyam_regex, item["text"].upper())
                    if m_item:
                        m_udyam = m_item
                        break

            if m_udyam:
                st = m_udyam.group(1).upper()
                dist = m_udyam.group(2).upper().replace('O', '0').replace('D', '0').replace('I', '1').replace('L', '1').replace('|', '1')
                raw_seq = re.sub(r'[^A-Z0-9]', '', m_udyam.group(3).upper())[:7]
                seq = raw_seq.replace('O', '0').replace('D', '0').replace('Q', '0').replace('I', '1').replace('L', '1').replace('|', '1')
                if st.isalpha() and len(st) == 2 and dist.isdigit() and len(dist) == 2 and seq.isdigit() and len(seq) == 7:
                    extracted_udyam = f"UDYAM-{st}-{dist}-{seq}"

            # Secondary fallback: search for 'UDYAM' and take the 11 cleaned characters following it
            if extracted_udyam == "NOT DETECTED" and "UDYAM" in full_text.upper():
                idx_u = full_text.upper().find("UDYAM")
                segment = full_text.upper()[idx_u + 5: idx_u + 45]
                clean_seg = re.sub(r'[\s\-_\.:\/]', '', segment)
                if len(clean_seg) >= 11:
                    cand_st = clean_seg[:2]
                    cand_dist = clean_seg[2:4].replace('O', '0').replace('D', '0').replace('I', '1').replace('L', '1').replace('|', '1')
                    cand_seq = clean_seg[4:11].replace('O', '0').replace('D', '0').replace('Q', '0').replace('I', '1').replace('L', '1').replace('|', '1')
                    if cand_st.isalpha() and cand_dist.isdigit() and cand_seq.isdigit():
                        extracted_udyam = f"UDYAM-{cand_st}-{cand_dist}-{cand_seq}"

            if extracted_udyam != "NOT DETECTED":
                lineage.append({
                    "document": "UDYAM_CERTIFICATE",
                    "field": "udyam",
                    "value": extracted_udyam,
                    "bbox": None,
                    "confidence": "NOT_AVAILABLE",
                    "page_number": 1
                })
            extracted["udyam"] = extracted_udyam

            # Enterprise Name
            udyam_name = "NOT DETECTED"
            biz_kws = ["ENTERPRISES", "PVT LTD", "LTD", "LIMITED", "INDUSTRIES", "TRADERS", "COMPANY"]
            for item in ocr_items:
                t_up = item["text"].upper()
                if any(k in t_up for k in biz_kws) and not any(h in t_up for h in ["MINISTRY", "GOVERNMENT", "MICRO", "SMALL", "MEDIUM"]):
                    cleaned = re.sub(r'[\u0900-\u097F]', '', item["text"]).strip()
                    cleaned = re.sub(r'^(?:\d+[\.\)]\s*)?(?:NAME\s+OF\s+ENTERPRISE|ENTERPRISE\s+NAME|NAME)\s*[:\-]?\s*', '', cleaned, flags=re.IGNORECASE).strip()
                    if len(cleaned) > 4:
                        udyam_name = cleaned
                        lineage.append({
                            "document": "UDYAM_CERTIFICATE",
                            "field": "entity_name",
                            "value": udyam_name,
                            "bbox": item["bbox"],
                            "confidence": round(float(item["conf"]), 3) if item.get("conf") is not None else "NOT_AVAILABLE",
                            "page_number": item.get("page", 1)
                        })
                        break
            extracted["entity_name"] = udyam_name

        elif doc.detected_type == DOC_TYPE_BALANCE_SHEET:
            to_match = re.search(r'(?:Turnover|Revenue|Sales)[^\n\r\d]*?[:\-]?\s*(?:Rs\.?\s*)?(\d+(?:\.\d+)?)\s*(Lakhs|Lakh|Cr|Crore)', full_text, re.IGNORECASE)
            if not to_match:
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
        norm_gst_name = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", company_name.upper())).strip() if (company_name and company_name != "NOT DETECTED") else None
        norm_pan_name = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", pan_name.upper())).strip() if (pan_docs and pan_name and pan_name != "NOT DETECTED") else None
        pan_num_for_dossier = standalone_pan if (pan_docs and standalone_pan != "NOT DETECTED") else None

        # Embedded PAN comparison
        pan_gstin_check = compare_pan_with_gstin(standalone_pan, gstin_no)

        consistency_engine = CrossDocumentConsistencyEngine(turnover_tolerance_percentage=5.0)
        dossier = {
            "pan_card": {
                "pan_number": pan_num_for_dossier,
                "legal_name": norm_pan_name
            },
            "gst_certificate": {
                "gstin": gstin_no if gstin_no != "NOT DETECTED" else None,
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

        # 3. Cross-Document Consistency Checks (Entity Name & Turnover)
        # Explicitly find and append the Entity Name cross-document check
        name_check = next((c for c in p9_result.get("audit_trail", []) if "Entity Name" in c.get("check", "")), None)
        if name_check:
            audit_data.append({
                "Stage": "Phase 9 Forensic",
                "Rule Checked": name_check["check"],
                "Status": name_check["status"],
                "Finding Details": name_check.get("reason") or "Legal entity name matches across certificates."
            })
        elif norm_gst_name and norm_pan_name:
            name_res = consistency_engine.check_entity_name_consistency(norm_gst_name, norm_pan_name, "GST Certificate", "PAN Card")
            audit_data.append({
                "Stage": "Phase 9 Forensic",
                "Rule Checked": name_res["check"],
                "Status": name_res["status"],
                "Finding Details": name_res.get("reason") or "Legal entity name matches across certificates."
            })

        # Turnover Consistency (if Balance Sheet provided)
        to_check = next((c for c in p9_result.get("audit_trail", []) if "Turnover" in c.get("check", "")), None)
        if to_check:
            audit_data.append({
                "Stage": "Phase 9 Forensic",
                "Rule Checked": to_check["check"],
                "Status": to_check["status"],
                "Finding Details": to_check.get("reason") or "Satisfied without conflict."
            })

        # Append any other forensic checks present in p9 audit_trail
        for c in p9_result.get("audit_trail", []):
            if "Turnover" not in c.get("check", "") and "Entity Name" not in c.get("check", "") and "PAN" not in c.get("check", ""):
                audit_data.append({
                    "Stage": "Phase 9 Forensic",
                    "Rule Checked": c["check"],
                    "Status": c["status"],
                    "Finding Details": c.get("reason") or "Satisfied without conflict."
                })

        # Missing optional documents do not automatically cause total failure
        hard_failures = [a for a in audit_data if a["Status"] == "FAIL"]
        pending_reviews = [a for a in audit_data if a["Status"] in ["PENDING_REVIEW", "NOT_VERIFIED", "REVIEW"]]
        minor_discrepancies = [a for a in audit_data if a["Status"] == "REVIEW"]

        # Discrepancies count includes hard failures as well as minor/review discrepancies
        total_discrepancies_count = len(hard_failures) + len(minor_discrepancies)
        if len(p9_result.get("discrepancies", [])) > total_discrepancies_count:
            total_discrepancies_count = len(p9_result.get("discrepancies", []))

        if len(hard_failures) > 0:
            compliance_status = "FLAGGED"
        elif len(pending_reviews) > 0:
            compliance_status = "PENDING_REVIEW"
        else:
            compliance_status = "PASSED"

        results = {
            "is_verified": True,
            "compliance_status": compliance_status,
            "discrepancies_count": total_discrepancies_count,
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
