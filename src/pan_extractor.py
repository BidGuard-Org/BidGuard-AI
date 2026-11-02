"""Universal PAN Card Document Intelligence Engine for BidGuard AI.

A template-agnostic, spatial, bilingual (English & Hindi) PAN card extraction engine.
Provides:
- Strict statutory syntax validation and entity type classification.
- Spatial bounding-box and bilingual label-value association.
- Deterministic disambiguation of Cardholder Name vs. Father's Name.
- Disambiguation of Date of Birth (DOB) vs. Date of Issue.
- Aadhaar vs. PAN negative separation.
- Sensitive identity masking and auditable evidence lineage.
- Zero hardcoded names, documents, or individual values.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Statutory Indian PAN Pattern: 5 Letters + 4 Digits + 1 Letter
PAN_REGEX = r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b"

# Standard Date Pattern: DD/MM/YYYY or DD-MM-YYYY
DATE_REGEX = r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b"

# Aadhaar Patterns: 12 digits (spaced, hyphenated, or continuous)
AADHAAR_SPACED_REGEX = r"\b([2-9]\d{3})[\s\-]+(\d{4})[\s\-]+(\d{4})\b"
AADHAAR_CONTINUOUS_REGEX = r"\b([2-9]\d{7})(\d{4})\b"
AADHAAR_REGEX = r"\b[2-9]{1}[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b"


def mask_aadhaar(text: Optional[str]) -> str:
    """
    Masks 12-digit Aadhaar numbers for privacy compliance under UIDAI guidelines.
    Preserves only the last 4 digits for forensic classification auditing.
    e.g., '5482 1934 8291' -> 'XXXX-XXXX-8291'
          '5482-1934-8291' -> 'XXXX-XXXX-8291'
          '548219348291'   -> 'XXXXXXXX8291'
    """
    if not text:
        return ""
    # Mask 4-4-4 spaced or hyphenated
    masked = re.sub(AADHAAR_SPACED_REGEX, r"XXXX-XXXX-\3", text)
    # Mask continuous 12-digit sequence
    def _mask_cont(m):
        val = m.group(0)
        return "X" * 8 + val[-4:]
    masked = re.sub(AADHAAR_CONTINUOUS_REGEX, _mask_cont, masked)
    return masked

# Statutory entity code mapping under Indian Income Tax Act (4th character of PAN)
PAN_ENTITY_TYPES = {
    "C": "Company",
    "P": "Individual / Person",
    "H": "Hindu Undivided Family (HUF)",
    "F": "Firm / Limited Liability Partnership (LLP)",
    "A": "Association of Persons (AOP)",
    "T": "Trust",
    "B": "Body of Individuals (BOI)",
    "L": "Local Authority",
    "J": "Artificial Juridical Person",
    "G": "Government Agency"
}

# Standard bilingual boilerplate headers to ignore when isolating names
PAN_BOILERPLATE_HEADERS = [
    "INCOME TAX DEPARTMENT",
    "आयकर विभाग",
    "GOVT. OF INDIA",
    "GOVERNMENT OF INDIA",
    "भारत सरकार",
    "BHARAT SARKAR",
    "AYAKAR VIBHAG",
    "PERMANENT ACCOUNT NUMBER",
    "PERMANENT ACCOUNT NUMBER CARD",
    "ACCOUNT NUMBER CARD",
    "स्थायी लेखा संख्या कार्ड",
    "स्थायी लेखा संख्या",
    "SIGNATURE",
    "हस्ताक्षर",
    "INDIA",
    "TAX",
    "INCOME",
    "DEPARTMENT"
]

# Field Identification Bilingual Marker Keywords
NAME_LABELS = ["NAME", "नाम", "CARDHOLDER NAME", "APPLICANT NAME"]
FATHER_NAME_LABELS = ["FATHER'S NAME", "FATHERS NAME", "FATHER NAME", "पिता का नाम", "पिता"]
DOB_LABELS = ["DATE OF BIRTH", "DOB", "जन्म तिथि", "जन्म तारीख", "DATE OF INCORPORATION", "निगमन की तारीख"]
PAN_LABELS = ["PERMANENT ACCOUNT NUMBER", "PAN", "स्थायी लेखा संख्या"]

# Comprehensive Bilingual Label Token Set for Pure Label Detection & Stripping
LABEL_TOKENS = {
    "NAME", "नाम", "CARDHOLDER", "APPLICANT",
    "FATHER", "FATHERS", "FATHER'S", "पिता", "का", "की", "के",
    "DOB", "BIRTH", "DATE", "OF", "जन्म", "तारीख", "तिथि",
    "INCORPORATION", "निगमन",
    "PAN", "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "स्थायी", "लेखा", "संख्या", "कार्ड",
    "SIGNATURE", "हस्ताक्षर", "PHOTO", "फोटो", "GOVT", "GOVERNMENT", "INDIA", "BHARAT",
    "INCOME", "TAX", "DEPARTMENT", "आयकर", "विभाग", "भारत", "सरकार"
}


def is_label_only_line(text: str) -> bool:
    """Checks if a text token or line consists solely of bilingual label tokens."""
    clean = re.sub(r"[^\w\s\u0900-\u097F]", " ", text).strip().upper()
    tokens = clean.split()
    if not tokens:
        return True
    return all(tok in LABEL_TOKENS or tok == "S" for tok in tokens)


def strip_label_prefix(text: str) -> str:
    """Removes leading bilingual label keywords (e.g. 'Name: MOHIT VERMA' -> 'MOHIT VERMA')."""
    words = text.split()
    idx = 0
    while idx < len(words):
        w_clean = re.sub(r"[^\w\u0900-\u097F]", "", words[idx]).upper()
        if w_clean in LABEL_TOKENS or w_clean in ["S"]:
            idx += 1
        else:
            break
    remaining = " ".join(words[idx:]).strip()
    return remaining if remaining else text


def is_valid_pan(pan: Optional[str], strict_entity: bool = False) -> bool:
    """Validates 10-character Indian PAN statutory syntax.
    
    If strict_entity is True, also verifies that the 4th character matches
    a recognized entity category under the Indian Income Tax Act.
    """
    if not pan or not isinstance(pan, str):
        return False
    clean = pan.strip().upper()
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", clean):
        return False
    if strict_entity:
        return clean[3] in PAN_ENTITY_TYPES
    return True


def get_pan_entity_type(pan: Optional[str]) -> str:
    """Returns statutory entity category from the 4th character of PAN."""
    if not is_valid_pan(pan):
        return "UNKNOWN"
    code = pan.strip().upper()[3]
    return PAN_ENTITY_TYPES.get(code, "Other Legal Entity")


def mask_pan(pan: Optional[str]) -> str:
    """Masks PAN for privacy in officer logs (e.g., 'ABCDE1234F' -> 'XXXXX1234F')."""
    if not pan or len(pan) < 10:
        return "NOT_AVAILABLE"
    clean = pan.strip().upper()
    return "X" * 5 + clean[5:]


def mask_dob(dob: Optional[str]) -> str:
    """Masks day and month of birth for privacy (e.g., '14/05/1988' -> '**/**/1988')."""
    if not dob or len(dob) < 10:
        return "NOT_AVAILABLE"
    return f"**/**/{dob[-4:]}"


def compare_pan_with_gstin(standalone_pan: Optional[str], gstin: Optional[str]) -> Dict[str, Any]:
    """
    Compares standalone PAN against characters 3 to 12 of Indian GSTIN (0-indexed [2:12]).
    Does not invent values if either field is missing.
    """
    clean_pan = standalone_pan.strip().upper() if standalone_pan else None
    clean_gstin = gstin.strip().upper() if gstin else None

    if not clean_pan or clean_pan in ["NOT DETECTED", "NOT FOUND"]:
        return {
            "check": "PAN Embedded in GSTIN Check",
            "status": "NOT_VERIFIED",
            "passed": False,
            "standalone_pan": None,
            "gstin_embedded_pan": clean_gstin[2:12] if clean_gstin and len(clean_gstin) >= 12 else None,
            "reason": "Standalone PAN card was not uploaded or PAN number not detected."
        }

    if not clean_gstin or len(clean_gstin) < 12:
        return {
            "check": "PAN Embedded in GSTIN Check",
            "status": "NOT_VERIFIED",
            "passed": False,
            "standalone_pan": clean_pan,
            "gstin_embedded_pan": None,
            "reason": "GSTIN not available or insufficient length to extract embedded PAN."
        }

    embedded_pan = clean_gstin[2:12]
    match = (clean_pan == embedded_pan)

    return {
        "check": "PAN Embedded in GSTIN Check",
        "status": "PASS" if match else "FAIL",
        "passed": match,
        "standalone_pan": clean_pan,
        "gstin_embedded_pan": embedded_pan,
        "reason": None if match else f"Embedded PAN '{embedded_pan}' in GSTIN does not match standalone PAN '{clean_pan}'."
    }


def clean_text_token(token: str) -> str:
    """Removes non-alphanumeric noise while keeping standard punctuation."""
    return re.sub(r"[^\w\s\.\&\/\-]", "", token).strip()


def check_aadhaar_contamination(raw_text: str, ocr_items: List[Dict[str, Any]]) -> bool:
    """
    Detects if the document is an Aadhaar card instead of a PAN card.
    Guarantees that an Aadhaar card is never misclassified or extracted as a PAN card.
    """
    t_up = raw_text.upper()
    aadhaar_keywords = [
        "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
        "UIDAI",
        "MERA AADHAAR",
        "MERI PEHCHAN",
        "आधार",
        "HELP@UIDAI.GOV.IN",
        "WWW.UIDAI.GOV.IN",
        "VID"
    ]
    if any(k in t_up or k in raw_text for k in aadhaar_keywords):
        return True
    if re.search(AADHAAR_SPACED_REGEX, t_up) or re.search(AADHAAR_CONTINUOUS_REGEX, t_up):
        return True
    return False


def extract_pan_details(
    ocr_items: List[Dict[str, Any]], 
    raw_text: str,
    source_doc_name: str = "PAN_CARD",
    page_number: int = 1
) -> Dict[str, Any]:
    """
    General-purpose, spatial, page-aware and bilingual extraction of Indian PAN cards.
    
    Extracts:
        - pan: 10-char validated PAN number
        - masked_pan: privacy-masked PAN
        - entity_type: derived statutory entity (Company, Individual, Firm, etc.)
        - cardholder_name: cardholder/legal entity name (multi-line supported)
        - father_name: father's name (where present on individual cards)
        - dob: date of birth / date of incorporation
        - masked_dob: privacy-masked DOB
        - pan_status / name_status / father_name_status / dob_status:
            'VERIFIED', 'PENDING_REVIEW', 'NOT_VERIFIED', or 'NOT_APPLICABLE'
        - lineage: spatial bounding boxes, actual confidence, page number, and extraction method
    """
    lineage: List[Dict[str, Any]] = []

    # 1. Negative Check: Aadhaar Card Rejection with Privacy Redaction
    if check_aadhaar_contamination(raw_text, ocr_items):
        safe_raw_text = mask_aadhaar(raw_text)
        return {
            "document_type": "Aadhaar",
            "aadhaar_detected": True,
            "is_valid_format": False,
            "pan": "NOT DETECTED",
            "pan_number": None,
            "masked_pan": "NOT_AVAILABLE",
            "entity_type": "UNKNOWN",
            "cardholder_name": "NOT DETECTED",
            "name": None,
            "father_name": None,
            "dob": "NOT DETECTED",
            "masked_dob": "NOT_AVAILABLE",
            "pan_status": "REJECTED_AADHAAR",
            "name_status": "REJECTED_AADHAAR",
            "father_name_status": "NOT_APPLICABLE",
            "dob_status": "NOT_VERIFIED",
            "pan_confidence": None,
            "name_confidence": None,
            "father_confidence": None,
            "dob_confidence": None,
            "lineage": [],
            "raw_text": safe_raw_text,
            "page_number": page_number,
            "rejection_reason": "Document detected as Aadhaar card, not a statutory Indian PAN card."
        }

    # 2. Normalize OCR items with Page Awareness
    all_items: List[Dict[str, Any]] = []
    for item in ocr_items:
        txt = item.get("text", "").strip()
        if not txt:
            continue
        bbox = item.get("bbox")
        conf_val = item.get("conf")
        conf = float(conf_val) if conf_val is not None else None
        page = int(item.get("page", page_number))
        y0 = bbox[1] if (bbox and len(bbox) > 1) else 0
        x0 = bbox[0] if (bbox and len(bbox) > 0) else 0
        h = (bbox[3] - bbox[1]) if (bbox and len(bbox) > 3) else 20
        all_items.append({
            "text": txt,
            "bbox": bbox,
            "conf": conf,
            "extraction_method": item.get("extraction_method", "VISION_OCR" if conf is not None else "DIGITAL_TEXT_EXTRACTION"),
            "page": page,
            "y0": y0,
            "x0": x0,
            "h": h
        })

    # Sort all items by Page first, then y0 (top to bottom), then x0 (left to right)
    all_items.sort(key=lambda it: (it["page"], it["y0"], it["x0"]))

    # Detect which page contains the PAN card
    target_page = page_number
    for it in all_items:
        m = re.search(PAN_REGEX, it["text"].upper())
        if m and is_valid_pan(m.group(1)):
            target_page = it["page"]
            break
    else:
        for it in all_items:
            t_up = it["text"].upper()
            if any(k in t_up for k in ["PERMANENT ACCOUNT NUMBER", "स्थायी लेखा संख्या", "आयकर विभाग"]):
                target_page = it["page"]
                break

    # Isolate tokens strictly belonging to the target PAN page (never mix pages)
    sorted_items = [it for it in all_items if it["page"] == target_page]
    if not sorted_items:
        sorted_items = all_items

    # -------------------------------------------------------------
    # 3. Extract PAN Number from Target Page
    # -------------------------------------------------------------
    extracted_pan = "NOT DETECTED"
    pan_conf = None
    pan_bbox = None
    pan_method = "NOT_AVAILABLE"

    for item in sorted_items:
        t_up = item["text"].upper()
        clean_tok = re.sub(r"[^A-Z0-9]", "", t_up)
        m = re.search(PAN_REGEX, t_up)
        if m:
            cand = m.group(1)
            if len(clean_tok) == 10 or "PAN" in t_up or "PERMANENT" in t_up:
                if is_valid_pan(cand):
                    extracted_pan = cand
                    pan_conf = round(item["conf"], 3) if item["conf"] is not None else None
                    pan_bbox = item["bbox"]
                    pan_method = item["extraction_method"]
                    break

    # Raw text fallback (honest: no fabricated confidence or bounding box)
    if extracted_pan == "NOT DETECTED":
        m_raw = re.search(PAN_REGEX, raw_text.upper())
        if m_raw and is_valid_pan(m_raw.group(1)):
            extracted_pan = m_raw.group(1)
            pan_conf = None
            pan_bbox = None
            pan_method = "RAW_TEXT_REGEX"

    if extracted_pan != "NOT DETECTED":
        lineage.append({
            "document": source_doc_name,
            "field": "pan",
            "value": extracted_pan,
            "bbox": pan_bbox,
            "confidence": pan_conf if pan_conf is not None else "NOT_AVAILABLE",
            "extraction_method": pan_method,
            "page_number": target_page
        })

    entity_type = get_pan_entity_type(extracted_pan)
    is_corporate = (extracted_pan != "NOT DETECTED" and len(extracted_pan) == 10 and extracted_pan[3] in ["C", "F", "L", "G", "J"])

    # -------------------------------------------------------------
    # 4. Extract Date of Birth / Incorporation Date
    # -------------------------------------------------------------
    extracted_dob = "NOT DETECTED"
    dob_conf = None
    dob_bbox = None
    dob_method = "NOT_AVAILABLE"

    date_candidates: List[Dict[str, Any]] = []
    for item in sorted_items:
        dm = re.search(DATE_REGEX, item["text"])
        if dm:
            date_val = dm.group(1).replace("-", "/").replace(".", "/")
            date_candidates.append({
                "date": date_val,
                "bbox": item["bbox"],
                "conf": item["conf"],
                "method": item["extraction_method"],
                "y0": item["y0"]
            })

    if date_candidates:
        selected_date = date_candidates[0]
        for dc in date_candidates:
            surrounding_text = " ".join([it["text"].upper() for it in sorted_items if abs(it["y0"] - dc["y0"]) < 60])
            if any(lbl in surrounding_text for lbl in DOB_LABELS):
                selected_date = dc
                break
        extracted_dob = selected_date["date"]
        dob_conf = round(selected_date["conf"], 3) if selected_date["conf"] is not None else None
        dob_bbox = selected_date["bbox"]
        dob_method = selected_date["method"]
        lineage.append({
            "document": source_doc_name,
            "field": "dob",
            "value": extracted_dob,
            "bbox": dob_bbox,
            "confidence": dob_conf if dob_conf is not None else "NOT_AVAILABLE",
            "extraction_method": dob_method,
            "page_number": target_page
        })

    # -------------------------------------------------------------
    # 5. Extract Cardholder Name & Father's Name (Multi-line & Spatial)
    # -------------------------------------------------------------
    cardholder_name = "NOT DETECTED"
    name_conf = None
    name_bbox = None
    name_method = "NOT_AVAILABLE"

    father_name: Optional[str] = None if is_corporate else "NOT DETECTED"
    father_conf = None
    father_bbox = None
    father_method = "NOT_AVAILABLE"

    # Identify vertical anchor boundaries for bilingual labels
    y_father_label = None

    for it in sorted_items:
        t_up = it["text"].upper()
        if is_label_only_line(it["text"]):
            if any(fl in t_up for fl in FATHER_NAME_LABELS) and y_father_label is None:
                y_father_label = it["y0"]

    # Filter candidate name lines: alphabetic text, length >= 3, not boilerplate, not PAN, not Date
    candidate_items: List[Dict[str, Any]] = []
    for item in sorted_items:
        t_raw = item["text"].strip()
        t_up = t_raw.upper()

        if any(h in t_up for h in PAN_BOILERPLATE_HEADERS):
            continue
        if is_label_only_line(t_raw):
            continue
        if re.search(PAN_REGEX, t_up) or re.search(DATE_REGEX, t_up):
            continue

        cleaned_text = strip_label_prefix(t_raw)
        if is_label_only_line(cleaned_text):
            continue

        alpha_only = re.sub(r"[^A-Za-z\s\.\&\-]", "", cleaned_text).strip()
        if len(alpha_only) < 3:
            continue

        words = [w for w in alpha_only.split() if len(w) > 1]
        if not words:
            continue

        candidate_items.append({
            "text": alpha_only,
            "bbox": item["bbox"],
            "conf": item["conf"],
            "method": item["extraction_method"],
            "y0": item["y0"],
            "h": item["h"]
        })

    ambiguous_evidence = False

    if candidate_items:
        if is_corporate:
            # Corporate / Firm PAN cards: combine multi-line legal company name
            corp_candidates = []
            for c in candidate_items:
                corp_candidates.append(c["text"])
                if any(w in c["text"].upper() for w in ["LTD", "LIMITED", "PVT", "LLP", "CORP", "ASSOCIATES", "PARTNERS", "ENTERPRISES"]):
                    break
            cardholder_name = " ".join(corp_candidates) if corp_candidates else candidate_items[0]["text"]
            name_conf = round(candidate_items[0]["conf"], 3) if candidate_items[0]["conf"] is not None else None
            name_bbox = candidate_items[0]["bbox"]
            name_method = candidate_items[0]["method"]
            father_name = None
        else:
            # Individual PAN cards:
            c1_lines = []
            c2_lines = []

            if y_father_label is not None:
                # Label-anchored separation
                for c in candidate_items:
                    if c["y0"] < y_father_label:
                        c1_lines.append(c)
                    else:
                        c2_lines.append(c)
            elif len(candidate_items) == 2:
                # Standard individual card without labels: line 1 is cardholder, line 2 is father
                c1_lines.append(candidate_items[0])
                c2_lines.append(candidate_items[1])
            elif len(candidate_items) > 2:
                # 3 or more candidate lines without labels: multi-line cardholder followed by father
                c1_lines.append(candidate_items[0])
                idx = 1
                while idx < len(candidate_items):
                    prev_c = c1_lines[-1]
                    curr_c = candidate_items[idx]
                    gap = curr_c["y0"] - (prev_c["y0"] + prev_c["h"])
                    if gap <= max(15, int(prev_c["h"] * 0.8)) and len(c1_lines) < 2 and (len(candidate_items) - idx) >= 1:
                        c1_lines.append(curr_c)
                        idx += 1
                    else:
                        break
                while idx < len(candidate_items):
                    c2_lines.append(candidate_items[idx])
                    idx += 1
            else:
                c1_lines = candidate_items

            if c1_lines:
                cardholder_name = " ".join([c["text"] for c in c1_lines])
                c1_confs = [c["conf"] for c in c1_lines if c["conf"] is not None]
                name_conf = round(float(np.mean(c1_confs)), 3) if c1_confs else None
                name_bbox = c1_lines[0]["bbox"]
                name_method = c1_lines[0]["method"]

            if c2_lines:
                father_name = " ".join([c["text"] for c in c2_lines])
                c2_confs = [c["conf"] for c in c2_lines if c["conf"] is not None]
                father_conf = round(float(np.mean(c2_confs)), 3) if c2_confs else None
                father_bbox = c2_lines[0]["bbox"]
                father_method = c2_lines[0]["method"]

            # Fifth-Character Surname Ambiguity Check
            if extracted_pan != "NOT DETECTED" and len(extracted_pan) == 10 and extracted_pan[3] == "P":
                pan_surname_char = extracted_pan[4]
                words_c1 = cardholder_name.split() if cardholder_name != "NOT DETECTED" else []
                surname_c1 = words_c1[-1].upper() if words_c1 else ""

                words_c2 = father_name.split() if (father_name and father_name != "NOT DETECTED") else []
                surname_c2 = words_c2[-1].upper() if words_c2 else ""

                # If primary candidate surname does not match, but secondary candidate matches:
                # Do NOT automatically swap. Return PENDING_REVIEW for officer validation.
                if surname_c1 and not surname_c1.startswith(pan_surname_char):
                    if surname_c2 and surname_c2.startswith(pan_surname_char):
                        ambiguous_evidence = True

    if cardholder_name != "NOT DETECTED":
        lineage.append({
            "document": source_doc_name,
            "field": "entity_name",
            "value": cardholder_name,
            "bbox": name_bbox,
            "confidence": name_conf if name_conf is not None else "NOT_AVAILABLE",
            "extraction_method": name_method,
            "page_number": target_page
        })

    if father_name and father_name != "NOT DETECTED":
        lineage.append({
            "document": source_doc_name,
            "field": "father_name",
            "value": father_name,
            "bbox": father_bbox,
            "confidence": father_conf if father_conf is not None else "NOT_AVAILABLE",
            "extraction_method": father_method,
            "page_number": target_page
        })

    # -------------------------------------------------------------
    # 6. Field Confidence and Verification Status Assignment
    # -------------------------------------------------------------
    MIN_CONFIDENCE = 0.65

    # PAN Status
    pan_status = "NOT_VERIFIED"
    if extracted_pan != "NOT DETECTED":
        if is_valid_pan(extracted_pan):
            if pan_conf is None:
                pan_status = "VERIFIED"
            else:
                pan_status = "VERIFIED" if pan_conf >= MIN_CONFIDENCE else "PENDING_REVIEW"
        else:
            pan_status = "FAIL"

    # Name Status
    name_status = "NOT_VERIFIED"
    if cardholder_name != "NOT DETECTED":
        if ambiguous_evidence:
            name_status = "PENDING_REVIEW"
        elif name_conf is None:
            name_status = "VERIFIED"
        else:
            name_status = "VERIFIED" if name_conf >= MIN_CONFIDENCE else "PENDING_REVIEW"

    # Father Name Status
    if is_corporate:
        father_name_status = "NOT_APPLICABLE"
    elif father_name and father_name != "NOT DETECTED":
        if ambiguous_evidence:
            father_name_status = "PENDING_REVIEW"
        elif father_conf is None:
            father_name_status = "VERIFIED"
        else:
            father_name_status = "VERIFIED" if father_conf >= MIN_CONFIDENCE else "PENDING_REVIEW"
    else:
        father_name_status = "NOT_VERIFIED"

    # DOB Status
    dob_status = "NOT_VERIFIED"
    if extracted_dob != "NOT DETECTED":
        if dob_conf is None:
            dob_status = "VERIFIED"
        else:
            dob_status = "VERIFIED" if dob_conf >= MIN_CONFIDENCE else "PENDING_REVIEW"

    return {
        "document_type": "PAN Card",
        "aadhaar_detected": False,
        "is_valid_format": is_valid_pan(extracted_pan),
        "pan": extracted_pan,
        "pan_number": extracted_pan if extracted_pan != "NOT DETECTED" else None,
        "masked_pan": mask_pan(extracted_pan),
        "entity_type": entity_type,
        "cardholder_name": cardholder_name,
        "name": cardholder_name if cardholder_name != "NOT DETECTED" else None,
        "father_name": father_name,
        "dob": extracted_dob,
        "masked_dob": mask_dob(extracted_dob),
        "pan_status": pan_status,
        "name_status": name_status,
        "father_name_status": father_name_status,
        "dob_status": dob_status,
        "pan_confidence": pan_conf,
        "name_confidence": name_conf,
        "father_confidence": father_conf,
        "dob_confidence": dob_conf,
        "lineage": lineage,
        "raw_text": mask_aadhaar(raw_text),
        "page_number": target_page,
        "ambiguous_evidence": ambiguous_evidence
    }
