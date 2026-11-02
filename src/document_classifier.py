"""Document Type Classifier for BidGuard AI.

Classifies uploaded procurement documents into statutory categories using
content-based text analysis and pattern matching (never relying solely on filenames).
"""

import re
from typing import Dict, Any, List, Optional

# Canonical Document Categories
DOC_TYPE_PAN = "PAN Card"
DOC_TYPE_AADHAAR = "Aadhaar"
DOC_TYPE_GST = "GST Registration Certificate"
DOC_TYPE_BALANCE_SHEET = "Balance Sheet"
DOC_TYPE_ITR = "ITR / ITR-V"
DOC_TYPE_UDYAM = "Udyam Certificate"
DOC_TYPE_UNKNOWN = "Unknown / Other"

ALL_DOCUMENT_TYPES = [
    DOC_TYPE_PAN,
    DOC_TYPE_AADHAAR,
    DOC_TYPE_GST,
    DOC_TYPE_BALANCE_SHEET,
    DOC_TYPE_ITR,
    DOC_TYPE_UDYAM,
    DOC_TYPE_UNKNOWN
]

# Structural Regex Patterns
PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
AADHAAR_REGEX = r"\b[2-9]{1}[0-9]{3}\s[0-9]{4}\s[0-9]{4}\b"
GSTIN_REGEX = r"\b[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b"
UDYAM_REGEX = r"\bUDYAM-[A-Z]{2}-[0-9]{2}-[0-9]{7}\b"
ITR_ACK_REGEX = r"\b[0-9]{15}\b"

# Weighted Keyphrases for Document Classification (English & Hindi/Bilingual)
CLASSIFICATION_RULES = {
    DOC_TYPE_PAN: {
        "strong_phrases": [
            "INCOME TAX DEPARTMENT",
            "आयकर विभाग",
            "PERMANENT ACCOUNT NUMBER",
            "PERMANENT ACCOUNT NUMBER CARD",
            "स्थायी लेखा संख्या कार्ड",
            "स्थायी लेखा संख्या",
            "GOVT. OF INDIA",
            "GOVERNMENT OF INDIA",
            "भारत सरकार"
        ],
        "keywords": [
            "PAN",
            "FATHER'S NAME",
            "FATHERS NAME",
            "पिता का नाम",
            "DATE OF BIRTH",
            "DOB",
            "जन्म तिथि",
            "SIGNATURE",
            "हस्ताक्षर"
        ],
        "regex": PAN_REGEX,
        "weight_strong": 4.5,
        "weight_keyword": 1.5,
        "weight_regex": 3.0,
    },
    DOC_TYPE_AADHAAR: {
        "strong_phrases": [
            "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
            "भारत विशिष्ट पहचान प्राधिकरण",
            "MERA AADHAAR MERI PEHCHAN",
            "मेरा आधार मेरी पहचान",
            "MERA AADHAAR",
            "UIDAI",
            "AADHAAR",
            "आधार",
            "ENROLMENT NO"
        ],
        "keywords": [
            "VID",
            "HELP@UIDAI.GOV.IN",
            "WWW.UIDAI.GOV.IN",
            "YEAR OF BIRTH",
            "जन्म वर्ष",
            "MALE",
            "FEMALE",
            "पुरुष",
            "महिला"
        ],
        "regex": AADHAAR_REGEX,
        "weight_strong": 5.0,
        "weight_keyword": 2.0,
        "weight_regex": 4.0,
    },
    DOC_TYPE_GST: {
        "strong_phrases": [
            "FORM GST REG-06",
            "REGISTRATION CERTIFICATE",
            "GOODS AND SERVICES TAX",
            "CENTRAL GOODS AND SERVICES TAX",
            "STATE GOODS AND SERVICES TAX",
            "TAX INVOICE"
        ],
        "keywords": [
            "GSTIN",
            "LEGAL NAME",
            "TRADE NAME",
            "PRINCIPAL PLACE OF BUSINESS",
            "DATE OF LIABILITY",
            "PERIOD OF VALIDITY",
            "TYPE OF REGISTRATION",
            "JURISDICTION"
        ],
        "regex": GSTIN_REGEX,
        "weight_strong": 4.5,
        "weight_keyword": 1.5,
        "weight_regex": 3.5,
    },
    DOC_TYPE_BALANCE_SHEET: {
        "strong_phrases": [
            "BALANCE SHEET",
            "STATEMENT OF FINANCIAL POSITION",
            "STATEMENT OF PROFIT AND LOSS",
            "PROFIT AND LOSS ACCOUNT",
            "AUDITED FINANCIAL STATEMENTS",
            "INDEPENDENT AUDITOR'S REPORT",
            "CHARTERED ACCOUNTANTS"
        ],
        "keywords": [
            "ANNUAL TURNOVER",
            "EQUITY AND LIABILITIES",
            "SHAREHOLDER'S FUNDS",
            "CURRENT ASSETS",
            "NON-CURRENT ASSETS",
            "TOTAL ASSETS",
            "TOTAL LIABILITIES",
            "UDIN",
            "NOTE NO",
            "FINANCIAL YEAR",
            "AS ON 31ST MARCH",
            "AS AT 31ST MARCH"
        ],
        "regex": None,
        "weight_strong": 4.5,
        "weight_keyword": 1.5,
        "weight_regex": 0.0,
    },
    DOC_TYPE_ITR: {
        "strong_phrases": [
            "INDIAN INCOME TAX RETURN",
            "INCOME TAX RETURN ACKNOWLEDGEMENT",
            "ITR-V",
            "ITR-1",
            "ITR-2",
            "ITR-3",
            "ITR-4",
            "ITR-5",
            "ITR-6",
            "ITR-7",
            "E-FILING"
        ],
        "keywords": [
            "ASSESSMENT YEAR",
            "ACKNOWLEDGEMENT NUMBER",
            "TOTAL INCOME",
            "TAX PAYABLE",
            "GROSS TOTAL INCOME",
            "VERIFICATION FORM"
        ],
        "regex": ITR_ACK_REGEX,
        "weight_strong": 4.0,
        "weight_keyword": 1.5,
        "weight_regex": 2.0,
    },
    DOC_TYPE_UDYAM: {
        "strong_phrases": [
            "UDYAM REGISTRATION CERTIFICATE",
            "UDYAM REGISTRATION",
            "MINISTRY OF MICRO, SMALL & MEDIUM ENTERPRISES",
            "MINISTRY OF MSME",
            "MICRO, SMALL & MEDIUM ENTERPRISES"
        ],
        "keywords": [
            "UDYAM",
            "MSME",
            "NAME OF ENTERPRISE",
            "TYPE OF ENTERPRISE",
            "MAJOR ACTIVITY",
            "SOCIAL CATEGORY",
            "DATE OF INCORPORATION",
            "NATIONAL INDUSTRY CLASSIFICATION"
        ],
        "regex": UDYAM_REGEX,
        "weight_strong": 4.5,
        "weight_keyword": 1.5,
        "weight_regex": 3.5,
    }
}


def classify_document_text(text: str) -> Dict[str, Any]:
    """
    Classifies a document based strictly on its extracted text content.
    Does not examine the filename.
    
    Returns a dict with:
        - document_type: str
        - confidence: float (0.0 to 1.0)
        - scores: Dict[str, float]
        - matched_patterns: List[str]
    """
    if not text or not isinstance(text, str):
        return {
            "document_type": DOC_TYPE_UNKNOWN,
            "confidence": 0.0,
            "scores": {k: 0.0 for k in CLASSIFICATION_RULES},
            "matched_patterns": []
        }

    upper_text = text.upper()
    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {k: [] for k in CLASSIFICATION_RULES}

    for category, rules in CLASSIFICATION_RULES.items():
        score = 0.0
        # Check strong unique phrases
        for phrase in rules["strong_phrases"]:
            if phrase in upper_text or phrase in text:
                score += rules["weight_strong"]
                matched[category].append(phrase)

        # Check secondary keywords
        for kw in rules["keywords"]:
            if kw in upper_text or kw in text:
                score += rules["weight_keyword"]
                matched[category].append(kw)

        # Check regex presence
        reg = rules["regex"]
        if reg and re.search(reg, upper_text):
            score += rules["weight_regex"]
            matched[category].append(f"regex:{reg}")

        scores[category] = score

    # -------------------------------------------------------------
    # Explicit Disambiguation Rules
    # -------------------------------------------------------------
    # 1. Strict Aadhaar Isolation:
    # Aadhaar must NEVER be classified as PAN merely because it has personal details or ID numbers.
    has_aadhaar_markers = (
        scores[DOC_TYPE_AADHAAR] >= 4.0 or
        re.search(AADHAAR_REGEX, upper_text) is not None or
        "UIDAI" in upper_text or
        "MERA AADHAAR" in upper_text or
        "आधार" in text
    )
    if has_aadhaar_markers:
        scores[DOC_TYPE_AADHAAR] += 6.0
        scores[DOC_TYPE_PAN] = max(0.0, scores[DOC_TYPE_PAN] - 6.0)

    # 2. GSTIN Disambiguation:
    # If a 15-char GSTIN and GST headers are present, classify as GST Certificate
    # even though GSTIN contains a 10-char PAN substring.
    if scores[DOC_TYPE_GST] >= 4.0 and re.search(GSTIN_REGEX, upper_text):
        if any(h in upper_text for h in ["FORM GST REG", "GOODS AND SERVICES TAX", "GSTIN"]):
            scores[DOC_TYPE_PAN] = max(0.0, scores[DOC_TYPE_PAN] - 4.0)

    # 3. Udyam Priority:
    if re.search(UDYAM_REGEX, upper_text) or "UDYAM REGISTRATION" in upper_text:
        scores[DOC_TYPE_UDYAM] += 6.0

    # 4. Balance Sheet Priority:
    if "BALANCE SHEET" in upper_text and any(w in upper_text for w in ["EQUITY", "ASSETS", "AUDITOR", "PROFIT", "LOSS"]):
        scores[DOC_TYPE_BALANCE_SHEET] += 4.0

    # Determine best category
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # Threshold for confident assignment
    CONFIDENCE_THRESHOLD = 2.5
    if best_score < CONFIDENCE_THRESHOLD:
        return {
            "document_type": DOC_TYPE_UNKNOWN,
            "confidence": 0.20,
            "scores": scores,
            "matched_patterns": matched.get(best_cat, [])
        }

    # Normalize confidence to [0.50, 0.99]
    norm_conf = min(0.99, round(0.50 + (best_score / (best_score + 5.0)) * 0.49, 3))

    return {
        "document_type": best_cat,
        "confidence": norm_conf,
        "scores": scores,
        "matched_patterns": matched[best_cat]
    }
