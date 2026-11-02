"""Generalization Test Suite & Metric Evaluation for PAN Document Intelligence Engine.

Validates template-agnostic performance across diverse Indian entities, bilingual text,
individual vs. corporate cards, spatial layouts, image preprocessing, and strict
Aadhaar/non-identity negative separation.
Zero hardcoded individual names, sample-specific heuristics, or sensitive documents.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2
import fitz

from src.document_classifier import (
    classify_document_text,
    DOC_TYPE_PAN,
    DOC_TYPE_AADHAAR,
    DOC_TYPE_GST,
    DOC_TYPE_BALANCE_SHEET,
    DOC_TYPE_UNKNOWN
)
from src.pan_extractor import (
    extract_pan_details,
    is_valid_pan,
    get_pan_entity_type,
    mask_pan,
    mask_dob,
    compare_pan_with_gstin
)
from src.pan_preprocessor import DocumentPreprocessor


# =====================================================================
# 1. CORPORATE VS. INDIVIDUAL ENTITY INTELLIGENCE
# =====================================================================

def test_corporate_pan_extraction_and_classification():
    """Corporate entity PAN cards have legal entity names, incorporation dates, and no father's name."""
    ocr_items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 300, 50], "conf": 0.98},
        {"text": "GOVT. OF INDIA", "bbox": [50, 55, 200, 75], "conf": 0.97},
        {"text": "KAVERI INFRASTRUCTURE PRIVATE LIMITED", "bbox": [50, 100, 420, 130], "conf": 0.94},
        {"text": "DATE OF INCORPORATION", "bbox": [50, 140, 240, 160], "conf": 0.92},
        {"text": "18/09/2012", "bbox": [50, 165, 150, 185], "conf": 0.95},
        {"text": "PERMANENT ACCOUNT NUMBER CARD", "bbox": [50, 200, 360, 220], "conf": 0.96},
        {"text": "AAACK1234K", "bbox": [50, 230, 210, 260], "conf": 0.99}
    ]
    raw_text = "\n".join([i["text"] for i in ocr_items])

    # 1. Document classification
    cls_res = classify_document_text(raw_text)
    assert cls_res["document_type"] == DOC_TYPE_PAN

    # 2. Field extraction
    pan_details = extract_pan_details(ocr_items, raw_text)
    assert pan_details["pan"] == "AAACK1234K"
    assert pan_details["is_valid_format"] is True
    assert pan_details["entity_type"] == "Company"
    assert pan_details["cardholder_name"] == "KAVERI INFRASTRUCTURE PRIVATE LIMITED"
    assert pan_details["name_status"] == "VERIFIED"
    assert pan_details["father_name"] is None
    assert pan_details["father_name_status"] == "NOT_APPLICABLE"
    assert pan_details["dob"] == "18/09/2012"
    assert pan_details["masked_pan"] == "XXXXX1234K"
    assert pan_details["masked_dob"] == "**/**/2012"


def test_individual_bilingual_pan_extraction():
    """Individual PAN card with English and Hindi headers, cardholder name, father's name, and DOB."""
    ocr_items = [
        {"text": "आयकर विभाग", "bbox": [40, 20, 160, 45], "conf": 0.96},
        {"text": "INCOME TAX DEPARTMENT", "bbox": [180, 20, 360, 45], "conf": 0.97},
        {"text": "भारत सरकार", "bbox": [40, 50, 150, 70], "conf": 0.95},
        {"text": "GOVT. OF INDIA", "bbox": [170, 50, 290, 70], "conf": 0.96},
        {"text": "नाम / Name", "bbox": [40, 85, 130, 105], "conf": 0.94},
        {"text": "MOHIT VERMA", "bbox": [40, 110, 220, 135], "conf": 0.95},
        {"text": "पिता का नाम / Father's Name", "bbox": [40, 145, 240, 165], "conf": 0.93},
        {"text": "RAMESH VERMA", "bbox": [40, 170, 230, 195], "conf": 0.94},
        {"text": "जन्म तिथि / Date of Birth", "bbox": [40, 205, 230, 225], "conf": 0.92},
        {"text": "05/11/1990", "bbox": [40, 230, 140, 250], "conf": 0.96},
        {"text": "स्थायी लेखा संख्या / PAN", "bbox": [40, 260, 220, 280], "conf": 0.93},
        {"text": "BNZPM8721R", "bbox": [40, 285, 200, 315], "conf": 0.98}
    ]
    raw_text = "\n".join([i["text"] for i in ocr_items])

    cls_res = classify_document_text(raw_text)
    assert cls_res["document_type"] == DOC_TYPE_PAN

    pan_details = extract_pan_details(ocr_items, raw_text)
    assert pan_details["pan"] == "BNZPM8721R"
    assert pan_details["entity_type"] == "Individual / Person"
    assert pan_details["cardholder_name"] == "MOHIT VERMA"
    assert pan_details["father_name"] == "RAMESH VERMA"
    assert pan_details["dob"] == "05/11/1990"
    assert pan_details["masked_pan"] == "XXXXX8721R"
    assert pan_details["masked_dob"] == "**/**/1990"


def test_cardholder_vs_father_spatial_disambiguation():
    """Cardholder name is spatially positioned above father's name on standard cards."""
    ocr_items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [40, 20, 280, 45], "conf": 0.97},
        {"text": "DEEPAK JOSHI", "bbox": [40, 90, 210, 115], "conf": 0.93},
        {"text": "NAVEEN JOSHI", "bbox": [40, 150, 220, 175], "conf": 0.92},
        {"text": "22/04/1984", "bbox": [40, 210, 140, 230], "conf": 0.95},
        {"text": "CJQPD4512N", "bbox": [40, 260, 200, 290], "conf": 0.97}
    ]
    raw_text = "\n".join([i["text"] for i in ocr_items])

    pan_details = extract_pan_details(ocr_items, raw_text)
    assert pan_details["pan"] == "CJQPD4512N"
    assert pan_details["cardholder_name"] == "DEEPAK JOSHI"
    assert pan_details["father_name"] == "NAVEEN JOSHI"


# =====================================================================
# 2. PREPROCESSING & VISION CORRECTION PIPELINE
# =====================================================================

def test_vision_preprocessor_clahe_contrast_enhancement():
    """CLAHE improves contrast in low-light/shadowed document captures."""
    # Create low-contrast synthetic image (grayscale narrow range [100, 130])
    low_contrast = np.random.randint(100, 130, (300, 400, 3), dtype=np.uint8)
    enhanced = DocumentPreprocessor.enhance_contrast_clahe(low_contrast)

    # Standard deviation should increase due to histogram spreading
    assert np.std(enhanced) >= np.std(low_contrast)
    assert enhanced.shape == low_contrast.shape


def test_vision_preprocessor_deskewing():
    """Deskewing correctly detects and rotates skewed document images."""
    # Create an image with horizontal black text bar on white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 180), (500, 220), (0, 0, 0), -1)

    # Artificially rotate image by 5.0 degrees
    center = (300, 200)
    rot_mat = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (600, 400), borderValue=(255, 255, 255))

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    detected_angle = DocumentPreprocessor.estimate_skew_angle(gray)

    # Angle should be non-zero and aligned with rotation
    assert abs(detected_angle) > 0.5

    # Run deskewing
    deskewed = DocumentPreprocessor.deskew_image(rotated)
    assert deskewed is not None
    assert deskewed.shape[0] > 0 and deskewed.shape[1] > 0


def test_pdf_rendering_pipeline():
    """PyMuPDF high-DPI rendering converts multi-page documents to preprocessed numpy images."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create a simple PDF using PyMuPDF
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((50, 50), "INCOME TAX DEPARTMENT", fontsize=16)
        page.insert_text((50, 100), "PERMANENT ACCOUNT NUMBER: AAACK1234K", fontsize=14)
        doc.save(tmp_path)
        doc.close()

        # Render using DocumentPreprocessor at 300 DPI
        rendered_pages = DocumentPreprocessor.render_pdf_to_images(tmp_path, dpi=300)
        assert len(rendered_pages) == 1
        assert rendered_pages[0].shape[0] > 600  # High-DPI rendered height
        assert rendered_pages[0].shape[1] > 800  # High-DPI rendered width
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =====================================================================
# 3. NEGATIVE SEPARATION: AADHAAR & NON-IDENTITY REJECTION
# =====================================================================

def test_aadhaar_card_negative_separation():
    """Aadhaar cards must NEVER be classified as PAN cards or extract 12-digit numbers as PAN."""
    aadhaar_ocr_items = [
        {"text": "भारत सरकार", "bbox": [40, 20, 140, 40], "conf": 0.98},
        {"text": "GOVERNMENT OF INDIA", "bbox": [150, 20, 310, 40], "conf": 0.97},
        {"text": "Unique Identification Authority of India", "bbox": [40, 50, 380, 70], "conf": 0.96},
        {"text": "मेरा आधार मेरी पहचान", "bbox": [40, 80, 220, 100], "conf": 0.95},
        {"text": "SUNIL KUMAR SHARMA", "bbox": [40, 120, 240, 140], "conf": 0.94},
        {"text": "DOB: 14/05/1988", "bbox": [40, 150, 160, 170], "conf": 0.93},
        {"text": "5482 1934 8291", "bbox": [40, 200, 220, 230], "conf": 0.98}
    ]
    raw_text = "\n".join([i["text"] for i in aadhaar_ocr_items])

    # 1. Document classifier must classify as Aadhaar, NOT PAN Card
    cls_res = classify_document_text(raw_text)
    assert cls_res["document_type"] == DOC_TYPE_AADHAAR
    assert cls_res["document_type"] != DOC_TYPE_PAN

    # 2. PAN extractor must reject Aadhaar markers and never extract a 12-digit number as PAN
    pan_details = extract_pan_details(aadhaar_ocr_items, raw_text)
    assert pan_details["pan"] == "NOT DETECTED"
    assert pan_details["pan_status"] == "REJECTED_AADHAAR"
    assert pan_details["cardholder_name"] == "NOT DETECTED"


def test_gst_certificate_negative_separation():
    """GST certificates must be classified as GST and not confused with standalone PAN cards."""
    gst_text = (
        "GOVERNMENT OF INDIA\n"
        "FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number: 27AAPFU0939F1ZV\n"
        "Legal Name: APEX INFOTECH PRIVATE LIMITED\n"
        "Date of Liability: 01/07/2017"
    )
    cls_res = classify_document_text(gst_text)
    assert cls_res["document_type"] == DOC_TYPE_GST
    assert cls_res["document_type"] != DOC_TYPE_PAN


# =====================================================================
# 4. COMPREHENSIVE GENERALIZATION BENCHMARK & METRICS
# =====================================================================

def test_generalization_benchmark_metrics():
    """
    Evaluates system accuracy across a diverse test set of 12 distinct document cases:
    - 4 Individual PAN cards (different names, layouts, bilingual text)
    - 3 Corporate / LLP PAN cards
    - 2 Aadhaar cards (negative separation)
    - 2 GST Certificates (negative separation)
    - 1 Balance Sheet (negative separation)
    
    Computes precision, recall, classification accuracy, and False Acceptance Rate (FAR).
    """
    test_dossiers = [
        # 1. Individual 1
        {
            "category": "individual_pan",
            "text": "INCOME TAX DEPARTMENT GOVT. OF INDIA\nANANYA CHATTERJEE\nSUBHASH CHATTERJEE\n12/08/1992\nPERMANENT ACCOUNT NUMBER CARD\nBZDPC3412E",
            "items": [
                {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 250, 40], "conf": 0.98},
                {"text": "ANANYA CHATTERJEE", "bbox": [50, 80, 260, 105], "conf": 0.96},
                {"text": "SUBHASH CHATTERJEE", "bbox": [50, 130, 270, 155], "conf": 0.95},
                {"text": "12/08/1992", "bbox": [50, 180, 150, 200], "conf": 0.97},
                {"text": "BZDPC3412E", "bbox": [50, 230, 190, 255], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "BZDPC3412E",
            "expected_name": "ANANYA CHATTERJEE",
            "expected_father": "SUBHASH CHATTERJEE",
            "is_pan_card": True
        },
        # 2. Individual 2 (Bilingual)
        {
            "category": "individual_pan",
            "text": "आयकर विभाग INCOME TAX DEPARTMENT\nभारत सरकार GOVT. OF INDIA\nनाम Name VIKRAM SINGHANIA\nपिता का नाम Father's Name HARISH SINGHANIA\nजन्म तिथि Date of Birth 25/03/1986\nस्थायी लेखा संख्या PAN DFGPS9843K",
            "items": [
                {"text": "आयकर विभाग", "bbox": [40, 20, 140, 40], "conf": 0.95},
                {"text": "INCOME TAX DEPARTMENT", "bbox": [150, 20, 320, 40], "conf": 0.97},
                {"text": "VIKRAM SINGHANIA", "bbox": [40, 90, 220, 115], "conf": 0.94},
                {"text": "HARISH SINGHANIA", "bbox": [40, 150, 230, 175], "conf": 0.93},
                {"text": "25/03/1986", "bbox": [40, 200, 140, 220], "conf": 0.96},
                {"text": "DFGPS9843K", "bbox": [40, 250, 180, 275], "conf": 0.98}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "DFGPS9843K",
            "expected_name": "VIKRAM SINGHANIA",
            "expected_father": "HARISH SINGHANIA",
            "is_pan_card": True
        },
        # 3. Individual 3
        {
            "category": "individual_pan",
            "text": "INCOME TAX DEPARTMENT\nPERMANENT ACCOUNT NUMBER CARD\nPRANAV PATEL\nJAYESH PATEL\n04/10/1994\nEFGPP6541L",
            "items": [
                {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 260, 45], "conf": 0.97},
                {"text": "PRANAV PATEL", "bbox": [50, 85, 200, 110], "conf": 0.95},
                {"text": "JAYESH PATEL", "bbox": [50, 140, 200, 165], "conf": 0.94},
                {"text": "04/10/1994", "bbox": [50, 190, 140, 210], "conf": 0.96},
                {"text": "EFGPP6541L", "bbox": [50, 240, 180, 265], "conf": 0.98}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "EFGPP6541L",
            "expected_name": "PRANAV PATEL",
            "expected_father": "JAYESH PATEL",
            "is_pan_card": True
        },
        # 4. Individual 4
        {
            "category": "individual_pan",
            "text": "INCOME TAX DEPARTMENT GOVT. OF INDIA\nRITU SAXENA\nMAHESH SAXENA\n19/01/1989\nKLMPS7892M",
            "items": [
                {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 250, 45], "conf": 0.97},
                {"text": "RITU SAXENA", "bbox": [50, 85, 180, 110], "conf": 0.95},
                {"text": "MAHESH SAXENA", "bbox": [50, 140, 200, 165], "conf": 0.94},
                {"text": "19/01/1989", "bbox": [50, 190, 140, 210], "conf": 0.96},
                {"text": "KLMPS7892M", "bbox": [50, 240, 180, 265], "conf": 0.98}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "KLMPS7892M",
            "expected_name": "RITU SAXENA",
            "expected_father": "MAHESH SAXENA",
            "is_pan_card": True
        },
        # 5. Corporate 1
        {
            "category": "corporate_pan",
            "text": "INCOME TAX DEPARTMENT GOVT. OF INDIA\nBLUESTAR LOGISTICS PRIVATE LIMITED\nDATE OF INCORPORATION 15/06/2015\nPERMANENT ACCOUNT NUMBER CARD\nAAACB4512P",
            "items": [
                {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 260, 45], "conf": 0.98},
                {"text": "BLUESTAR LOGISTICS PRIVATE LIMITED", "bbox": [50, 80, 390, 105], "conf": 0.96},
                {"text": "DATE OF INCORPORATION 15/06/2015", "bbox": [50, 130, 320, 150], "conf": 0.94},
                {"text": "AAACB4512P", "bbox": [50, 190, 180, 215], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "AAACB4512P",
            "expected_name": "BLUESTAR LOGISTICS PRIVATE LIMITED",
            "expected_father": None,
            "is_pan_card": True
        },
        # 6. Corporate 2 (Firm / LLP)
        {
            "category": "firm_pan",
            "text": "INCOME TAX DEPARTMENT GOVT. OF INDIA\nZENITH ENGINEERING PARTNERS LLP\nDATE OF INCORPORATION 28/02/2019\nAAAFZ3214Q",
            "items": [
                {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 260, 45], "conf": 0.98},
                {"text": "ZENITH ENGINEERING PARTNERS LLP", "bbox": [50, 80, 380, 105], "conf": 0.95},
                {"text": "28/02/2019", "bbox": [50, 130, 150, 150], "conf": 0.94},
                {"text": "AAAFZ3214Q", "bbox": [50, 180, 180, 205], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "AAAFZ3214Q",
            "expected_name": "ZENITH ENGINEERING PARTNERS LLP",
            "expected_father": None,
            "is_pan_card": True
        },
        # 7. Corporate 3
        {
            "category": "corporate_pan",
            "text": "INCOME TAX DEPARTMENT GOVT. OF INDIA\nNARMADA RENEWABLES PRIVATE LIMITED\n10/11/2021\nAAACN8912T",
            "items": [
                {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 260, 45], "conf": 0.98},
                {"text": "NARMADA RENEWABLES PRIVATE LIMITED", "bbox": [50, 80, 400, 105], "conf": 0.96},
                {"text": "10/11/2021", "bbox": [50, 130, 150, 150], "conf": 0.95},
                {"text": "AAACN8912T", "bbox": [50, 180, 180, 205], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_PAN,
            "expected_pan": "AAACN8912T",
            "expected_name": "NARMADA RENEWABLES PRIVATE LIMITED",
            "expected_father": None,
            "is_pan_card": True
        },
        # 8. Aadhaar Card 1 (Negative Separation)
        {
            "category": "aadhaar_negative",
            "text": "Unique Identification Authority of India UIDAI\nमेरा आधार मेरी पहचान\nROHIT KASHYAP\nDOB 11/04/1993\n4819 2381 9012",
            "items": [
                {"text": "Unique Identification Authority of India", "bbox": [40, 20, 360, 40], "conf": 0.97},
                {"text": "मेरा आधार मेरी पहचान", "bbox": [40, 50, 200, 70], "conf": 0.95},
                {"text": "ROHIT KASHYAP", "bbox": [40, 90, 180, 110], "conf": 0.94},
                {"text": "DOB 11/04/1993", "bbox": [40, 130, 160, 150], "conf": 0.96},
                {"text": "4819 2381 9012", "bbox": [40, 180, 200, 205], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_AADHAAR,
            "expected_pan": "NOT DETECTED",
            "is_pan_card": False
        },
        # 9. Aadhaar Card 2 (Negative Separation)
        {
            "category": "aadhaar_negative",
            "text": "GOVERNMENT OF INDIA UIDAI\nMEENA AGRAWAL\nYEAR OF BIRTH 1987\n9281 4812 7391",
            "items": [
                {"text": "GOVERNMENT OF INDIA UIDAI", "bbox": [40, 20, 280, 40], "conf": 0.98},
                {"text": "MEENA AGRAWAL", "bbox": [40, 70, 190, 95], "conf": 0.95},
                {"text": "9281 4812 7391", "bbox": [40, 130, 200, 155], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_AADHAAR,
            "expected_pan": "NOT DETECTED",
            "is_pan_card": False
        },
        # 10. GST Certificate 1 (Negative Separation)
        {
            "category": "gst_negative",
            "text": "GOVERNMENT OF INDIA FORM GST REG-06\nREGISTRATION CERTIFICATE\nGSTIN: 07AAACK1234K1Z2\nLegal Name: KAVERI INFRASTRUCTURE PRIVATE LIMITED\nIssue Date: 01/07/2017",
            "items": [
                {"text": "FORM GST REG-06", "bbox": [50, 20, 220, 40], "conf": 0.98},
                {"text": "REGISTRATION CERTIFICATE", "bbox": [50, 50, 280, 70], "conf": 0.97},
                {"text": "07AAACK1234K1Z2", "bbox": [50, 90, 220, 115], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_GST,
            "expected_pan": "NOT DETECTED",
            "is_pan_card": False
        },
        # 11. GST Certificate 2 (Negative Separation)
        {
            "category": "gst_negative",
            "text": "GOODS AND SERVICES TAX REGISTRATION\nGSTIN: 27AAPFU0939F1ZV\nSTATE OF MAHARASHTRA",
            "items": [
                {"text": "GOODS AND SERVICES TAX REGISTRATION", "bbox": [50, 20, 340, 40], "conf": 0.97},
                {"text": "27AAPFU0939F1ZV", "bbox": [50, 70, 220, 95], "conf": 0.99}
            ],
            "expected_type": DOC_TYPE_GST,
            "expected_pan": "NOT DETECTED",
            "is_pan_card": False
        },
        # 12. Balance Sheet (Negative Separation)
        {
            "category": "balance_sheet_negative",
            "text": "INDEPENDENT AUDITOR'S REPORT\nBALANCE SHEET AS OF 31ST MARCH 2024\nSTATEMENT OF PROFIT AND LOSS\nTOTAL REVENUE FROM OPERATIONS: 15000000",
            "items": [
                {"text": "INDEPENDENT AUDITOR'S REPORT", "bbox": [50, 20, 320, 45], "conf": 0.98},
                {"text": "BALANCE SHEET AS OF 31ST MARCH 2024", "bbox": [50, 60, 360, 85], "conf": 0.97}
            ],
            "expected_type": DOC_TYPE_BALANCE_SHEET,
            "expected_pan": "NOT DETECTED",
            "is_pan_card": False
        }
    ]

    total_cases = len(test_dossiers)
    correct_classification = 0
    pan_true_positives = 0
    pan_false_positives = 0
    pan_true_negatives = 0
    pan_false_negatives = 0
    name_correct = 0
    father_correct = 0

    pan_card_count = sum(1 for d in test_dossiers if d["is_pan_card"])
    non_pan_count = total_cases - pan_card_count

    for dossier in test_dossiers:
        # 1. Classification
        cls_res = classify_document_text(dossier["text"])
        if cls_res["document_type"] == dossier["expected_type"]:
            correct_classification += 1

        # 2. PAN Field Extraction
        pan_details = extract_pan_details(dossier["items"], dossier["text"])
        detected_pan = pan_details["pan"]

        if dossier["is_pan_card"]:
            if detected_pan == dossier["expected_pan"]:
                pan_true_positives += 1
            else:
                pan_false_negatives += 1

            if pan_details["cardholder_name"] == dossier["expected_name"]:
                name_correct += 1

            if pan_details.get("father_name") == dossier.get("expected_father"):
                father_correct += 1
        else:
            if detected_pan != "NOT DETECTED":
                pan_false_positives += 1
            else:
                pan_true_negatives += 1

    # Metric Calculations
    classification_accuracy = (correct_classification / total_cases) * 100.0
    pan_precision = (pan_true_positives / (pan_true_positives + pan_false_positives)) * 100.0 if (pan_true_positives + pan_false_positives) > 0 else 0.0
    pan_recall = (pan_true_positives / (pan_true_positives + pan_false_negatives)) * 100.0 if (pan_true_positives + pan_false_negatives) > 0 else 0.0
    name_accuracy = (name_correct / pan_card_count) * 100.0
    father_accuracy = (father_correct / pan_card_count) * 100.0
    false_acceptance_rate = (pan_false_positives / non_pan_count) * 100.0

    print("\n" + "=" * 65)
    print("  PAN CARD INTELLIGENCE ENGINE: GENERALIZATION BENCHMARK REPORT")
    print("=" * 65)
    print(f"Total Test Cases Evaluated : {total_cases}")
    print(f"  - PAN Cards (Indiv & Corp): {pan_card_count}")
    print(f"  - Non-PAN Negative Samples: {non_pan_count}")
    print("-" * 65)
    print(f"Document Classification Acc: {classification_accuracy:.1f}%")
    print(f"PAN Extraction Precision   : {pan_precision:.1f}%")
    print(f"PAN Extraction Recall      : {pan_recall:.1f}%")
    print(f"Cardholder Name Accuracy   : {name_accuracy:.1f}%")
    print(f"Father's Name Accuracy     : {father_accuracy:.1f}%")
    print(f"False Acceptance Rate (FAR): {false_acceptance_rate:.1f}%")
    print("=" * 65)

    # Statutory benchmarks
    assert classification_accuracy >= 90.0, f"Classification accuracy {classification_accuracy}% below 90%"
    assert pan_precision == 100.0, f"PAN precision {pan_precision}% must be 100% (zero false positives)"
    assert pan_recall == 100.0, f"PAN recall {pan_recall}% must be 100%"
    assert name_accuracy >= 85.0, f"Name accuracy {name_accuracy}% below 85%"
    assert father_accuracy >= 85.0, f"Father name accuracy {father_accuracy}% below 85%"
    assert false_acceptance_rate == 0.0, f"FAR {false_acceptance_rate}% must be strictly 0.0%"
