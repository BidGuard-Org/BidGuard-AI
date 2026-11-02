"""
Phase 2.2 Reliability and Privacy Fixes Test Suite
Covers:
1. Aadhaar Privacy (masking in text, raw_text, staging queue, and extractor output)
2. Page-Aware OCR token ordering and cross-page isolation
3. Hybrid PDF page-by-page handling and safe resource management (doc.close)
4. Confidence Integrity (honest None/NOT_AVAILABLE reporting, no fabricated numbers)
5. Multi-line name extraction and ambiguous surname PENDING_REVIEW behavior
6. Multiple PAN and GST document support with primary matching and partner logging
7. Image orientation and statutory 90/180/270 rotations
"""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image
import fitz  # PyMuPDF

from src.pan_extractor import (
    mask_aadhaar,
    check_aadhaar_contamination,
    extract_pan_details,
    compare_pan_with_gstin,
    is_valid_pan,
    get_pan_entity_type
)
from src.pan_preprocessor import (
    DocumentPreprocessor,
    load_image_with_exif,
    rotate_image,
    render_pdf_page_to_image,
    render_pdf_to_images
)
from src.upload_workflow import (
    DocumentStagingQueue,
    DOC_TYPE_PAN,
    DOC_TYPE_GST
)


def test_aadhaar_privacy_masking_all_formats():
    """Verify 12-digit Aadhaar numbers in spaced, hyphenated, and continuous formats are masked."""
    # 1. Spaced format: 9876 5432 1098 -> XXXX-XXXX-1098
    spaced = "My Aadhaar Number is 9876 5432 1098 issued by UIDAI"
    masked_spaced = mask_aadhaar(spaced)
    assert "9876 5432 1098" not in masked_spaced
    assert "XXXX-XXXX-1098" in masked_spaced

    # 2. Hyphenated format: 9876-5432-1098 -> XXXX-XXXX-1098
    hyphen = "Aadhaar Card: 9876-5432-1098 Government of India"
    masked_hyphen = mask_aadhaar(hyphen)
    assert "9876-5432-1098" not in masked_hyphen
    assert "XXXX-XXXX-1098" in masked_hyphen

    # 3. Continuous format: 987654321098 -> XXXXXXXX1098
    continuous = "Ref UID: 987654321098"
    masked_cont = mask_aadhaar(continuous)
    assert "987654321098" not in masked_cont
    assert "XXXXXXXX1098" in masked_cont

    # 4. 10-digit PAN numbers must NOT be accidentally masked as Aadhaar
    pan_str = "Income Tax PAN: ABCDE1234F"
    assert mask_aadhaar(pan_str) == pan_str


def test_aadhaar_contamination_redacts_all_sensitive_data_in_pan_extractor():
    """Verify Aadhaar document rejection redacts sensitive data and returns zero unmasked Aadhaar."""
    raw_aadhaar_doc = "GOVERNMENT OF INDIA UNIQUE IDENTIFICATION AUTHORITY OF INDIA 9876 5432 1098 MERA AADHAAR"
    
    assert check_aadhaar_contamination(raw_aadhaar_doc, []) is True

    result = extract_pan_details(
        raw_text=raw_aadhaar_doc,
        ocr_items=[{"text": "9876 5432 1098", "bbox": [10, 10, 100, 30], "conf": 0.98, "page": 1}],
        source_doc_name="aadhaar_sample.jpg",
        page_number=1
    )

    assert result["aadhaar_detected"] is True
    assert result["pan_number"] is None
    assert result["name"] is None
    assert result["name_status"] == "REJECTED_AADHAAR"
    assert result["pan_confidence"] is None

    # Sensitive 12-digit number must never appear in raw_text or any returned field
    assert "9876 5432 1098" not in result["raw_text"]
    assert "XXXX-XXXX-1098" in result["raw_text"]
    for k, v in result.items():
        if isinstance(v, str):
            assert "9876 5432 1098" not in v


def test_page_aware_ocr_token_ordering_and_isolation():
    """Verify OCR tokens are isolated strictly by target page and tokens from Page 1 are not mixed into Page 2."""
    ocr_items = [
        # Page 1: Decoy document / cover letter
        {"text": "PURCHASE ORDER INVOICE", "bbox": [20, 30, 200, 50], "conf": 0.97, "page": 1},
        {"text": "DECOY VENDOR NAME", "bbox": [20, 60, 200, 80], "conf": 0.95, "page": 1},
        # Page 2: Statutory PAN Card
        {"text": "INCOME TAX DEPARTMENT", "bbox": [30, 40, 220, 60], "conf": 0.96, "page": 2},
        {"text": "ANURAG SHUKLA", "bbox": [30, 80, 180, 100], "conf": 0.95, "page": 2},
        {"text": "PERMANENT ACCOUNT NUMBER", "bbox": [30, 120, 260, 140], "conf": 0.97, "page": 2},
        {"text": "ABFPS1234E", "bbox": [30, 150, 160, 170], "conf": 0.98, "page": 2},
        {"text": "01/01/1990", "bbox": [30, 180, 120, 200], "conf": 0.96, "page": 2}
    ]

    result = extract_pan_details(
        raw_text="",
        ocr_items=ocr_items,
        source_doc_name="dossier.pdf",
        page_number=1
    )

    # Must target Page 2 where the PAN exists
    assert result["pan_number"] == "ABFPS1234E"
    assert result["name"] == "ANURAG SHUKLA"
    assert "DECOY" not in result["name"]
    assert result["page_number"] == 2

    # All lineage evidence items must record page_number == 2
    for entry in result["lineage"]:
        assert entry["page_number"] == 2


def test_hybrid_pdf_page_by_page_inspection_and_handle_cleanup():
    """Verify PyMuPDF document handle cleanup without file locks on Windows."""
    # Create a synthetic 2-page PDF
    temp_dir = tempfile.mkdtemp()
    temp_pdf = os.path.join(temp_dir, "test_hybrid.pdf")
    try:
        doc = fitz.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 100), "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER ABCDE1234F")
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 100), "Scanned Page Placeholder")
        doc.save(temp_pdf)
        doc.close()

        # Render Page 2 individually
        img = render_pdf_page_to_image(temp_pdf, page_number=1, dpi=150)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

        # Render all pages
        all_imgs = render_pdf_to_images(temp_pdf, dpi=150)
        assert len(all_imgs) == 2

        # Verify safe handle closure: file can be deleted immediately on Windows
        os.remove(temp_pdf)
        assert not os.path.exists(temp_pdf)
    finally:
        if os.path.exists(temp_pdf):
            try:
                os.remove(temp_pdf)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def test_confidence_integrity_honest_reporting_no_fabrication():
    """Verify no fabricated confidence (0.85/0.98) or dummy bbox [0,0,10,10] are returned."""
    # Pure raw text extraction (digital text without vision OCR bounding boxes)
    result = extract_pan_details(
        raw_text="INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER ABCDE1234F",
        ocr_items=[],
        source_doc_name="digital_pan.pdf",
        page_number=1
    )

    assert result["pan_number"] == "ABCDE1234F"
    assert result["pan_confidence"] is None  # NOT 0.85 or 0.98
    assert result["pan_status"] == "VERIFIED"

    # Lineage must honestly denote extraction method and NOT_AVAILABLE confidence
    pan_lineage = next(l for l in result["lineage"] if l["field"] == "pan")
    assert pan_lineage["confidence"] == "NOT_AVAILABLE"
    assert pan_lineage["bbox"] is None  # NOT [0, 0, 10, 10]
    assert pan_lineage["extraction_method"] == "RAW_TEXT_REGEX"


def test_pan_multiline_name_extraction():
    """Verify multi-line cardholder names are combined properly."""
    ocr_items = [
        {"text": "NAME", "bbox": [50, 50, 100, 70], "conf": 0.95, "page": 1},
        {"text": "MOHAMMED", "bbox": [50, 75, 150, 95], "conf": 0.94, "page": 1},
        {"text": "SHOAIB AKHTAR", "bbox": [50, 98, 200, 118], "conf": 0.92, "page": 1},
        {"text": "FATHER'S NAME", "bbox": [50, 135, 180, 155], "conf": 0.95, "page": 1},
        {"text": "SHAHID AKHTAR", "bbox": [50, 160, 180, 180], "conf": 0.93, "page": 1},
        {"text": "PERMANENT ACCOUNT NUMBER", "bbox": [50, 200, 260, 220], "conf": 0.96, "page": 1},
        {"text": "ABFPS9876M", "bbox": [50, 230, 150, 250], "conf": 0.97, "page": 1}
    ]

    result = extract_pan_details(
        raw_text="",
        ocr_items=ocr_items,
        source_doc_name="pan_card.jpg",
        page_number=1
    )

    assert result["name"] == "MOHAMMED SHOAIB AKHTAR"
    assert result["father_name"] == "SHAHID AKHTAR"
    assert result["name_status"] == "VERIFIED"


def test_ambiguous_surname_flags_pending_review_without_swapping():
    """Verify candidate surname mismatch flags PENDING_REVIEW without blind swapping."""
    # PAN ABCPK1234F: 5th character 'K'. Candidate 1 has surname 'VERMA' (starts with 'V').
    # Candidate 2 has surname 'KUMAR' (starts with 'K').
    ocr_items = [
        {"text": "NAME", "bbox": [50, 50, 100, 70], "conf": 0.95, "page": 1},
        {"text": "ROHIT VERMA", "bbox": [50, 75, 180, 95], "conf": 0.95, "page": 1},
        {"text": "FATHER'S NAME", "bbox": [50, 110, 180, 130], "conf": 0.95, "page": 1},
        {"text": "KISHORE KUMAR", "bbox": [50, 135, 180, 155], "conf": 0.95, "page": 1},
        {"text": "PERMANENT ACCOUNT NUMBER", "bbox": [50, 170, 250, 190], "conf": 0.95, "page": 1},
        {"text": "ABCPK1234F", "bbox": [50, 200, 150, 220], "conf": 0.95, "page": 1}
    ]

    result = extract_pan_details(
        raw_text="",
        ocr_items=ocr_items,
        source_doc_name="pan_card.jpg",
        page_number=1
    )

    # Must NOT blindly swap candidate 2 into cardholder name
    assert result["name"] == "ROHIT VERMA"
    assert result["father_name"] == "KISHORE KUMAR"
    # Ambiguous evidence must flag PENDING_REVIEW
    assert result["name_status"] == "PENDING_REVIEW"
    assert result["father_name_status"] == "PENDING_REVIEW"


def test_multiple_pan_documents_and_gst_matching():
    """Verify multiple PANs are processed separately, primary matches GSTIN, and secondary is logged."""
    queue = DocumentStagingQueue()

    # 1. GST doc with embedded PAN AAPFU0939F
    gst_doc = queue.add_file("gst.pdf", b"gst_dummy_content")
    gst_doc.detected_type = DOC_TYPE_GST
    gst_doc.extracted_fields = {
        "gstin": "27AAPFU0939F1ZV",
        "entity_name": "APEX INFOTECH PVT LTD",
        "turnover": 120.0,
        "turnover_str": "Rs. 120.0 Lakhs",
        "issue_date": "15/05/2024"
    }
    gst_doc.status = "Verified"

    # 2. Secondary / Partner PAN (does not match GSTIN embedded PAN)
    pan_partner = queue.add_file("pan_partner.jpg", b"partner_pan")
    pan_partner.detected_type = DOC_TYPE_PAN
    pan_partner.extracted_fields = {
        "pan": "BBLPK5678G",
        "pan_status": "VERIFIED",
        "cardholder_name": "PRIYA KAPOOR",
        "name_status": "VERIFIED",
        "entity_type": "Individual / Person"
    }
    pan_partner.status = "Verified"

    # 3. Primary Entity PAN (matches GSTIN embedded PAN)
    pan_company = queue.add_file("pan_company.jpg", b"company_pan")
    pan_company.detected_type = DOC_TYPE_PAN
    pan_company.extracted_fields = {
        "pan": "AAPFU0939F",
        "pan_status": "VERIFIED",
        "cardholder_name": "APEX INFOTECH PVT LTD",
        "name_status": "VERIFIED",
        "entity_type": "Company"
    }
    pan_company.status = "Verified"

    results = queue.execute_consolidated_verification(ocr_reader=None, min_turnover_req=50.0)

    # Primary PAN was correctly identified and matched against GSTIN
    pan_gst_check = next(a for a in results["audit_data"] if a["Rule Checked"] == "PAN Embedded in GSTIN Check")
    assert pan_gst_check["Status"] == "PASS"

    # Secondary PAN was logged in the audit trail without causing a FAIL
    sec_audit = next(a for a in results["audit_data"] if "Secondary / Partner PAN" in a["Rule Checked"])
    assert sec_audit["Status"] == "PASS"
    assert "PRIYA KAPOOR" in sec_audit["Finding Details"]

    # Slots contain multi-PAN counts and extractions
    slots = results["slots"]
    assert slots["multi_pan_count"] == 2
    assert len(slots["all_pan_extractions"]) == 2
    assert results["discrepancies_count"] == 0


def test_orientation_and_statutory_rotations():
    """Verify 90, 180, and 270 degree statutory rotations and EXIF loading."""
    # Create a non-square test image: height 100, width 200
    img = np.zeros((100, 200, 3), dtype=np.uint8)

    rot90 = rotate_image(img, 90)
    assert rot90.shape[0] == 200 and rot90.shape[1] == 100

    rot180 = rotate_image(img, 180)
    assert rot180.shape[0] == 100 and rot180.shape[1] == 200

    rot270 = rotate_image(img, 270)
    assert rot270.shape[0] == 200 and rot270.shape[1] == 100

    rot0 = rotate_image(img, 0)
    assert rot0.shape[0] == 100 and rot0.shape[1] == 200

    # Test load_image_with_exif with PIL Image
    pil_img = Image.new("RGB", (150, 80), color=(255, 255, 255))
    loaded_np = load_image_with_exif(pil_img)
    assert isinstance(loaded_np, np.ndarray)
    assert loaded_np.shape == (80, 150, 3)
