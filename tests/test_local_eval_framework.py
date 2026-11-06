"""Automated Test Suite for Local Real-World Document Evaluation Framework.

Validates the test harness, evaluator, runner, report exporter, and privacy
protections using SYNTHETIC IN-MEMORY FIXTURES ONLY.
NOTE: Passing these synthetic tests verifies test harness functionality,
not real-world accuracy across diverse physical paper documents.
"""

import os
import json
import tempfile
import pytest
import fitz  # PyMuPDF

from local_eval.privacy import (
    mask_name,
    mask_pan,
    mask_dob,
    mask_aadhaar,
    mask_gstin,
    mask_udyam,
    hash_identifier,
    contains_unmasked_aadhaar,
    sanitize_message
)
from local_eval.evaluator import DocumentEvaluator
from local_eval.runner import EvaluationRunner, FOLDER_CATEGORY_MAP
from src.document_classifier import (
    DOC_TYPE_PAN,
    DOC_TYPE_AADHAAR,
    DOC_TYPE_GST,
    DOC_TYPE_UDYAM
)


# =====================================================================
# 1. PRIVACY PROTECTION & IDENTIFIER MASKING TESTS
# =====================================================================

def test_privacy_name_masking():
    """Verify person and legal names are masked to initial and asterisks."""
    assert mask_name("Mohit Verma") == "M*** V***"
    assert mask_name("Apex Infotech Private Limited") == "A*** I*** P*** L***"
    assert mask_name("R") == "R"
    assert mask_name("") == "NOT_DETECTED"
    assert mask_name(None) == "NOT_DETECTED"
    assert mask_name("NOT DETECTED") == "NOT_DETECTED"
    assert mask_name("UNKNOWN") == "NOT_DETECTED"


def test_privacy_pan_masking():
    """Verify PANs are masked to XXXXX1234F and placeholders handled cleanly."""
    assert mask_pan("ABCDE1234F") == "XXXXX1234F"
    assert mask_pan("NOT DETECTED") == "NOT_AVAILABLE"
    assert mask_pan("INVALID") == "NOT_AVAILABLE"
    assert mask_pan("") == "NOT_AVAILABLE"
    assert mask_pan(None) == "NOT_AVAILABLE"


def test_privacy_dob_masking():
    """Verify DOBs are masked to **/**/YYYY and placeholders handled cleanly."""
    assert mask_dob("05/11/1990") == "**/**/1990"
    assert mask_dob("15-08-1985") == "**/**/1985"
    assert mask_dob("NOT DETECTED") == "NOT_AVAILABLE"
    assert mask_dob("INVALID") == "NOT_AVAILABLE"
    assert mask_dob("") == "NOT_AVAILABLE"
    assert mask_dob(None) == "NOT_AVAILABLE"


def test_privacy_gstin_masking():
    """Verify GSTINs have embedded PAN and check digit masked."""
    masked = mask_gstin("27AAPFU0939F1ZV")
    assert masked == "27XXXXX0939F1Z*"
    assert "AAPFU" not in masked
    assert mask_gstin("INVALID") == "NOT_AVAILABLE"
    assert mask_gstin("NOT DETECTED") == "NOT_AVAILABLE"
    assert mask_gstin(None) == "NOT_AVAILABLE"


def test_privacy_udyam_masking():
    """Verify Udyam numbers have middle serial numbers redacted."""
    masked = mask_udyam("UDYAM-MH-01-0012345")
    assert masked == "UDYAM-MH-XX-XXX2345"
    assert "0012345" not in masked
    assert mask_udyam("INVALID") == "NOT_AVAILABLE"
    assert mask_udyam("NOT DETECTED") == "NOT_AVAILABLE"
    assert mask_udyam(None) == "NOT_AVAILABLE"


def test_privacy_aadhaar_leak_detector():
    """Verify unmasked 12-digit Aadhaar detector catches spaced and continuous numbers."""
    # Unmasked forms must be detected
    assert contains_unmasked_aadhaar("Customer UID: 9876 5432 1098 valid") is True
    assert contains_unmasked_aadhaar("Customer UID: 9876-5432-1098 valid") is True
    assert contains_unmasked_aadhaar("Customer UID: 987654321098 valid") is True

    # Masked forms must NOT trigger leak detection
    assert contains_unmasked_aadhaar("Customer UID: XXXX-XXXX-1098 valid") is False
    assert contains_unmasked_aadhaar("Customer UID: XXXXXXXX1098 valid") is False

    # PAN and GSTIN must NOT trigger false positive
    assert contains_unmasked_aadhaar("PAN: ABCDE1234F GSTIN: 27AAPFU0939F1ZV") is False


def test_privacy_sanitize_tracebacks_and_errors():
    """Verify sensitive PAN, Aadhaar, GSTIN, Udyam, and dates are scrubbed from error messages."""
    raw_error = (
        "Exception in doc with PAN ABCDE1234F, Aadhaar 9876 5432 1098, "
        "GSTIN 27AAPFU0939F1ZV, Udyam UDYAM-MH-01-0012345, DOB 05/11/1990"
    )
    sanitized = sanitize_message(raw_error)
    assert "ABCDE1234F" not in sanitized
    assert "9876 5432 1098" not in sanitized
    assert "27AAPFU0939F1ZV" not in sanitized
    assert "0012345" not in sanitized
    assert "05/11/1990" not in sanitized
    assert "XXXXX1234F" in sanitized
    assert "XXXX-XXXX-1098" in sanitized
    assert "27XXXXX0939F1Z*" in sanitized
    assert "UDYAM-MH-XX-XXX2345" in sanitized
    assert "**/**/1990" in sanitized


# =====================================================================
# 2. DOCUMENT EVALUATOR TESTS (SYNTHETIC FIXTURES)
# =====================================================================

def test_evaluator_synthetic_digital_pan_pdf():
    """Evaluate synthetic digital PAN PDF without storing unmasked PII."""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "synthetic_pan.pdf")
    try:
        # Create synthetic digital PAN PDF
        doc = fitz.open()
        page = doc.new_page(width=595, height=400)
        page.insert_text(
            (50, 80),
            "INCOME TAX DEPARTMENT GOVT. OF INDIA\n"
            "PERMANENT ACCOUNT NUMBER CARD\n"
            "ABCDE1234F\n"
            "MOHIT VERMA\n"
            "RAMESH VERMA\n"
            "05/11/1990"
        )
        doc.save(pdf_path)
        doc.close()

        evaluator = DocumentEvaluator(temp_dir=os.path.join(temp_dir, "tmp"))
        gt = {
            "expected_category": DOC_TYPE_PAN,
            "expected_pan": "ABCDE1234F",
            "expected_name": "MOHIT VERMA"
        }

        res = evaluator.evaluate_file(pdf_path, ground_truth=gt, ocr_reader=None)

        # Classification check
        assert res["classification"]["detected_type"] == DOC_TYPE_PAN
        assert res["classification"]["match"] is True

        # Field evaluation checks (all masked)
        pan_eval = res["evaluations"]["pan"]
        assert pan_eval["match"] is True
        assert pan_eval["valid_syntax"] is True
        assert pan_eval["extracted_masked"] == "XXXXX1234F"

        # Privacy check
        assert res["privacy"]["masking_effective"] is True
        assert res["privacy"]["aadhaar_leak_detected"] is False
    finally:
        if os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError: pass


def test_evaluator_synthetic_aadhaar_rejection():
    """Evaluate synthetic Aadhaar document and verify rejection + zero leak."""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "synthetic_aadhaar.pdf")
    try:
        doc = fitz.open()
        page = doc.new_page(width=595, height=400)
        page.insert_text(
            (50, 80),
            "GOVERNMENT OF INDIA\n"
            "UNIQUE IDENTIFICATION AUTHORITY OF INDIA UIDAI\n"
            "MERA AADHAAR MERI PEHCHAN\n"
            "ROHIT KASHYAP\n"
            "9876 5432 1098"
        )
        doc.save(pdf_path)
        doc.close()

        evaluator = DocumentEvaluator(temp_dir=os.path.join(temp_dir, "tmp"))
        gt = {
            "expected_category": DOC_TYPE_AADHAAR,
            "expect_aadhaar_rejected": True
        }

        res = evaluator.evaluate_file(pdf_path, ground_truth=gt, ocr_reader=None)

        assert res["classification"]["detected_type"] == DOC_TYPE_AADHAAR
        assert res["evaluations"]["aadhaar"]["classified_as_aadhaar"] is True
        assert res["evaluations"]["aadhaar"]["rejection_effective"] is True
        assert res["privacy"]["masking_effective"] is True
        assert res["status"] == "SUCCESS"
    finally:
        if os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError: pass


def test_evaluator_missing_expected_values_not_evaluated():
    """Verify missing expected ground truth is reported as NOT_EVALUATED, never counted as SUCCESS."""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "unannotated_pan.pdf")
    try:
        doc = fitz.open()
        page = doc.new_page(width=595, height=400)
        page.insert_text(
            (50, 80),
            "INCOME TAX DEPARTMENT GOVT. OF INDIA\n"
            "PERMANENT ACCOUNT NUMBER CARD\n"
            "ABCDE1234F\n"
            "MOHIT VERMA"
        )
        doc.save(pdf_path)
        doc.close()

        evaluator = DocumentEvaluator(temp_dir=os.path.join(temp_dir, "tmp"))

        # Empty ground truth (no expected category, no expected fields)
        res = evaluator.evaluate_file(pdf_path, ground_truth={}, ocr_reader=None)

        assert res["status"] == "NOT_EVALUATED"
        assert res["classification"]["match"] == "NOT_EVALUATED"
        assert res["classification"]["expected_type"] == "NOT_SPECIFIED"
        assert res["evaluations"]["pan"]["match"] == "NOT_EVALUATED"
        assert res["evaluations"]["name"]["match"] == "NOT_EVALUATED"
    finally:
        if os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError: pass


def test_evaluator_incorrect_extracted_values_are_failures():
    """Verify incorrect extracted values count as FAILURES with appropriate error categories."""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "mismatched_pan.pdf")
    try:
        doc = fitz.open()
        page = doc.new_page(width=595, height=400)
        page.insert_text(
            (50, 80),
            "INCOME TAX DEPARTMENT GOVT. OF INDIA\n"
            "PERMANENT ACCOUNT NUMBER CARD\n"
            "ABCDE1234F\n"
            "MOHIT VERMA"
        )
        doc.save(pdf_path)
        doc.close()

        evaluator = DocumentEvaluator(temp_dir=os.path.join(temp_dir, "tmp"))

        # Ground truth with WRONG expected PAN
        gt = {
            "expected_category": DOC_TYPE_PAN,
            "expected_pan": "ZZZZZ9999Z"  # Mismatch against ABCDE1234F
        }

        res = evaluator.evaluate_file(pdf_path, ground_truth=gt, ocr_reader=None)

        assert res["status"] == "FAILED"
        assert res["error_category"] == "PAN_EXTRACTION_FAILURE"
        assert res["evaluations"]["pan"]["match"] is False
    finally:
        if os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError: pass


def test_evaluator_processing_error_is_not_success():
    """Verify corrupted or unreadable documents yield FAILED status and error category."""
    temp_dir = tempfile.mkdtemp()
    corrupt_file = os.path.join(temp_dir, "corrupt.pdf")
    try:
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_PDF_HEADER_DATA")

        evaluator = DocumentEvaluator(temp_dir=os.path.join(temp_dir, "tmp"))
        gt = {"expected_category": DOC_TYPE_PAN}

        res = evaluator.evaluate_file(corrupt_file, ground_truth=gt, ocr_reader=None)

        assert res["status"] == "FAILED"
        assert res["error_category"] in ("FILE_UNREADABLE", "OCR_PROCESSING_ERROR", "CLASSIFICATION_MISMATCH")
    finally:
        if os.path.exists(corrupt_file):
            try: os.remove(corrupt_file)
            except OSError: pass


# =====================================================================
# 3. EVALUATION RUNNER BATCH & REPORT TESTS
# =====================================================================

def test_runner_batch_execution_and_report_generation():
    """Verify batch runner execution, aggregate metrics, and report export."""
    data_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()

    try:
        # Create subfolders for PAN, GST, and Udyam
        pan_dir = os.path.join(data_dir, "pan")
        gst_dir = os.path.join(data_dir, "gst")
        udyam_dir = os.path.join(data_dir, "udyam")
        os.makedirs(pan_dir, exist_ok=True)
        os.makedirs(gst_dir, exist_ok=True)
        os.makedirs(udyam_dir, exist_ok=True)

        # 1. PAN Doc
        doc1 = fitz.open()
        p1 = doc1.new_page(width=595, height=400)
        p1.insert_text((50, 80), "INCOME TAX DEPARTMENT GOVT. OF INDIA\nPERMANENT ACCOUNT NUMBER CARD\nAAACK1234K\nKAVERI INFRASTRUCTURE PRIVATE LIMITED\n18/09/2012")
        doc1.save(os.path.join(pan_dir, "corp_pan.pdf"))
        doc1.close()

        # 2. GST Doc
        doc2 = fitz.open()
        p2 = doc2.new_page(width=595, height=400)
        p2.insert_text((50, 80), "GOVERNMENT OF INDIA FORM GST REG-06\nREGISTRATION CERTIFICATE\nGSTIN: 27AAPFU0939F1ZV\nAPEX INFOTECH PVT LTD\nIssue Date: 15/05/2024\nDeclared Turnover: Rs. 100 Lakhs")
        doc2.save(os.path.join(gst_dir, "cert_gst.pdf"))
        doc2.close()

        # 3. Udyam Doc
        doc3 = fitz.open()
        p3 = doc3.new_page(width=595, height=400)
        p3.insert_text((50, 80), "UDYAM REGISTRATION CERTIFICATE\nMINISTRY OF MICRO, SMALL & MEDIUM ENTERPRISES\nUDYAM-MH-01-0012345\nNAME OF ENTERPRISE: APEX INFOTECH")
        doc3.save(os.path.join(udyam_dir, "cert_udyam.pdf"))
        doc3.close()

        # Create Ground Truth
        gt_path = os.path.join(data_dir, "test_ground_truth.json")
        gt_data = {
            "documents": {
                "pan/corp_pan.pdf": {
                    "expected_category": DOC_TYPE_PAN,
                    "expected_pan": "AAACK1234K"
                },
                "gst/cert_gst.pdf": {
                    "expected_category": DOC_TYPE_GST,
                    "expected_gstin": "27AAPFU0939F1ZV"
                },
                "udyam/cert_udyam.pdf": {
                    "expected_category": DOC_TYPE_UDYAM,
                    "expected_udyam": "UDYAM-MH-01-0012345"
                }
            }
        }
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f)

        runner = EvaluationRunner(
            data_dir=data_dir,
            ground_truth_path=gt_path,
            output_dir=out_dir,
            use_ocr=False,
            quiet=True
        )

        report = runner.run()

        # Check Aggregate Summary
        assert report["summary"]["total_documents_tested"] == 3
        assert report["summary"]["passed_documents"] == 3
        assert report["summary"]["classification_accuracy_pct"] == 100.0
        assert report["field_extraction_accuracy"]["aadhaar_zero_leak_rate_pct"] == 100.0

        # Check File Outputs
        report_files = os.listdir(out_dir)
        json_files = [f for f in report_files if f.endswith(".json")]
        md_files = [f for f in report_files if f.endswith(".md")]
        assert len(json_files) == 1
        assert len(md_files) == 1

        # Check Zero-PII Guarantee in exported JSON
        with open(os.path.join(out_dir, json_files[0]), "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["summary"]["total_documents_tested"] == 3

        # Confirm no unmasked Aadhaar numbers in entire JSON content
        json_content_str = json.dumps(saved_json)
        assert contains_unmasked_aadhaar(json_content_str) is False
        assert "9876 5432 1098" not in json_content_str

    finally:
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_runner_empty_directory_handling():
    """Verify runner handles empty directories gracefully without crashing."""
    empty_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()
    try:
        runner = EvaluationRunner(
            data_dir=empty_dir,
            ground_truth_path=None,
            output_dir=out_dir,
            use_ocr=False,
            quiet=True
        )
        report = runner.run()
        assert report["total_documents"] == 0
    finally:
        import shutil
        shutil.rmtree(empty_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


# =====================================================================
# 4. REAL-WORLD EXTRACTION & PRIVACY REGRESSION TESTS (SIH26100)
# =====================================================================

def test_pan_corporate_entity_name_and_not_applicable_father():
    """Verify corporate PAN cards extract legal entity name without father's name."""
    from src.pan_extractor import extract_pan_details

    # Synthetic corporate PAN items (4th char 'C' for Company, with Legal Entity label)
    items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 40, 300, 60], "conf": 0.98, "page": 1},
        {"text": "Legal Entity Name / Name of Holder:", "bbox": [50, 90, 350, 110], "conf": 0.95, "page": 1},
        {"text": "DELTA LOGISTICS & INFRASTRUCTURE PVT LTD", "bbox": [50, 120, 450, 140], "conf": 0.99, "page": 1},
        {"text": "Date of Incorporation: 14/05/2018", "bbox": [50, 160, 300, 180], "conf": 0.97, "page": 1},
        {"text": "Permanent Account Number: AAACD1234D", "bbox": [50, 200, 350, 220], "conf": 0.99, "page": 1}
    ]
    raw_text = "INCOME TAX DEPARTMENT\nLegal Entity Name / Name of Holder:\nDELTA LOGISTICS & INFRASTRUCTURE PVT LTD\nDate of Incorporation: 14/05/2018\nPermanent Account Number: AAACD1234D"

    res = extract_pan_details(items, raw_text)
    assert res["pan"] == "AAACD1234D"
    assert res["entity_type"] == "Company"
    assert res["cardholder_name"] == "DELTA LOGISTICS & INFRASTRUCTURE PVT LTD"
    assert res["father_name"] is None
    assert res["father_name_status"] == "NOT_APPLICABLE"


def test_pan_bilingual_label_name_and_father_disambiguation():
    """Verify individual bilingual PAN card cleanly separates cardholder and father names."""
    from src.pan_extractor import extract_pan_details

    # Synthetic individual PAN with Name / Father's Name bilingual labels
    items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 30, 300, 50], "conf": 0.99, "page": 1},
        {"text": "Name / नाम", "bbox": [50, 80, 180, 100], "conf": 0.95, "page": 1},
        {"text": "RAHUL", "bbox": [50, 110, 120, 130], "conf": 0.98, "page": 1},
        {"text": "SHARMA", "bbox": [130, 111, 220, 131], "conf": 0.97, "page": 1},
        {"text": "Father's Name / पिता का नाम", "bbox": [50, 150, 260, 170], "conf": 0.94, "page": 1},
        {"text": "SURESH", "bbox": [50, 180, 140, 200], "conf": 0.98, "page": 1},
        {"text": "SHARMA", "bbox": [150, 181, 240, 201], "conf": 0.96, "page": 1},
        {"text": "10/08/1992", "bbox": [50, 220, 160, 240], "conf": 0.99, "page": 1},
        {"text": "ABZPS1234S", "bbox": [50, 260, 200, 280], "conf": 0.99, "page": 1}
    ]
    raw_text = "INCOME TAX DEPARTMENT\nName / नाम\nRAHUL SHARMA\nFather's Name / पिता का नाम\nSURESH SHARMA\n10/08/1992\nABZPS1234S"

    res = extract_pan_details(items, raw_text)
    assert res["pan"] == "ABZPS1234S"
    assert res["cardholder_name"] == "RAHUL SHARMA"
    assert res["father_name"] == "SURESH SHARMA"
    assert res["dob"] == "10/08/1992"


def test_gst_candidate_preservation_when_checksum_fails():
    """Verify GSTIN candidates are preserved and marked CHECKSUM_FAILED without dropping."""
    from src.upload_workflow import DocumentStagingQueue

    # Synthetic GST doc with mathematically invalid checksum '07AAAAA1234A1Z5'
    # and separately printed PAN 'ABCDE1234F'
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=500)
    page.insert_text(
        (50, 60),
        "GOVERNMENT OF INDIA FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number: 07AAAAA1234A1Z5\n"
        "Legal Name: ACME GLOBAL TRADING PVT LTD\n"
        "Permanent Account Number: ABCDE1234F\n"
        "Date of Liability: 01/04/2021"
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc = queue.add_file("synthetic_gst.pdf", pdf_bytes)
    queue.process_document_ocr(doc, ocr_reader=None)

    assert doc.detected_type == DOC_TYPE_GST
    assert doc.extracted_fields["gstin"] == "07AAAAA1234A1Z5"
    assert doc.extracted_fields["gstin_status"] == "CHECKSUM_FAILED"
    assert doc.extracted_fields["gstin_checksum_valid"] is False
    assert doc.extracted_fields["entity_name"] == "ACME GLOBAL TRADING PVT LTD"
    assert doc.extracted_fields["pan"] == "ABCDE1234F"
    assert doc.extracted_fields["embedded_gstin_pan"] == "AAAAA1234A"
    assert doc.extracted_fields["pan_consistency"] is False


def test_aadhaar_classification_without_uidai_keyword():
    """Verify Aadhaar is correctly classified even if 'UIDAI' phrase is absent."""
    from src.document_classifier import classify_document_text, DOC_TYPE_AADHAAR, DOC_TYPE_PAN

    # Text typical of cropped or lower-res Aadhaar card with Government of India + Gender + S/O
    aadhaar_sample = (
        "GOVERNMENT OF INDIA\n"
        "MOHIT KUMAR\n"
        "DOB: 12/05/1994\n"
        "MALE / पुरुष\n"
        "S/O: RAMESH KUMAR\n"
        "ADDRESS: HOUSE NO 123 CIVIL LINES\n"
        "XXXX-XXXX-9012"
    )

    res = classify_document_text(aadhaar_sample)
    assert res["document_type"] == DOC_TYPE_AADHAAR
    assert res["document_type"] != DOC_TYPE_PAN


def test_udyam_noisy_ocr_normalization():
    """Verify Udyam numbers with whitespace and 'O'/'0' confusion are normalized cleanly."""
    from src.upload_workflow import DocumentStagingQueue

    # Text with OCR noise in Udyam registration
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 60),
        "UDYAM REGISTRATION CERTIFICATE\n"
        "MINISTRY OF MSME\n"
        "UDYAM - UP - 12 - OO29329\n"
        "NAME OF ENTERPRISE: GOND TRADING COMPANY"
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc = queue.add_file("synthetic_udyam.pdf", pdf_bytes)
    queue.process_document_ocr(doc, ocr_reader=None)

    assert doc.detected_type == DOC_TYPE_UDYAM
    assert doc.extracted_fields["udyam"] == "UDYAM-UP-12-0029329"


def test_zero_pii_report_leak_regression():
    """Verify exported evaluation reports contain zero unmasked synthetic identifiers."""
    data_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()

    try:
        pan_dir = os.path.join(data_dir, "pan")
        os.makedirs(pan_dir, exist_ok=True)

        doc = fitz.open()
        p = doc.new_page(width=500, height=300)
        p.insert_text(
            (50, 60),
            "INCOME TAX DEPARTMENT\n"
            "PERMANENT ACCOUNT NUMBER CARD\n"
            "ABCDE1234F\n"
            "ANANYA DESHMUKH\n"
            "SURESH DESHMUKH\n"
            "15/08/1990"
        )
        doc.save(os.path.join(pan_dir, "pan_sample.pdf"))
        doc.close()

        gt_path = os.path.join(data_dir, "gt.json")
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump({
                "documents": {
                    "pan/pan_sample.pdf": {
                        "expected_category": DOC_TYPE_PAN,
                        "expected_pan": "ABCDE1234F",
                        "expected_name": "ANANYA DESHMUKH",
                        "expected_father_name": "SURESH DESHMUKH",
                        "expected_dob": "15/08/1990"
                    }
                }
            }, f)

        runner = EvaluationRunner(data_dir=data_dir, ground_truth_path=gt_path, output_dir=out_dir, use_ocr=False, quiet=True)
        runner.run()

        # Read generated JSON and Markdown
        for fname in os.listdir(out_dir):
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            # Ensure zero unmasked personal identifiers appear in reports
            assert "ABCDE1234F" not in content, f"Unmasked PAN leaked in {fname}"
            assert "ANANYA DESHMUKH" not in content, f"Unmasked Name leaked in {fname}"
            assert "SURESH DESHMUKH" not in content, f"Unmasked Father Name leaked in {fname}"
            assert "15/08/1990" not in content, f"Unmasked DOB leaked in {fname}"
    finally:
        import shutil
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_gstin_extraction_with_whitespace_and_entity_ocr_confusion():
    """Verify GSTIN candidates with component whitespace and entity 'I'/'1' confusion are extracted."""
    from src.upload_workflow import DocumentStagingQueue

    # 1. GST with space-separated components and kerning
    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 60),
        "GOVERNMENT OF INDIA FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number : 07 AAAAA 1234 A 1 Z 5\n"
        "Legal Name: SHREEYA LOGISTICS & INFRA PVT LTD"
    )
    pdf_bytes1 = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc1 = queue.add_file("gst_spaced.pdf", pdf_bytes1)
    queue.process_document_ocr(doc1, ocr_reader=None)
    assert doc1.extracted_fields["gstin"] == "07AAAAA1234A1Z5"
    assert doc1.extracted_fields["gstin_status"] == "CHECKSUM_FAILED"

    # 2. GST with OCR entity character confusion ('I' instead of '1')
    pdf_doc2 = fitz.open()
    page2 = pdf_doc2.new_page(width=600, height=400)
    page2.insert_text(
        (50, 60),
        "GOVERNMENT OF INDIA FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number: 07SHYLA8743MIZG\n"
        "Legal Name: SHREEYA LOGISTICS & INFRA PVT LTD"
    )
    pdf_bytes2 = pdf_doc2.tobytes()
    pdf_doc2.close()

    doc2 = queue.add_file("gst_entity_confused.pdf", pdf_bytes2)
    queue.process_document_ocr(doc2, ocr_reader=None)
    assert doc2.extracted_fields["gstin"] == "07SHYLA8743M1ZG"
    assert doc2.extracted_fields["gstin_status"] == "CHECKSUM_FAILED"


def test_gst_pan_discrepancy_and_separate_pan_detection():
    """Verify GST certificate with separate PAN differing from embedded PAN reports discrepancy correctly."""
    from src.upload_workflow import DocumentStagingQueue

    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=500)
    page.insert_text(
        (50, 60),
        "GOVERNMENT OF INDIA FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number: 07SHYLA8743M1ZG\n"
        "Legal Name: SHREEYA LOGISTICS & INFRA PVT LTD\n"
        "Permanent Account Number : ABCDE 1234 F\n"
        "Date of Liability: 01/07/2017"
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc = queue.add_file("gst_discrepancy.pdf", pdf_bytes)
    queue.process_document_ocr(doc, ocr_reader=None)

    assert doc.extracted_fields["gstin"] == "07SHYLA8743M1ZG"
    assert doc.extracted_fields["pan"] == "ABCDE1234F"
    assert doc.extracted_fields["separate_printed_pan"] == "ABCDE1234F"
    assert doc.extracted_fields["embedded_gstin_pan"] == "SHYLA8743M"
    assert doc.extracted_fields["pan_consistency"] is False


def test_pan_name_noise_word_and_surname_matching():
    """Verify PAN card name extraction strips label noise words and matches surname."""
    from src.pan_extractor import extract_pan_details

    # Synthetic items where label word 'HOLDER' or 'HASTAKSHAR' is adjacent to cardholder name
    items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 30, 300, 50], "conf": 0.99, "page": 1},
        {"text": "HOLDER", "bbox": [50, 70, 110, 90], "conf": 0.88, "page": 1},
        {"text": "TWITTERPREET", "bbox": [120, 70, 240, 90], "conf": 0.99, "page": 1},
        {"text": "SINGH", "bbox": [250, 70, 320, 90], "conf": 0.99, "page": 1},
        {"text": "BALWINDER SINGH", "bbox": [50, 120, 250, 140], "conf": 0.98, "page": 1},
        {"text": "14/05/1995", "bbox": [50, 170, 160, 190], "conf": 0.99, "page": 1},
        {"text": "BWZPS1234R", "bbox": [50, 220, 200, 240], "conf": 0.99, "page": 1}
    ]
    raw_text = "INCOME TAX DEPARTMENT\nHOLDER TWITTERPREET SINGH\nBALWINDER SINGH\n14/05/1995\nBWZPS1234R"

    res = extract_pan_details(items, raw_text)
    assert res["pan"] == "BWZPS1234R"
    assert res["cardholder_name"] == "TWITTERPREET SINGH"
    assert res["father_name"] == "BALWINDER SINGH"
    assert res["dob"] == "14/05/1995"


def test_udyam_internal_spacing_and_o0_confusion():
    """Verify Udyam numbers with internal spaces and letter confusion are normalized."""
    from src.upload_workflow import DocumentStagingQueue

    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 60),
        "UDYAM REGISTRATION CERTIFICATE\n"
        "MINISTRY OF MSME\n"
        "UDYAM - UP - 12 - 00 29329\n"
        "NAME OF ENTERPRISE: ROSHAN ENTERPRISES"
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc = queue.add_file("udyam_spaced.pdf", pdf_bytes)
    queue.process_document_ocr(doc, ocr_reader=None)

    assert doc.detected_type == DOC_TYPE_UDYAM
    assert doc.extracted_fields["udyam"] == "UDYAM-UP-12-0029329"


def test_pan_name_separate_line_header_noise_disambiguation():
    """Verify separate line header/noise above name does not contaminate cardholder name."""
    from src.pan_extractor import extract_pan_details

    # Line 0: Header noise (separate line, y0=50)
    # Line 1: Cardholder Name (y0=90, ends in 'S' matching PAN 5th char 'S')
    # Line 2: Father's Name (y0=130)
    items = [
        {"text": "INCOME TAX DEPARTMENT", "bbox": [50, 20, 300, 40], "conf": 0.99, "page": 1, "y0": 20, "h": 20, "extraction_method": "VISION_OCR"},
        {"text": "HEADER NOISE", "bbox": [50, 50, 150, 70], "conf": 0.85, "page": 1, "y0": 50, "h": 20, "extraction_method": "VISION_OCR"},
        {"text": "TWITTERPREET SINGH", "bbox": [50, 90, 250, 110], "conf": 0.98, "page": 1, "y0": 90, "h": 20, "extraction_method": "VISION_OCR"},
        {"text": "BALWINDER SINGH", "bbox": [50, 130, 240, 150], "conf": 0.97, "page": 1, "y0": 130, "h": 20, "extraction_method": "VISION_OCR"},
        {"text": "14/05/1995", "bbox": [50, 170, 160, 190], "conf": 0.99, "page": 1, "y0": 170, "h": 20, "extraction_method": "VISION_OCR"},
        {"text": "BWZPS1234R", "bbox": [50, 220, 200, 240], "conf": 0.99, "page": 1, "y0": 220, "h": 20, "extraction_method": "VISION_OCR"}
    ]
    raw_text = "INCOME TAX DEPARTMENT\nHEADER NOISE\nTWITTERPREET SINGH\nBALWINDER SINGH\n14/05/1995\nBWZPS1234R"

    res = extract_pan_details(items, raw_text)
    assert res["pan"] == "BWZPS1234R"
    assert res["cardholder_name"] == "TWITTERPREET SINGH"
    assert res["father_name"] == "BALWINDER SINGH"
    assert res["dob"] == "14/05/1995"


def test_gst_split_token_gstin_extraction():
    """Verify GSTIN split across multiple token lines in scanned layout is reconstructed."""
    from src.upload_workflow import DocumentStagingQueue

    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 60),
        "FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number\n"
        "07\n"
        "AAAAA1234A\n"
        "1Z5\n"
        "Legal Name: SHREEYA LOGISTICS & INFRA PVT LTD\n"
        "Permanent Account Number: AAAAA1234A"
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc = queue.add_file("gst_split.pdf", pdf_bytes)
    queue.process_document_ocr(doc, ocr_reader=None)

    assert doc.detected_type == DOC_TYPE_GST
    assert doc.extracted_fields["gstin"] == "07AAAAA1234A1Z5"
    assert doc.extracted_fields["pan"] == "AAAAA1234A"


def test_gst_split_separate_pan_extraction():
    """Verify separate PAN split into multiple token lines is reconstructed and detected."""
    from src.upload_workflow import DocumentStagingQueue

    pdf_doc = fitz.open()
    page = pdf_doc.new_page(width=600, height=400)
    page.insert_text(
        (50, 60),
        "FORM GST REG-06\n"
        "REGISTRATION CERTIFICATE\n"
        "Registration Number: 07SHYLA8743M1ZG\n"
        "Legal Name: SHREEYA LOGISTICS & INFRA PVT LTD\n"
        "Permanent Account Number\n"
        "ABCDE\n"
        "1234F"
    )
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    queue = DocumentStagingQueue()
    doc = queue.add_file("gst_split_pan.pdf", pdf_bytes)
    queue.process_document_ocr(doc, ocr_reader=None)

    assert doc.detected_type == DOC_TYPE_GST
    assert doc.extracted_fields["gstin"] == "07SHYLA8743M1ZG"
    assert doc.extracted_fields["pan"] == "ABCDE1234F"
    assert doc.extracted_fields["separate_printed_pan"] == "ABCDE1234F"
    assert doc.extracted_fields["embedded_gstin_pan"] == "SHYLA8743M"
    assert doc.extracted_fields["pan_consistency"] is False
