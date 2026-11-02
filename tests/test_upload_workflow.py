import pytest
from src.upload_workflow import DocumentStagingQueue, format_file_size
from src.document_classifier import DOC_TYPE_PAN, DOC_TYPE_GST

def test_format_file_size():
    """format_file_size correctly formats bytes, KB, and MB."""
    assert format_file_size(500) == "500 B"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(1024 * 1024 * 3) == "3.00 MB"

def test_upload_queue_staging_does_not_trigger_verification():
    """Adding files stages them in the queue without running compliance verification."""
    queue = DocumentStagingQueue()
    doc = queue.add_file("sample_doc.pdf", b"%PDF-1.4 sample content")
    
    assert len(queue.documents) == 1
    assert queue.is_verified is False
    assert queue.verification_results is None
    assert doc.status == "Queued"

def test_upload_queue_summary_display():
    """get_queue_summary returns accurate table metadata for UI rendering."""
    queue = DocumentStagingQueue()
    queue.add_file("gst_cert.pdf", b"%PDF-1.4 " + b"0" * 2000)
    queue.add_file("pan_card.png", b"\x89PNG\r\n\x1a\n" + b"0" * 4000)

    summary = queue.get_queue_summary()
    assert len(summary) == 2
    filenames = [s["Filename"] for s in summary]
    assert "gst_cert.pdf" in filenames
    assert "pan_card.png" in filenames
    assert all(s["Status"] == "Queued" for s in summary)

def test_remove_file_before_verification():
    """User can remove a file from queue; queue length updates and verification resets."""
    queue = DocumentStagingQueue()
    queue.add_file("doc1.pdf", b"data1")
    queue.add_file("doc2.pdf", b"data2")
    assert len(queue.documents) == 2

    removed = queue.remove_file("doc1.pdf")
    assert removed is True
    assert len(queue.documents) == 1
    assert "doc1.pdf" not in queue.documents

    # Non-existent file removal returns False
    assert queue.remove_file("unknown.pdf") is False

def test_duplicate_processing_prevention():
    """execute_consolidated_verification caches results and prevents redundant reprocessing on reruns."""
    queue = DocumentStagingQueue()
    
    # Create mock verified state
    queue.is_verified = True
    queue.verification_results = {"compliance_status": "PASSED", "discrepancies_count": 0}

    # Calling verification again returns cached result immediately without OCR reader
    res = queue.execute_consolidated_verification(ocr_reader=None)
    assert res["compliance_status"] == "PASSED"

def test_missing_optional_balance_sheet_does_not_fail():
    """When only GST certificate is provided, missing optional balance sheet does not fail compliance."""
    queue = DocumentStagingQueue()
    
    # Simulate a GST document in the queue
    doc = queue.add_file("gst.txt", b"GST content")
    doc.detected_type = DOC_TYPE_GST
    doc.extracted_fields = {
        "gstin": "27AAPFU0939F1ZV",
        "entity_name": "APEX INFOTECH PVT LTD",
        "turnover": 100.0,
        "turnover_str": "Rs. 100.0 Lakhs",
        "issue_date": "15/05/2024"
    }
    doc.status = "Verified"

    results = queue.execute_consolidated_verification(ocr_reader=None, min_turnover_req=50.0)
    assert results["compliance_status"] in ["PASSED", "PENDING_REVIEW"]
    # Check that missing balance sheet didn't create a hard FAIL
    audit_statuses = [a["Status"] for a in results["audit_data"] if "Balance Sheet" in a["Rule Checked"] or "Turnover Variance" in a["Rule Checked"]]
    assert "FAIL" not in audit_statuses

def test_pan_card_and_gst_cross_check_consolidated():
    """Consolidated verification evaluates both standalone PAN and GST certificate."""
    queue = DocumentStagingQueue()

    # 1. GST doc
    gst_doc = queue.add_file("gst.pdf", b"gst")
    gst_doc.detected_type = DOC_TYPE_GST
    gst_doc.extracted_fields = {
        "gstin": "27AAPFU0939F1ZV",
        "entity_name": "APEX INFOTECH PVT LTD",
        "turnover": 100.0,
        "turnover_str": "Rs. 100.0 Lakhs",
        "issue_date": "15/05/2024"
    }
    gst_doc.status = "Verified"

    # 2. PAN doc (matching PAN)
    pan_doc = queue.add_file("pan.jpg", b"pan")
    pan_doc.detected_type = DOC_TYPE_PAN
    pan_doc.extracted_fields = {
        "pan": "AAPFU0939F",
        "pan_status": "VERIFIED",
        "cardholder_name": "APEX INFOTECH PVT LTD",
        "name_status": "VERIFIED"
    }
    pan_doc.status = "Verified"

    results = queue.execute_consolidated_verification(ocr_reader=None, min_turnover_req=50.0)
    
    # Forensic PAN check must pass
    pan_check = next(a for a in results["audit_data"] if "PAN Embedded" in a["Rule Checked"])
    assert pan_check["Status"] == "PASS"
    assert results["discrepancies_count"] == 0
