import pytest
from src.document_classifier import (
    classify_document_text,
    DOC_TYPE_PAN,
    DOC_TYPE_GST,
    DOC_TYPE_BALANCE_SHEET,
    DOC_TYPE_ITR,
    DOC_TYPE_UDYAM,
    DOC_TYPE_UNKNOWN
)

def test_classify_pan_card():
    """Classifies text containing Income Tax Department and PAN card markers as PAN Card."""
    pan_text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    Permanent Account Number Card
    ABCDE1234F
    Name: RAMESH KUMAR
    Father's Name: SURESH KUMAR
    Date of Birth: 12/04/1988
    Signature
    """
    res = classify_document_text(pan_text)
    assert res["document_type"] == DOC_TYPE_PAN
    assert res["confidence"] > 0.60
    assert "INCOME TAX DEPARTMENT" in res["matched_patterns"]

def test_classify_gst_certificate():
    """Classifies text containing Form GST REG-06 and GSTIN as GST Registration Certificate."""
    gst_text = """
    Government of India
    Form GST REG-06
    Registration Certificate
    Registration Number: 27AAPFU0939F1ZV
    Legal Name: APEX INFOTECH PRIVATE LIMITED
    Trade Name: APEX INFOTECH
    Principal Place of Business: Mumbai, Maharashtra
    Date of Liability: 01/07/2017
    Period of Validity: From 01/07/2017 To Perpetual
    Type of Registration: Regular
    """
    res = classify_document_text(gst_text)
    assert res["document_type"] == DOC_TYPE_GST
    assert res["confidence"] > 0.60
    assert "FORM GST REG-06" in res["matched_patterns"]

def test_classify_balance_sheet():
    """Classifies text containing Balance sheet and financial position headers as Balance Sheet."""
    bs_text = """
    SHARMA & ASSOCIATES
    CHARTERED ACCOUNTANTS
    INDEPENDENT AUDITOR'S REPORT
    BALANCE SHEET AS AT 31ST MARCH 2024
    EQUITY AND LIABILITIES
    Shareholder's Funds
    Non-Current Liabilities
    Current Liabilities
    TOTAL ASSETS: Rs. 150.0 Lakhs
    ANNUAL TURNOVER: Rs. 85.0 Lakhs
    UDIN: 24012345AAAAAA1234
    """
    res = classify_document_text(bs_text)
    assert res["document_type"] == DOC_TYPE_BALANCE_SHEET
    assert res["confidence"] > 0.60
    assert "BALANCE SHEET" in res["matched_patterns"]

def test_classify_itr_acknowledgement():
    """Classifies text containing Indian Income Tax Return Acknowledgement as ITR."""
    itr_text = """
    INDIAN INCOME TAX RETURN ACKNOWLEDGEMENT
    ITR-V
    Assessment Year: 2024-25
    e-Filing Acknowledgement Number: 123456789012345
    PAN: ABCDE1234F
    Name: APEX INFOTECH PVT LTD
    Gross Total Income: Rs. 50,00,000
    Tax Payable: Nil
    """
    res = classify_document_text(itr_text)
    assert res["document_type"] == DOC_TYPE_ITR
    assert res["confidence"] > 0.60

def test_classify_udyam_certificate():
    """Classifies text containing Udyam registration markers as Udyam Certificate."""
    udyam_text = """
    UDYAM REGISTRATION CERTIFICATE
    MINISTRY OF MICRO, SMALL & MEDIUM ENTERPRISES
    UDYAM REGISTRATION NUMBER: UDYAM-MH-01-0012345
    NAME OF ENTERPRISE: APEX INFOTECH PVT LTD
    TYPE OF ENTERPRISE: SMALL
    MAJOR ACTIVITY: SERVICES
    """
    res = classify_document_text(udyam_text)
    assert res["document_type"] == DOC_TYPE_UDYAM
    assert res["confidence"] > 0.60

def test_classify_unknown_document():
    """Generic or unclassifiable text falls back to Unknown / Other."""
    random_text = """
    Recipe for Indian Masala Chai
    Boil water, add tea leaves, ginger, cardamom, and milk.
    Simmer for 5 minutes and serve hot.
    """
    res = classify_document_text(random_text)
    assert res["document_type"] == DOC_TYPE_UNKNOWN
    assert res["confidence"] < 0.50

def test_classification_independent_of_filename():
    """Classification operates strictly on content text with no filename input."""
    # A GST certificate content will be identified as GST regardless of what file it was uploaded as
    text = "Form GST REG-06 Registration Certificate GSTIN: 27AAPFU0939F1ZV"
    res = classify_document_text(text)
    assert res["document_type"] == DOC_TYPE_GST

def test_gst_disambiguation_over_embedded_pan():
    """GST certificate containing embedded PAN is categorized as GST Certificate, not PAN card."""
    text = """
    FORM GST REG-06 REGISTRATION CERTIFICATE
    GOODS AND SERVICES TAX
    GSTIN: 27AAACI4798L1ZU
    PAN: AAACI4798L
    LEGAL NAME: INFOSYS LIMITED
    """
    res = classify_document_text(text)
    assert res["document_type"] == DOC_TYPE_GST
