#!/usr/bin/env python3
"""
BidGuard AI — Synthetic Demo Dossier Generator
Problem Statement: SIH26100 (GeM Procurement Compliance Verification)

Generates 3 realistic, professional document dossiers in both PDF and 200-DPI PNG format:
  1. Set A: Fully Compliant (Green / Low Risk)
  2. Set B: Moderate Discrepancy (Amber / Medium Risk)
  3. Set C: Fraudulent / Mismatched (Red / Critical Risk)
"""

import os
import sys
from pathlib import Path
import fitz  # PyMuPDF

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_pipeline import compute_gstin_checksum, is_valid_gstin


def insert_centered_text(page, center_x: float, y: float, text: str, fontname: str = "helv", fontsize: float = 11.0, color=(0, 0, 0)):
    """Computes exact string width and inserts horizontally centered text."""
    try:
        t_len = fitz.get_text_length(str(text), fontname=fontname, fontsize=fontsize)
        start_x = max(20.0, center_x - (t_len / 2.0))
    except Exception:
        start_x = max(20.0, center_x - 100.0)
    page.insert_text(fitz.Point(start_x, y), str(text), fontname=fontname, fontsize=fontsize, color=color)


def draw_header_banner(page, rect, fill_color, title, subtitle=None):
    """Draws a solid colored header banner with centered white text."""
    page.draw_rect(rect, color=fill_color, fill=fill_color)
    center_x = (rect.x0 + rect.x1) / 2.0
    y_pos = rect.y0 + 22.0
    insert_centered_text(page, center_x, y_pos, title, fontname="helv", fontsize=12.5, color=(1, 1, 1))
    if subtitle:
        insert_centered_text(page, center_x, y_pos + 16.0, subtitle, fontname="helv", fontsize=9.5, color=(0.95, 0.85, 0.3))


def draw_table_row(page, x0, y, w, col_widths, texts, is_header=False, bg_color=None):
    """Draws a structured table row with borders and padded text."""
    h = 24 if is_header else 20
    rect = fitz.Rect(x0, y, x0 + w, y + h)
    
    if bg_color:
        page.draw_rect(rect, color=(0.7, 0.7, 0.7), fill=bg_color, width=0.5)
    else:
        page.draw_rect(rect, color=(0.7, 0.7, 0.7), width=0.5)

    curr_x = x0
    for idx, (cw, txt) in enumerate(zip(col_widths, texts)):
        if idx > 0:
            page.draw_line(fitz.Point(curr_x, y), fitz.Point(curr_x, y + h), color=(0.7, 0.7, 0.7), width=0.5)
        
        fontsize = 9 if is_header else 8.5
        fontcolor = (0.1, 0.1, 0.1) if not is_header else (0.05, 0.15, 0.3)
        
        page.insert_text(
            fitz.Point(curr_x + 6, y + 14),
            str(txt),
            fontname="helv",
            fontsize=fontsize,
            color=fontcolor
        )
        curr_x += cw
    return y + h


# =====================================================================
# 1. PAN CARD GENERATOR
# =====================================================================
def generate_pan_card(out_pdf: str, data: dict):
    doc = fitz.open()
    # Card format: 540 x 340 points
    page = doc.new_page(width=540, height=340)

    # Background card tint & outer border
    card_rect = fitz.Rect(12, 12, 528, 328)
    page.draw_rect(card_rect, color=(0.12, 0.28, 0.52), fill=(0.95, 0.97, 1.0), width=2.0)
    
    # Inner border
    inner_rect = fitz.Rect(16, 16, 524, 324)
    page.draw_rect(inner_rect, color=(0.7, 0.8, 0.9), width=0.8)

    # Top Header Banner
    draw_header_banner(
        page,
        fitz.Rect(16, 16, 524, 75),
        fill_color=(0.08, 0.24, 0.48),
        title="INCOME TAX DEPARTMENT  |  आयकर विभाग",
        subtitle="GOVT. OF INDIA  |  PERMANENT ACCOUNT NUMBER CARD"
    )

    # Left Photo / Hologram Placeholder Box
    photo_rect = fitz.Rect(32, 90, 130, 215)
    page.draw_rect(photo_rect, color=(0.4, 0.5, 0.6), fill=(0.88, 0.92, 0.96), width=1.0)
    page.insert_text(fitz.Point(40, 155), "[ EMBLEM / PHOTO ]", fontname="helv", fontsize=8.0, color=(0.3, 0.4, 0.5))

    # Signature box under photo
    sig_rect = fitz.Rect(32, 230, 130, 275)
    page.draw_rect(sig_rect, color=(0.5, 0.5, 0.5), fill=(1, 1, 1), width=0.8)
    page.insert_text(fitz.Point(38, 255), "हस्ताक्षर / Signature", fontname="helv", fontsize=8.0, color=(0.4, 0.4, 0.4))

    # Center-Right Entity Details
    x_lbl = 150
    y = 105

    # Legal Name
    page.insert_text(fitz.Point(x_lbl, y), "Name / नाम:", fontname="helv", fontsize=8.5, color=(0.35, 0.35, 0.35))
    y += 16
    page.insert_text(fitz.Point(x_lbl, y), data["name"].upper(), fontname="helv", fontsize=11.5, color=(0, 0, 0))
    y += 24

    # Father's Name or Category
    if data.get("father_name"):
        page.insert_text(fitz.Point(x_lbl, y), "Father's Name / पिता का नाम:", fontname="helv", fontsize=8.5, color=(0.35, 0.35, 0.35))
        y += 16
        page.insert_text(fitz.Point(x_lbl, y), data["father_name"].upper(), fontname="helv", fontsize=10.5, color=(0, 0, 0))
        y += 24
    else:
        page.insert_text(fitz.Point(x_lbl, y), "Entity Category / संस्था प्रकार:", fontname="helv", fontsize=8.5, color=(0.35, 0.35, 0.35))
        y += 16
        page.insert_text(fitz.Point(x_lbl, y), data.get("entity_type", "COMPANY / CORPORATION").upper(), fontname="helv", fontsize=10.5, color=(0, 0, 0))
        y += 24

    # Date of Birth / Incorporation
    date_lbl = "Date of Birth / जन्म तिथि:" if data.get("father_name") else "Date of Incorporation / निगमन तिथि:"
    page.insert_text(fitz.Point(x_lbl, y), date_lbl, fontname="helv", fontsize=8.5, color=(0.35, 0.35, 0.35))
    y += 16
    page.insert_text(fitz.Point(x_lbl, y), data["dob"], fontname="helv", fontsize=10.5, color=(0, 0, 0))
    y += 30

    # Permanent Account Number (Highlighted Box)
    pan_box = fitz.Rect(x_lbl, y - 10, x_lbl + 260, y + 26)
    page.draw_rect(pan_box, color=(0.1, 0.3, 0.6), fill=(1, 1, 1), width=1.2)
    page.insert_text(fitz.Point(x_lbl + 8, y + 14), "Permanent Account Number:", fontname="helv", fontsize=8.0, color=(0.3, 0.3, 0.3))
    page.insert_text(fitz.Point(x_lbl + 130, y + 15), data["pan"], fontname="helv", fontsize=13.0, color=(0.05, 0.15, 0.4))

    # QR Code placeholder
    qr_box = fitz.Rect(430, 220, 505, 295)
    page.draw_rect(qr_box, color=(0.3, 0.3, 0.3), fill=(1, 1, 1), width=0.8)
    page.insert_text(fitz.Point(440, 260), "[ QR CODE ]", fontname="helv", fontsize=8.0, color=(0.4, 0.4, 0.4))

    # Bottom Micro-notice
    insert_centered_text(
        page,
        270.0,
        316.0,
        "THIS CARD IS STATUTORY EVIDENCE OF PERMANENT ACCOUNT NUMBER UNDER INCOME TAX ACT, 1961",
        fontname="helv",
        fontsize=6.5,
        color=(0.4, 0.4, 0.4)
    )

    doc.save(out_pdf)
    doc.close()


# =====================================================================
# 2. GST REGISTRATION CERTIFICATE GENERATOR
# =====================================================================
def generate_gst_certificate(out_pdf: str, data: dict):
    doc = fitz.open()
    # A4: 595 x 842 points
    page = doc.new_page(width=595, height=842)

    # Double Outer Border
    page.draw_rect(fitz.Rect(20, 20, 575, 822), color=(0.12, 0.25, 0.45), width=1.5)
    page.draw_rect(fitz.Rect(25, 25, 570, 817), color=(0.12, 0.25, 0.45), width=0.5)

    # Top Header
    y = 55
    insert_centered_text(page, 297.0, y, "Government of India", fontname="helv", fontsize=13.0, color=(0.1, 0.1, 0.1))
    y += 18
    insert_centered_text(page, 297.0, y, "Form GST REG-06", fontname="helv", fontsize=11.0, color=(0.2, 0.2, 0.2))
    y += 15
    insert_centered_text(page, 297.0, y, "[See Rule 10(1)]", fontname="helv", fontsize=9.0, color=(0.4, 0.4, 0.4))
    y += 24
    insert_centered_text(page, 297.0, y, "Registration Certificate", fontname="helv", fontsize=16.0, color=(0.08, 0.24, 0.48))
    y += 16
    insert_centered_text(page, 297.0, y, "(Under Central Goods and Services Tax Act, 2017)", fontname="helv", fontsize=9.5, color=(0.3, 0.3, 0.3))
    y += 25

    # Decorative separator line
    page.draw_line(fitz.Point(40, y), fitz.Point(555, y), color=(0.15, 0.3, 0.55), width=1.2)
    y += 25

    # Details Table
    fields = [
        ("1. Registration Number (GSTIN)", data["gstin"]),
        ("2. Legal Name", data["legal_name"]),
        ("3. Trade Name", data["trade_name"]),
        ("4. Constitution of Business", data.get("constitution", "Private Limited Company")),
        ("5. Address of Principal Place of Business", data["address"]),
        ("6. Date of Liability", "01/07/2017"),
        ("7. Period of Validity", "From 01/07/2017 to Continuing"),
        ("8. Type of Registration", "Regular"),
        ("9. Particulars of Approving Authority", "State Tax Officer, Ward-63, Range-12")
    ]

    for label, val in fields:
        is_gstin = "GSTIN" in label
        row_h = 32 if is_gstin else 24
        
        bg = (0.93, 0.96, 1.0) if is_gstin else None
        cell_rect = fitz.Rect(40, y, 555, y + row_h)
        if bg:
            page.draw_rect(cell_rect, color=(0.75, 0.75, 0.75), fill=bg, width=0.5)
        else:
            page.draw_rect(cell_rect, color=(0.75, 0.75, 0.75), width=0.5)

        page.draw_line(fitz.Point(220, y), fitz.Point(220, y + row_h), color=(0.75, 0.75, 0.75), width=0.5)
        page.insert_text(fitz.Point(48, y + (row_h / 2) + 3), label, fontname="helv", fontsize=9.0, color=(0.2, 0.2, 0.2))
        
        val_size = 11.5 if is_gstin else 9.5
        val_color = (0.05, 0.15, 0.45) if is_gstin else (0, 0, 0)
        page.insert_text(fitz.Point(228, y + (row_h / 2) + 3), val, fontname="helv", fontsize=val_size, color=val_color)
        
        y += row_h

    y += 30

    # Statutory Annual Turnover Box
    to_box = fitz.Rect(40, y, 555, y + 75)
    page.draw_rect(to_box, color=(0.2, 0.4, 0.7), fill=(0.96, 0.98, 1.0), width=1.0)
    
    page.insert_text(fitz.Point(52, y + 20), "ANNUAL TURNOVER DECLARATION & STATUTORY FILINGS", fontname="helv", fontsize=10.0, color=(0.1, 0.25, 0.5))
    page.insert_text(fitz.Point(52, y + 40), f"Annual Turnover Declared for GST Filing (FY 2025-26): Rs. {data['turnover_str']}", fontname="helv", fontsize=11.0, color=(0, 0, 0))
    page.insert_text(fitz.Point(52, y + 58), "Filing Verification: GSTR-1, GSTR-3B, GSTR-9 Returns Verified without default.", fontname="helv", fontsize=8.5, color=(0.3, 0.3, 0.3))

    y += 110

    # Digital Signature Block
    sig_box = fitz.Rect(340, y, 555, y + 75)
    page.draw_rect(sig_box, color=(0.4, 0.6, 0.4), fill=(0.96, 1.0, 0.96), width=0.8)
    page.insert_text(fitz.Point(350, y + 18), "DIGITALLY SIGNED", fontname="helv", fontsize=9.5, color=(0.1, 0.45, 0.1))
    page.insert_text(fitz.Point(350, y + 34), "Signer: Jurisdictional Tax Officer", fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(350, y + 48), "Date: 15/07/2017 11:24:08 IST", fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(350, y + 62), "Status: Cryptographically Verified", fontname="helv", fontsize=8.0, color=(0.3, 0.5, 0.3))

    # Bottom notice
    insert_centered_text(
        page,
        297.0,
        805.0,
        "This is an official system generated GSTIN Registration Certificate under GSTN Authority.",
        fontname="helv",
        fontsize=8.0,
        color=(0.5, 0.5, 0.5)
    )

    doc.save(out_pdf)
    doc.close()


# =====================================================================
# 3. UDYAM REGISTRATION CERTIFICATE GENERATOR
# =====================================================================
def generate_udyam_certificate(out_pdf: str, data: dict):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Decorative Green Border
    page.draw_rect(fitz.Rect(20, 20, 575, 822), color=(0.1, 0.4, 0.22), width=1.5)
    page.draw_rect(fitz.Rect(25, 25, 570, 817), color=(0.1, 0.4, 0.22), width=0.5)

    # Header
    y = 55
    insert_centered_text(page, 297.0, y, "GOVERNMENT OF INDIA", fontname="helv", fontsize=12.0, color=(0.1, 0.1, 0.1))
    y += 18
    insert_centered_text(page, 297.0, y, "MINISTRY OF MICRO, SMALL & MEDIUM ENTERPRISES", fontname="helv", fontsize=11.0, color=(0.15, 0.15, 0.15))
    y += 24
    insert_centered_text(page, 297.0, y, "UDYAM REGISTRATION CERTIFICATE", fontname="helv", fontsize=16.0, color=(0.08, 0.38, 0.2))
    y += 20

    page.draw_line(fitz.Point(40, y), fitz.Point(555, y), color=(0.15, 0.45, 0.25), width=1.0)
    y += 25

    # Details
    fields = [
        ("UDYAM REGISTRATION NUMBER", data["udyam_number"]),
        ("NAME OF ENTERPRISE", data["enterprise_name"]),
        ("TYPE OF ENTERPRISE", data.get("enterprise_type", "SMALL ENTERPRISE")),
        ("MAJOR ACTIVITY", "SERVICES"),
        ("SOCIAL CATEGORY OF ENTREPRENEUR", "GENERAL"),
        ("DATE OF INCORPORATION / REGISTRATION", "15/06/2018"),
        ("DATE OF COMMENCEMENT OF BUSINESS", "01/07/2018"),
        ("NATIONAL INDUSTRY CLASSIFICATION (NIC)", "6201 - Computer programming, consultancy and related activities"),
        ("DATE OF UDYAM REGISTRATION", "20/08/2020")
    ]

    for label, val in fields:
        is_udyam = "REGISTRATION NUMBER" in label
        row_h = 32 if is_udyam else 24
        
        bg = (0.94, 0.98, 0.94) if is_udyam else None
        cell_rect = fitz.Rect(40, y, 555, y + row_h)
        if bg:
            page.draw_rect(cell_rect, color=(0.75, 0.75, 0.75), fill=bg, width=0.5)
        else:
            page.draw_rect(cell_rect, color=(0.75, 0.75, 0.75), width=0.5)

        page.draw_line(fitz.Point(240, y), fitz.Point(240, y + row_h), color=(0.75, 0.75, 0.75), width=0.5)
        page.insert_text(fitz.Point(48, y + (row_h / 2) + 3), label, fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))
        
        val_size = 12.0 if is_udyam else 9.5
        val_color = (0.05, 0.35, 0.15) if is_udyam else (0, 0, 0)
        page.insert_text(fitz.Point(248, y + (row_h / 2) + 3), val, fontname="helv", fontsize=val_size, color=val_color)
        
        y += row_h

    y += 40

    # Official Seal / Notice Box
    notice_box = fitz.Rect(40, y, 555, y + 80)
    page.draw_rect(notice_box, color=(0.2, 0.5, 0.3), fill=(0.97, 0.99, 0.97), width=0.8)
    page.insert_text(fitz.Point(52, y + 22), "MINISTRY OF MICRO, SMALL AND MEDIUM ENTERPRISES", fontname="helv", fontsize=9.5, color=(0.1, 0.35, 0.15))
    page.insert_text(fitz.Point(52, y + 40), "This certificate is digitally issued under the Micro, Small and Medium Enterprises Development Act, 2006.", fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(52, y + 56), "No physical signature is required. Verification URL: https://udyamregistration.gov.in", fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))

    # Bottom notice
    insert_centered_text(
        page,
        297.0,
        805.0,
        "Official MSME Udyam Registration Portal  |  Ministry of MSME, Government of India",
        fontname="helv",
        fontsize=8.0,
        color=(0.5, 0.5, 0.5)
    )

    doc.save(out_pdf)
    doc.close()


# =====================================================================
# 4. AUDITED BALANCE SHEET GENERATOR
# =====================================================================
def generate_balance_sheet(out_pdf: str, data: dict):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Formal Corporate Border
    page.draw_rect(fitz.Rect(20, 20, 575, 822), color=(0.2, 0.2, 0.2), width=1.0)

    # Header
    y = 48
    insert_centered_text(page, 297.0, y, data["entity_name"].upper(), fontname="helv", fontsize=14.0, color=(0, 0, 0))
    y += 16
    insert_centered_text(page, 297.0, y, f"CIN: U72900DL2018PTC334512  |  PAN: {data['pan']}", fontname="helv", fontsize=8.5, color=(0.3, 0.3, 0.3))
    y += 14
    insert_centered_text(page, 297.0, y, f"Registered Office: {data['address']}", fontname="helv", fontsize=8.0, color=(0.4, 0.4, 0.4))
    y += 18
    page.draw_line(fitz.Point(35, y), fitz.Point(560, y), color=(0.2, 0.2, 0.2), width=1.0)
    y += 20

    insert_centered_text(page, 297.0, y, "INDEPENDENT AUDITOR'S REPORT & AUDITED FINANCIAL STATEMENTS", fontname="helv", fontsize=10.0, color=(0.15, 0.15, 0.15))
    y += 16
    insert_centered_text(page, 297.0, y, "BALANCE SHEET AS AT 31ST MARCH, 2026", fontname="helv", fontsize=12.0, color=(0.08, 0.24, 0.48))
    y += 22

    # Financial Table Grid
    col_w = [265, 55, 105, 100]
    total_w = sum(col_w)
    
    y = draw_table_row(
        page, 35, y, total_w, col_w,
        ["Particulars", "Note No.", "As at 31.03.2026", "As at 31.03.2025"],
        is_header=True,
        bg_color=(0.90, 0.93, 0.97)
    )

    rows = [
        ["I. EQUITY AND LIABILITIES", "", "", ""],
        ["  (1) Shareholders' Funds", "", "", ""],
        ["      (a) Share Capital", "1", "1,50,00,000", "1,50,00,000"],
        [f"      (b) Reserves & Surplus (Net Worth: Rs. {data['net_worth']})", "2", "3,30,00,000", "2,40,00,000"],
        ["  (2) Non-Current Liabilities", "3", "2,10,00,000", "1,80,00,000"],
        ["  (3) Current Liabilities (Trade Payables)", "4", "4,25,00,000", "3,15,00,000"],
        ["TOTAL EQUITY & LIABILITIES", "", "11,15,00,000", "8,85,00,000"],
        ["II. ASSETS", "", "", ""],
        ["  (1) Non-Current Assets (Fixed Assets)", "5", "5,10,00,000", "4,30,00,000"],
        ["  (2) Current Assets", "", "", ""],
        ["      (a) Trade Receivables", "6", "3,80,00,000", "2,95,00,000"],
        ["      (b) Cash and Bank Balances", "7", "2,25,00,000", "1,60,00,000"],
        ["TOTAL ASSETS", "", "11,15,00,000", "8,85,00,000"],
        ["III. STATEMENT OF PROFIT & LOSS HIGHLIGHTS", "", "", ""],
        [f"  Revenue from Operations (Annual Audited Turnover): Rs. {data['turnover_str']}", "8", f"Rs. {data['turnover_str']}", "10.10 Cr"],
        ["  Other Operating Income", "9", "45,00,000", "35,00,000"],
        ["  Profit Before Tax (PBT)", "10", "1,85,00,000", "1,20,00,000"],
        ["  Profit After Tax (PAT)", "11", "1,38,00,000", "90,00,000"],
    ]

    for r in rows:
        is_bold_row = r[0].startswith("TOTAL") or r[0].startswith("III.") or r[0].startswith("Revenue")
        bg = (0.95, 0.95, 0.95) if is_bold_row else None
        y = draw_table_row(page, 35, y, total_w, col_w, r, bg_color=bg)

    y += 25

    # Auditor Attestation & UDIN Block
    page.draw_rect(fitz.Rect(35, y, 560, y + 80), color=(0.3, 0.3, 0.3), fill=(0.98, 0.98, 0.98), width=0.8)
    
    page.insert_text(fitz.Point(45, y + 18), "In terms of our statutory audit report of even date attached.", fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(45, y + 34), "For & on behalf of Chartered Accountants: M/s R.K. Singhal & Associates", fontname="helv", fontsize=9.0, color=(0.1, 0.1, 0.1))
    page.insert_text(fitz.Point(45, y + 48), "CA R.K. Singhal, Partner (Membership No: 092418 | FRN: 014829N)", fontname="helv", fontsize=8.5, color=(0.2, 0.2, 0.2))
    page.insert_text(fitz.Point(45, y + 64), "UDIN: 26092418AAAAAC7821  |  Date: 28th April, 2026  |  Place: New Delhi", fontname="helv", fontsize=9.0, color=(0.05, 0.2, 0.45))

    doc.save(out_pdf)
    doc.close()


def convert_pdf_to_image(pdf_path: str, out_img: str, dpi: int = 200):
    """Renders the first page of a PDF document to a high-resolution PNG."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    pix.save(out_img)
    doc.close()


# =====================================================================
# MAIN DOSSIER BUILDER
# =====================================================================
def main():
    print("=" * 70)
    print("  BidGuard AI — Synthetic Demo Dossier Generator (SIH26100)")
    print("=" * 70)

    base_dir = PROJECT_ROOT / "demo_dossiers"
    base_dir.mkdir(parents=True, exist_ok=True)

    dossiers = [
        # -------------------------------------------------------------
        # DOSSIER A: FULLY COMPLIANT (LOW RISK)
        # -------------------------------------------------------------
        {
            "folder_name": "set_a_compliant",
            "profile_label": "Dossier A: Fully Compliant (Award Recommended / Low Risk)",
            "pan_data": {
                "name": "TechVanguard Solutions Pvt Ltd",
                "entity_type": "Private Limited Company",
                "dob": "15/05/2018",
                "pan": "AAACT1234K"
            },
            "gst_data": {
                "gstin": "07AAACT1234K1ZI",  # Valid ISO 7064 Mod 36,10 checksum
                "legal_name": "TechVanguard Solutions Pvt Ltd",
                "trade_name": "TechVanguard Solutions",
                "address": "Plot 42, Okhla Industrial Area Phase-III, New Delhi, Delhi 110020",
                "constitution": "Private Limited Company",
                "turnover_str": "12.45 Cr"
            },
            "udyam_data": {
                "udyam_number": "UDYAM-DL-01-0012345",
                "enterprise_name": "TECHVANGUARD SOLUTIONS PVT LTD",
                "enterprise_type": "SMALL ENTERPRISE"
            },
            "bs_data": {
                "entity_name": "TechVanguard Solutions Pvt Ltd",
                "pan": "AAACT1234K",
                "address": "Plot 42, Okhla Industrial Area Phase-III, New Delhi, Delhi 110020",
                "turnover_str": "12.50 Cr",
                "net_worth": "4.80 Cr"
            }
        },
        # -------------------------------------------------------------
        # DOSSIER B: MODERATE DISCREPANCY (MEDIUM RISK)
        # -------------------------------------------------------------
        {
            "folder_name": "set_b_discrepancy",
            "profile_label": "Dossier B: Minor Discrepancies (Officer Scrutiny / Medium Risk)",
            "pan_data": {
                "name": "Apex Buildcon Private Limited",
                "entity_type": "Private Limited Company",
                "dob": "10/08/2016",
                "pan": "AABCA5678B"
            },
            "gst_data": {
                "gstin": "08AABCA5678B1ZT",  # Valid checksum
                "legal_name": "Apex Buildcon Pvt Ltd",  # Abbreviation discrepancy
                "trade_name": "Apex Buildcon",
                "address": "B-14, Industrial Area, Sitapura, Jaipur, Rajasthan 302022",
                "constitution": "Private Limited Company",
                "turnover_str": "10.22 Cr"  # ~4.1% variance vs BS
            },
            "udyam_data": {
                "udyam_number": "UDYAM-RJ-02-0054321",
                "enterprise_name": "APEX BUILDCON PVT LTD",
                "enterprise_type": "SMALL ENTERPRISE"
            },
            "bs_data": {
                "entity_name": "Apex Buildcon Private Limited",
                "pan": "AABCA5678B",
                "address": "B-14, Industrial Area, Sitapura, Jaipur, Rajasthan 302022",
                "turnover_str": "9.80 Cr",
                "net_worth": "3.20 Cr"
            }
        },
        # -------------------------------------------------------------
        # DOSSIER C: FRAUDULENT / MISMATCHED (CRITICAL RISK)
        # -------------------------------------------------------------
        {
            "folder_name": "set_c_fraudulent",
            "profile_label": "Dossier C: Fraudulent / Mismatched (Immediate Disqualification / Critical Risk)",
            "pan_data": {
                "name": "FakeCorp Logistics",
                "entity_type": "Partnership Firm",
                "dob": "12/03/2019",
                "pan": "AABCF9999M"  # Standalone PAN
            },
            "gst_data": {
                "gstin": "07AABCK8888P1ZI",  # Embedded PAN is AABCK8888P (MISMATCH with AABCF9999M!)
                "legal_name": "FakeCorp Logistics Pvt Ltd",
                "trade_name": "FakeCorp Supply Chain",
                "address": "Shop 104, Transport Nagar, Azadpur, New Delhi 110033",
                "constitution": "Private Limited Company",
                "turnover_str": "6.00 Cr"  # Massive 60% variance vs BS
            },
            "udyam_data": {
                "udyam_number": "UDYAM-DL-03-0099887",
                "enterprise_name": "FAKECORP LOGISTICS",
                "enterprise_type": "MICRO ENTERPRISE"
            },
            "bs_data": {
                "entity_name": "FakeCorp Logistics",
                "pan": "AABCF9999M",
                "address": "Shop 104, Transport Nagar, Azadpur, New Delhi 110033",
                "turnover_str": "15.00 Cr",
                "net_worth": "1.10 Cr"
            }
        }
    ]

    total_generated = 0

    for d in dossiers:
        dossier_path = base_dir / d["folder_name"]
        dossier_path.mkdir(parents=True, exist_ok=True)
        print(f"\n[*] Generating: {d['profile_label']}")
        print(f"    Target directory: {dossier_path.relative_to(PROJECT_ROOT)}")

        # 1. PAN Card
        pan_pdf = dossier_path / "pan_card.pdf"
        pan_png = dossier_path / "pan_card.png"
        generate_pan_card(str(pan_pdf), d["pan_data"])
        convert_pdf_to_image(str(pan_pdf), str(pan_png))
        print(f"    [+] Created PAN Card: {pan_pdf.name} & {pan_png.name}")
        total_generated += 2

        # 2. GST Registration Certificate
        gst_pdf = dossier_path / "gst_certificate.pdf"
        gst_png = dossier_path / "gst_certificate.png"
        generate_gst_certificate(str(gst_pdf), d["gst_data"])
        convert_pdf_to_image(str(gst_pdf), str(gst_png))
        print(f"    [+] Created GST Certificate: {gst_pdf.name} & {gst_png.name}")
        total_generated += 2

        # 3. Udyam Registration Certificate
        udyam_pdf = dossier_path / "udyam_certificate.pdf"
        udyam_png = dossier_path / "udyam_certificate.png"
        generate_udyam_certificate(str(udyam_pdf), d["udyam_data"])
        convert_pdf_to_image(str(udyam_pdf), str(udyam_png))
        print(f"    [+] Created Udyam Certificate: {udyam_pdf.name} & {udyam_png.name}")
        total_generated += 2

        # 4. Audited Balance Sheet
        bs_pdf = dossier_path / "balance_sheet.pdf"
        bs_png = dossier_path / "balance_sheet.png"
        generate_balance_sheet(str(bs_pdf), d["bs_data"])
        convert_pdf_to_image(str(bs_pdf), str(bs_png))
        print(f"    [+] Created Balance Sheet: {bs_pdf.name} & {bs_png.name}")
        total_generated += 2

    # Create README for the demo dossiers
    readme_content = """# BidGuard AI — Synthetic Demonstration Dossiers
**Problem Statement ID:** SIH26100 (GeM Procurement Compliance Verification)

These dossiers provide 100% synthetic, realistic document sets formatted for live jury demonstration in the Streamlit Officer Audit Portal (`dashboard.py`).

## 1. Set A: `set_a_compliant` (Green / Low Risk)
- **Profile:** Ideal fully compliant enterprise vendor.
- **Entity:** `TechVanguard Solutions Pvt Ltd`
- **PAN:** `AAACT1234K`
- **GSTIN:** `07AAACT1234K1ZI` (Valid ISO 7064 Mod 36,10 checksum)
- **Udyam:** `UDYAM-DL-01-0012345`
- **Turnover Reconciliation:**
  - Audited Balance Sheet: `Rs. 12.50 Cr`
  - GST Filings: `Rs. 12.45 Cr`
  - Variance: `< 0.4%` (within 5% statutory threshold)
- **Expected Outcome:** 100% Compliant, LOW Risk, Award Recommended.

---

## 2. Set B: `set_b_discrepancy` (Amber / Medium Risk)
- **Profile:** Real-world vendor with minor typographical entity variation and mild turnover discrepancy.
- **Entity on PAN / BS:** `Apex Buildcon Private Limited`
- **Entity on GST / Udyam:** `Apex Buildcon Pvt Ltd`
- **PAN:** `AABCA5678B`
- **GSTIN:** `08AABCA5678B1ZT` (Valid checksum)
- **Udyam:** `UDYAM-RJ-02-0054321`
- **Turnover Reconciliation:**
  - Audited Balance Sheet: `Rs. 9.80 Cr`
  - GST Filings: `Rs. 10.22 Cr`
  - Variance: `~4.1%` (Within 5% threshold, triggers officer advisory note)
- **Expected Outcome:** Medium Risk, Conditional Approval, Officer Scrutiny Required.

---

## 3. Set C: `set_c_fraudulent` (Red / Critical Risk)
- **Profile:** Fraudulent submission with forged credentials and severe revenue misstatement.
- **Standalone PAN Card:** `AABCF9999M` (`FakeCorp Logistics`)
- **GST Certificate GSTIN:** `07AABCK8888P1ZI`
  - Embedded PAN is `AABCK8888P` (FATAL MISMATCH with standalone PAN `AABCF9999M`)
- **Turnover Reconciliation:**
  - Audited Balance Sheet: `Rs. 15.00 Cr`
  - GST Filings: `Rs. 6.00 Cr`
  - Variance: `> 60.0%` (Massive discrepancy exceeding statutory limits)
- **Expected Outcome:** Immediate Disqualification, Critical Risk, Forensic Flag `PAN_MISMATCH_IN_GSTIN`.
"""
    readme_path = base_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"\n[+] Created Documentation: {readme_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print(f"  [SUCCESS] All {total_generated} demo documents generated successfully!")
    print(f"  Location: {base_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
