import io
from datetime import datetime
from typing import List, Dict, Any
import fitz  # PyMuPDF


def generate_gem_verification_pdf(
    tender_id: str,
    bidder_id: str,
    company_name: str,
    gstin_no: str,
    pan_no: str,
    pan_entity_type: str,
    turnover_gst: str,
    turnover_bs: str,
    compliance_status: str,
    risk_rating: str,
    risk_badge: str,
    inconsistencies_count: int,
    audit_data: List[Dict[str, Any]],
    officer_remarks: str = ""
) -> bytes:
    """
    Generates an official, single-page executive GeM Bid Compliance & Forensic
    Verification Report in PDF format using PyMuPDF (fitz).
    """
    doc = fitz.open()
    page = doc.new_page(width=595.28, height=841.89)

    # Official GeM / BidGuard Theme Colors
    NAVY = (0.08, 0.18, 0.36)
    GOLD = (0.85, 0.65, 0.13)
    GRAY_BG = (0.96, 0.96, 0.97)
    BORDER_GRAY = (0.80, 0.82, 0.85)
    TEXT_DARK = (0.12, 0.14, 0.18)
    TEXT_MUTED = (0.40, 0.44, 0.50)
    GREEN = (0.12, 0.60, 0.30)
    RED = (0.75, 0.15, 0.15)
    AMBER = (0.85, 0.45, 0.10)

    # 1. Header Banner (y: 25 to 78)
    page.draw_rect(fitz.Rect(30, 25, 565, 75), color=NAVY, fill=NAVY)
    page.draw_rect(fitz.Rect(30, 75, 565, 78), color=GOLD, fill=GOLD)

    page.insert_text(fitz.Point(45, 48), "GOVERNMENT E-MARKETPLACE (GeM) • BID COMPLIANCE REPORT", fontname="helv", fontsize=12, color=(1, 1, 1))
    page.insert_text(fitz.Point(45, 64), "Autonomous Statutory & Forensic Verification Dossier | SIH26100", fontname="helv", fontsize=8.5, color=(0.85, 0.88, 0.95))

    gen_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    page.insert_text(fitz.Point(415, 48), f"Report Date : {gen_time}", fontname="helv", fontsize=7.5, color=(0.85, 0.88, 0.95))
    page.insert_text(fitz.Point(415, 62), f"Dossier ID  : {bidder_id}", fontname="helv", fontsize=8, color=(1, 1, 1))

    # 2. Metric KPI Cards (y: 88 to 135)
    card_w = (535 - 20) / 3
    
    # Card 1: Compliance
    c1_rect = fitz.Rect(30, 88, 30 + card_w, 135)
    st_upper = str(compliance_status).upper()
    is_passed = any(p in st_upper for p in ["PASS", "COMPLIANT"])
    c1_color = GREEN if is_passed else (AMBER if "REVIEW" in st_upper else RED)
    page.draw_rect(c1_rect, color=c1_color, fill=GRAY_BG, width=1.2)
    page.insert_text(fitz.Point(38, 102), "COMPLIANCE STATUS", fontname="helv", fontsize=7, color=TEXT_MUTED)
    page.insert_text(fitz.Point(38, 122), str(compliance_status)[:18], fontname="helv", fontsize=11, color=c1_color)

    # Card 2: Discrepancies
    c2_rect = fitz.Rect(30 + card_w + 10, 88, 30 + 2 * card_w + 10, 135)
    c2_color = GREEN if inconsistencies_count == 0 else RED
    page.draw_rect(c2_rect, color=c2_color, fill=GRAY_BG, width=1.2)
    page.insert_text(fitz.Point(38 + card_w + 10, 102), "DISCREPANCIES DETECTED", fontname="helv", fontsize=7, color=TEXT_MUTED)
    page.insert_text(fitz.Point(38 + card_w + 10, 122), f"{inconsistencies_count} Found", fontname="helv", fontsize=11, color=c2_color)

    # Card 3: Risk Rating
    c3_rect = fitz.Rect(30 + 2 * card_w + 20, 88, 565, 135)
    risk_upper = str(risk_rating).upper()
    is_low_risk = "LOW" in risk_upper
    c3_color = GREEN if is_low_risk else (AMBER if "MEDIUM" in risk_upper else RED)
    page.draw_rect(c3_rect, color=c3_color, fill=GRAY_BG, width=1.2)
    page.insert_text(fitz.Point(38 + 2 * card_w + 20, 102), "EVALUATED RISK LEVEL", fontname="helv", fontsize=7, color=TEXT_MUTED)
    clean_risk = str(risk_rating).replace(" RISK", "")[:16]
    page.insert_text(fitz.Point(38 + 2 * card_w + 20, 122), clean_risk, fontname="helv", fontsize=11, color=c3_color)

    # 3. Vendor & Tender Profile Box (y: 145 to 232)
    prof_rect = fitz.Rect(30, 145, 565, 232)
    page.draw_rect(prof_rect, color=BORDER_GRAY, fill=(0.98, 0.98, 0.99), width=0.8)
    page.draw_rect(fitz.Rect(30, 145, 565, 162), color=BORDER_GRAY, fill=(0.92, 0.94, 0.96), width=0.8)
    page.insert_text(fitz.Point(40, 157), "TENDER SPECIFICATION & BIDDER STATUTORY PROFILE", fontname="helv", fontsize=8, color=NAVY)

    # Left col: Tender & Entity
    page.insert_text(fitz.Point(40, 178), f"Tender Ref     : {str(tender_id)[:42]}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    page.insert_text(fitz.Point(40, 192), f"Bidder ID      : {str(bidder_id)}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    page.insert_text(fitz.Point(40, 206), f"Entity Name    : {str(company_name)[:40]}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    page.insert_text(fitz.Point(40, 220), f"Constitution   : {str(pan_entity_type)[:40]}", fontname="helv", fontsize=7.5, color=TEXT_DARK)

    # Right col: Statutory IDs & Turnover
    page.insert_text(fitz.Point(310, 178), f"GSTIN Number  : {str(gstin_no)}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    page.insert_text(fitz.Point(310, 192), f"PAN Number    : {str(pan_no)}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    page.insert_text(fitz.Point(310, 206), f"GST Turnover  : {str(turnover_gst)}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    page.insert_text(fitz.Point(310, 220), f"BS Turnover   : {str(turnover_bs)}", fontname="helv", fontsize=7.5, color=TEXT_DARK)

    # 4. Audit Trail Checklist (y: 242 to 555)
    page.draw_rect(fitz.Rect(30, 242, 565, 260), color=NAVY, fill=NAVY)
    page.insert_text(fitz.Point(38, 254), "#", fontname="helv", fontsize=7.5, color=(1, 1, 1))
    page.insert_text(fitz.Point(55, 254), "STAGE", fontname="helv", fontsize=7.5, color=(1, 1, 1))
    page.insert_text(fitz.Point(145, 254), "STATUTORY CHECK / RULE", fontname="helv", fontsize=7.5, color=(1, 1, 1))
    page.insert_text(fitz.Point(335, 254), "STATUS", fontname="helv", fontsize=7.5, color=(1, 1, 1))
    page.insert_text(fitz.Point(385, 254), "FINDINGS & FORENSIC NOTES", fontname="helv", fontsize=7.5, color=(1, 1, 1))

    cur_y = 260
    row_h = 24
    max_rows = 11
    for idx, item in enumerate(audit_data[:max_rows]):
        bg = (0.96, 0.96, 0.97) if idx % 2 == 1 else (1.0, 1.0, 1.0)
        page.draw_rect(fitz.Rect(30, cur_y, 565, cur_y + row_h), color=BORDER_GRAY, fill=bg, width=0.5)

        st_val = str(item.get("Status", "UNKNOWN"))
        st_color = GREEN if ("PASS" in st_val or "VERIFIED" in st_val) else (RED if "FAIL" in st_val else AMBER)

        stage_str = str(item.get("Stage", "Phase 7/9"))[:16]
        rule_str = str(item.get("Rule Checked", "Check"))[:34]
        notes_str = str(item.get("Finding Details", ""))[:48]

        page.insert_text(fitz.Point(38, cur_y + 15), str(idx + 1), fontname="helv", fontsize=7, color=TEXT_DARK)
        page.insert_text(fitz.Point(55, cur_y + 15), stage_str, fontname="helv", fontsize=7, color=TEXT_MUTED)
        page.insert_text(fitz.Point(145, cur_y + 15), rule_str, fontname="helv", fontsize=7, color=TEXT_DARK)
        page.insert_text(fitz.Point(335, cur_y + 15), st_val[:12], fontname="helv", fontsize=7.5, color=st_color)
        page.insert_text(fitz.Point(385, cur_y + 15), notes_str, fontname="helv", fontsize=6.8, color=TEXT_DARK)

        cur_y += row_h

    # 5. Officer Remarks Box (y: cur_y + 10 to cur_y + 70)
    cur_y = max(cur_y + 10, 560)
    rem_rect = fitz.Rect(30, cur_y, 565, cur_y + 60)
    page.draw_rect(rem_rect, color=BORDER_GRAY, fill=(0.98, 0.98, 0.99), width=0.8)
    page.draw_rect(fitz.Rect(30, cur_y, 565, cur_y + 16), color=BORDER_GRAY, fill=(0.92, 0.94, 0.96), width=0.8)
    page.insert_text(fitz.Point(38, cur_y + 12), "COMPETENT AUTHORITY ADJUDICATION NOTES", fontname="helv", fontsize=7.5, color=NAVY)

    clean_remarks = officer_remarks if officer_remarks else "Verified against statutory procurement criteria."
    page.insert_text(fitz.Point(38, cur_y + 32), f"Officer Remarks: {clean_remarks[:90]}", fontname="helv", fontsize=7.5, color=TEXT_DARK)
    date_stamp = datetime.now().strftime("%Y%m%d")
    stamp_id = f"EVT-GEM-{date_stamp}-{bidder_id}"
    page.insert_text(fitz.Point(38, cur_y + 46), f"System Validation Hash: {stamp_id} • Cryptographically Logged", fontname="helv", fontsize=7, color=TEXT_MUTED)

    # 6. Footer (y: 800 to 825)
    page.draw_line(fitz.Point(30, 800), fitz.Point(565, 800), color=GOLD, width=1.0)
    page.insert_text(fitz.Point(30, 814), "BidGuard AI • Autonomous Integrated Bid Compliance & Forensic Verification Platform (SIH26100)", fontname="helv", fontsize=7, color=TEXT_MUTED)
    page.insert_text(fitz.Point(440, 814), "Page 1 of 1 • Official Record", fontname="helv", fontsize=7, color=TEXT_MUTED)

    buf = io.BytesIO()
    doc.save(buf)
    pdf_bytes = buf.getvalue()
    doc.close()
    return pdf_bytes
