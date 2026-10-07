import streamlit as st
import pandas as pd

# Page Setup
st.set_page_config(
    page_title="BidGuard-AI | Officer Procurement Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Top Banner
st.title("🛡️ BidGuard-AI: Officer Audit & Statutory Verification Portal")
st.caption("Automated Procurement Document Forensics & Lineage Tracking System")
st.divider()

# Sidebar Setup
st.sidebar.header("📁 Dossier & Tender Selector")
selected_tender = st.sidebar.selectbox(
    "Active Tender Reference",
    ["TND-2026-001 (Oil & Gas Infrastructure Pipeline)", "TND-2026-002 (Civil Works Expansion)"]
)

bidder_mode = st.sidebar.radio(
    "Select Vendor Submission (Dossier)",
    ["Bidder 101: Apex Infotech (Clean Dossier)", "Bidder 102: Apex Enterprises (Conflicting Dossier)"]
)

st.sidebar.divider()
st.sidebar.markdown("**Evaluation Status:** Ready")
st.sidebar.info("Member 2 Core Modules: Vision OCR + Statutory Compliance + Cross-Doc Engine")

# Data Repository based on Selected Bidder
if "Clean" in bidder_mode:
    # Bidder 101 Data
    company_name = "Apex Infotech Pvt Ltd"
    gstin_no = "07AAAAA0000A1Z5"
    pan_no = "AAAAA0000A"
    gst_turnover_val = "₹ 5.00 Cr"
    bs_turnover_val = "₹ 5.10 Cr"
    
    compliance_status = "PASSED"
    inconsistencies_count = 0
    risk_rating = "LOW RISK"
    
    # Audit Logs
    audit_data = [
        {"Rule Checked": "Phase 7: Turnover Threshold (>= ₹5.0 Cr)", "Status": "PASS", "Finding": "Declared turnover qualifies criteria."},
        {"Rule Checked": "Phase 7: Registration Validity", "Status": "PASS", "Finding": "Valid up to 31/12/2026 (Active status)."},
        {"Rule Checked": "Phase 9: Embedded PAN inside GSTIN", "Status": "PASS", "Finding": f"GSTIN[2:12] '{gstin_no[2:12]}' matches PAN '{pan_no}'."},
        {"Rule Checked": "Phase 9: Legal Entity Name Match", "Status": "PASS", "Finding": "Identical normalized entity name across all certificates."},
        {"Rule Checked": "Phase 9: Turnover Reconciliation", "Status": "PASS", "Finding": "Discrepancy is 1.96% (Well within 5.0% tolerance)."}
    ]
else:
    # Bidder 102 Data (Fraudulent / Conflicting)
    company_name = "Apex Enterprises"
    gstin_no = "07AAAAA0000A1Z5"
    pan_no = "BBBBB9999B"  # Mismatch! Embedded PAN is AAAAA0000A
    gst_turnover_val = "₹ 10.00 Cr"
    bs_turnover_val = "₹ 4.00 Cr"   # 60% Discrepancy!
    
    compliance_status = "FLAGGED"
    inconsistencies_count = 2
    risk_rating = "HIGH RISK"
    
    audit_data = [
        {"Rule Checked": "Phase 7: Turnover Threshold (>= ₹5.0 Cr)", "Status": "PASS", "Finding": "Declared turnover qualifies criteria."},
        {"Rule Checked": "Phase 7: Registration Validity", "Status": "PASS", "Finding": "Valid up to 31/12/2026 (Active status)."},
        {"Rule Checked": "Phase 9: Embedded PAN inside GSTIN", "Status": "FAIL", "Finding": f"Embedded PAN '{gstin_no[2:12]}' does not match Standalone PAN '{pan_no}'."},
        {"Rule Checked": "Phase 9: Legal Entity Name Match", "Status": "PASS", "Finding": "Normalized legal business names match."},
        {"Rule Checked": "Phase 9: Turnover Reconciliation", "Status": "FAIL", "Finding": "Turnover variance 60.0% violates maximum 5.0% tolerance."}
    ]

# UI Layout: Left Column (Document Viewer) | Right Column (Decision & Audit Trail)
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.subheader("📄 Document Inspection & Spatial OCR")
    
    # Document Selector Dropdown
    selected_doc = st.selectbox(
        "Select Statutory Certificate to Inspect",
        ["GST_CERTIFICATE (Form GST REG-06)", "PAN_CARD (Permanent Account Number)", "AUDITED_BALANCE_SHEET (Form 3CD / CA Certified)"]
    )
    
    st.caption("Bounding boxes [x, y, w, h] indicate exact coordinate extractions from Phase 6 Vision Engine.")
    
    # Dynamic Document Rendering
    if "GST_CERTIFICATE" in selected_doc:
        st.markdown(
            f"""
            <div style="border: 2px solid #2e7d32; padding: 18px; border-radius: 8px; background-color: #0b1410; color: #e0e0e0;">
                <div style="display:flex; justify-content:space-between; border-bottom: 1px solid #2e7d32; padding-bottom: 6px;">
                    <strong>GOVERNMENT OF INDIA - GST REGISTRATION CERTIFICATE</strong>
                    <span style="color: #81c784;">OCR Confidence: 97.2%</span>
                </div>
                <br>
                <p><strong>Registration Number (GSTIN):</strong> <mark style="background-color: #1565c0; color: white; padding: 2px 6px; border-radius: 4px;">{gstin_no}</mark> <span style="color:#aaa; font-size:12px;">[Coord: 110, 160, 390, 200]</span></p>
                <p><strong>Legal Name:</strong> <mark style="background-color: #2e7d32; color: white; padding: 2px 6px; border-radius: 4px;">{company_name}</mark> <span style="color:#aaa; font-size:12px;">[Coord: 110, 210, 430, 250]</span></p>
                <p><strong>Trade Name:</strong> Apex Technologies <span style="color:#aaa; font-size:12px;">[Coord: 110, 260, 310, 290]</span></p>
                <p><strong>Annual Taxable Turnover Reported:</strong> <strong>{gst_turnover_val}</strong> <span style="color:#aaa; font-size:12px;">[Coord: 110, 310, 280, 340]</span></p>
                <p><strong>Certificate Status:</strong> <span style="color: #4caf50;">ACTIVE</span> | <strong>Valid Till:</strong> 31/12/2026 <span style="color:#aaa; font-size:12px;">[Coord: 110, 360, 290, 390]</span> | State Code: 07 (Delhi)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    elif "PAN_CARD" in selected_doc:
        st.markdown(
            f"""
            <div style="border: 2px solid #1976d2; padding: 18px; border-radius: 8px; background-color: #0b121e; color: #e0e0e0;">
                <div style="display:flex; justify-content:space-between; border-bottom: 1px solid #1976d2; padding-bottom: 6px;">
                    <strong>INCOME TAX DEPARTMENT - GOVT OF INDIA</strong>
                    <span style="color: #64b5f6;">OCR Confidence: 95.8%</span>
                </div>
                <br>
                <p><strong>Permanent Account Number (PAN):</strong> <mark style="background-color: #e65100; color: white; padding: 2px 6px; border-radius: 4px;">{pan_no}</mark> <span style="color:#aaa; font-size:12px;">[Coord: 130, 150, 360, 190]</span></p>
                <p><strong>Entity / Cardholder Name:</strong> <mark style="background-color: #2e7d32; color: white; padding: 2px 6px; border-radius: 4px;">{company_name}</mark> <span style="color:#aaa; font-size:12px;">[Coord: 130, 200, 420, 240]</span></p>
                <p><strong>Date of Incorporation:</strong> 14/08/2016 <span style="color:#aaa; font-size:12px;">[Coord: 130, 250, 260, 280]</span></p>
                <p><strong>Category:</strong> Company / Registered Business Entity</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    else:  # Audited Balance Sheet
        st.markdown(
            f"""
            <div style="border: 2px solid #f57c00; padding: 18px; border-radius: 8px; background-color: #1a1309; color: #e0e0e0;">
                <div style="display:flex; justify-content:space-between; border-bottom: 1px solid #f57c00; padding-bottom: 6px;">
                    <strong>CA INDEPENDENT AUDIT REPORT & BALANCE SHEET</strong>
                    <span style="color: #ffb74d;">OCR Confidence: 94.6%</span>
                </div>
                <br>
                <p><strong>Audited Enterprise:</strong> <mark style="background-color: #2e7d32; color: white; padding: 2px 6px; border-radius: 4px;">{company_name}</mark> <span style="color:#aaa; font-size:12px;">[Coord: 95, 120, 410, 160]</span></p>
                <p><strong>Net Annual Audited Turnover:</strong> <mark style="background-color: #6a1b9a; color: white; padding: 2px 6px; border-radius: 4px;">{bs_turnover_val}</mark> <span style="color:#aaa; font-size:12px;">[Coord: 95, 180, 320, 220]</span></p>
                <p><strong>Statutory Auditor UDIN:</strong> 24089123AAAAAA9901 <span style="color:#aaa; font-size:12px;">[Coord: 95, 240, 380, 270]</span></p>
                <p><strong>Auditor Verification Stamp:</strong> <span style="color: #4caf50;">VERIFIED CHARTERED ACCOUNTANT SEAL</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_right:
    st.subheader("🔍 Automated Forensic & Compliance Scorecard")
    
    # KPI Metric Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Compliance Status", compliance_status)
    kpi2.metric("Discrepancies", f"{inconsistencies_count} Found")
    kpi3.metric("Assigned Risk", risk_rating)
    
    st.markdown("#### Lineage Verification Audit Trail")
    df_audit = pd.DataFrame(audit_data)
    st.dataframe(df_audit, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Officer Final Decision Panel
    st.subheader("⚖️ Officer Adjudication Panel")
    btn1, btn2, btn3 = st.columns(3)
    
    if btn1.button("✅ Approve Tender Bid", use_container_width=True):
        if inconsistencies_count > 0:
            st.error(f"❌ REJECTED BY SYSTEM: Cannot approve bid for {company_name}. Forensic cross-document conflicts detected!")
        else:
            st.success(f"✔ APPROVED: Bid from {company_name} meets all statutory and forensic checks.")
            
    if btn2.button("⚠️ Request Clarification", use_container_width=True):
        st.warning(f"Clarification memo generated for {company_name} regarding mismatched data fields.")
        
    if btn3.button("❌ Reject Submission", use_container_width=True):
        st.error(f"Submission from {company_name} rejected under statutory procurement rules.")