# BidGuard AI — Synthetic Demonstration Dossiers
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
