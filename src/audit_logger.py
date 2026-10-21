import json
import os
from datetime import datetime

# Audit logs ko store karne ke liye folder aur file path
AUDIT_LOG_DIR = "reports/audit_logs"
AUDIT_LOG_FILE = os.path.join(AUDIT_LOG_DIR, "adjudication_audit_trail.json")

def ensure_audit_store():
    """Ensure audit log folder and JSON file exist."""
    if not os.path.exists(AUDIT_LOG_DIR):
        os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
    if not os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def log_adjudication_decision(tender_id: str, bidder_id: str, action: str, remarks: str, lineage_data: list) -> str:
    """
    Officer ke decision aur exact bounding box lineage ko persistent JSON record me save karta hai.
    """
    ensure_audit_store()
    
    event_id = f"AUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    audit_record = {
        "event_id": event_id,
        "timestamp": datetime.now().isoformat(),
        "tender_id": tender_id,
        "bidder_id": bidder_id,
        "adjudication": {
            "officer_role": "Procurement Officer",
            "decision": action,
            "remarks": remarks
        },
        "lineage_evidence": lineage_data
    }
    
    records = []
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            records = []
            
    records.append(audit_record)
    
    with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
        
    return event_id

def get_all_audit_logs():
    """Dashboard par audit history dikhane ke liye logs fetch karta hai."""
    ensure_audit_store()
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []