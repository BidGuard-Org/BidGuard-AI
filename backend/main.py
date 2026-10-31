
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from src.xai_explainer import explain_bid

import tempfile
from src.ocr_pipeline import DocumentOCRPipeline

from backend.services.mock_government_apis import verify_registration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

app = FastAPI(
    title="BidGuard AI API",
    description="Synthetic bid compliance research API. Not connected to government portals.",
    version="1.0.0",
)


class RegistrationRequest(BaseModel):
    registry: str = Field(min_length=1, max_length=20)
    registration_number: str = Field(min_length=1, max_length=100)
    document_id: str | None = None

class RiskPredictionRequest(BaseModel):
    features: dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok", "service": "BidGuard AI", "synthetic_data": True}


@app.post("/verify-registration")
def verify_registration_endpoint(request: RegistrationRequest):
    result = verify_registration(
        request.registry,
        request.registration_number,
        request.document_id,
    )

    if result["status"] == "UNSUPPORTED_REGISTRY":
        raise HTTPException(status_code=400, detail=result)

    return result


@app.get("/get-audit-trail/{bidder_id}")
def get_audit_trail(bidder_id: str):
    path = DATA_DIR / "audit_trail.csv"

    if not path.exists():
        raise HTTPException(status_code=503, detail="Audit data unavailable")

    df = pd.read_csv(path, dtype=str).fillna("")
    result = df[df["bidder_id"] == bidder_id]

    return {
        "bidder_id": bidder_id,
        "count": len(result),
        "records": result.to_dict(orient="records"),
        "synthetic": True,
    }


@app.get("/compliance/{bidder_id}")
def get_compliance(bidder_id: str):
    path = DATA_DIR / "compliance_results.csv"

    if not path.exists():
        raise HTTPException(status_code=503, detail="Compliance data unavailable")

    df = pd.read_csv(path, dtype=str).fillna("")
    result = df[df["bidder_id"] == bidder_id]

    return {
        "bidder_id": bidder_id,
        "count": len(result),
        "records": result.to_dict(orient="records"),
        "synthetic": True,
    }


@app.post("/predict-risk")
def predict_risk(request: RiskPredictionRequest):
    try:
        result = explain_bid(request.features)
        return {
            "success": True,
            "prediction": result,
            "synthetic_data": True,
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/extract-ocr")
async def extract_ocr(file: UploadFile = File(...)):
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, PNG, JPG and JPEG files are supported.",
        )

    temp_path = None

    try:
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        pipeline = DocumentOCRPipeline()
        result = pipeline.process_document(temp_path)

        result["document_id"] = Path(filename).name
        result["synthetic_data"] = False
        return result

    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
        await file.close()
