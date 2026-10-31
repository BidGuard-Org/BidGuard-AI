import os
from src.ocr_pipeline import DocumentOCRPipeline

image_path = "sample1_test.jpeg"

print(f"[1] Checking image file: {image_path}")
if not os.path.exists(image_path):
    print(f"[ERROR] '{image_path}' file nahi mili! Root folder mein check karein.")
    exit(1)

print("[2] Initializing EasyOCR Pipeline (Phase 6)...")
ocr = DocumentOCRPipeline()

print("[3] Processing document...")
result = ocr.process_document(image_path)

print("\n" + "="*40)
print("PHASE 6 TEST RESULT")
print("="*40)
print(f"Document ID: {result.get('document_id')}")
print(f"Extracted Fields:")
for k, v in result.get('extracted_fields', {}).items():
    print(f"  - {k}: {v}")
print("="*40)
