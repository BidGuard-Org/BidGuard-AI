import os
import re
import cv2
import fitz  # PyMuPDF
import numpy as np
import easyocr


class DocumentOCRPipeline:
    def __init__(self, languages=['en'], use_gpu=False):
        """
        Phase 6: Multi-Format Vision & OCR Pipeline (Handles both Images and PDFs).
        """
        print("[INFO] Initializing Vision OCR Engine (EasyOCR on CPU)...")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        print("[INFO] EasyOCR loaded successfully!")

    def _load_image_or_pdf(self, file_path, page_number=1):
        """
        Accepts either an image (.png, .jpg) or a multi-page PDF (.pdf)
        and converts it into an OpenCV BGR numpy image.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Case 1: If input is a PDF
        if ext == ".pdf":
            doc = fitz.open(file_path)
            if page_number > len(doc) or page_number < 1:
                page_number = 1

            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=200)
            img_bytes = np.frombuffer(pix.samples, dtype=np.uint8)

            if pix.alpha:
                img = img_bytes.reshape(pix.height, pix.width, 4)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = img_bytes.reshape(pix.height, pix.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        # Case 2: Standard Image
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Image nahi mili: {file_path}")
        return img

    def _extract_key_entities(self, text):
        """
        Extracts key statutory entities like GSTIN, PAN, and Dates using regex
        with automatic character-confusion normalization.
        """
        # Strict pattern pehle check karein
        gstin_pattern = r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b'
        pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
        date_pattern = r'\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b'

        gstin_match = re.search(gstin_pattern, text)
        pan_match = re.search(pan_pattern, text)
        date_match = re.search(date_pattern, text)

        # Fallback: Agar "GSTIN:" label ke aage 15-char code hai jo misread hua hai
        extracted_gstin = gstin_match.group(0) if gstin_match else None
        if not extracted_gstin:
            label_match = re.search(r'GSTIN[:\s\-]+([A-Z0-9]{15})', text, re.IGNORECASE)
            if label_match:
                candidate = list(label_match.group(1).upper())
                # Normalize digits where letters were misread:
                # GSTIN format: 2 Digits + 5 Letters + 4 Digits + 1 Letter + 1 Alphanum + 'Z' + 1 Checksum
                replacements = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'T': '1', 'S': '5', 'B': '8', 'Z': '2'}
                
                # State code (index 0, 1) must be digits
                for idx in [0, 1]:
                    if candidate[idx] in replacements:
                        candidate[idx] = replacements[candidate[idx]]
                
                # Entity PAN digits (index 7, 8, 9, 10) must be digits
                for idx in [7, 8, 9, 10]:
                    if candidate[idx] in replacements:
                        candidate[idx] = replacements[candidate[idx]]

                # 14th character is almost always 'Z'
                candidate[13] = 'Z'
                
                normalized = "".join(candidate)
                if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}', normalized):
                    extracted_gstin = normalized

        # Fallback: Extract PAN directly from GSTIN (chars 3 to 12)
        extracted_pan = pan_match.group(0) if pan_match else None
        if not extracted_pan and extracted_gstin:
            extracted_pan = extracted_gstin[2:12]

        return {
            "gstin": extracted_gstin,
            "pan": extracted_pan,
            "date": date_match.group(0) if date_match else None
        }

    def process_document(self, file_path, doc_id="DOC-0001", doc_type="STATUTORY_CERTIFICATE", page_number=1, mock_text=""):
        """
        Core OCR Pipeline: Processes Image or PDF and returns Contract 1 compliant JSON schema.
        """
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File does not exist: {file_path}"}

        # 1. Image ya PDF load karein
        bgr_img = self._load_image_or_pdf(file_path, page_number=page_number)

        # 2. Real EasyOCR Execution
        results = self.reader.readtext(bgr_img)
        text_lines = []
        all_detections = []
        primary_box = [100, 150, 420, 210]

        for i, (bbox, text, prob) in enumerate(results):
            text_lines.append(text)
            (tl, tr, br, bl) = bbox
            x = int(tl[0])
            y = int(tl[1])
            w = int(br[0] - tl[0])
            h = int(br[1] - tl[1])

            all_detections.append({
                "region_id": i + 1,
                "text": text,
                "bounding_box": [x, y, w, h],
                "confidence": round(float(prob), 3)
            })

        extracted_text = " ".join(text_lines)
        if all_detections:
            primary_box = all_detections[0]["bounding_box"]

        # 3. Statutory Entity Extraction (Regex)
        entities = self._extract_key_entities(extracted_text)

        # 4. Contract 1 Compliant Schema
        avg_conf = round(sum(d["confidence"] for d in all_detections) / len(all_detections), 3) if all_detections else 0.95

        return {
            "document_id": doc_id,
            "document_type": doc_type,
            "page_number": page_number,
            "extracted_fields": {
                "gstin": entities.get("gstin"),
                "pan": entities.get("pan"),
                "detected_date": entities.get("date"),
                "raw_text_snippet": extracted_text[:300]
            },
            "ocr_confidence": avg_conf,
            "bounding_box": primary_box,
            "all_detections": all_detections
        }
