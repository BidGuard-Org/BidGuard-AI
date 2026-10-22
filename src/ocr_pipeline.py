import fitz  # PyMuPDF
import cv2
import numpy as np
import easyocr
import re
from typing import Dict, Any, List

class DocumentOCRPipeline:
    def __init__(self, languages: List[str] = ['en']):
        print("[INFO] Initializing Vision OCR Engine (EasyOCR on CPU)...")
        self.reader = easyocr.Reader(languages, gpu=False)
        print("[INFO] EasyOCR loaded successfully!")

    def extract_from_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """Converts each page of a PDF into high-res BGR numpy images using PyMuPDF."""
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)
        return images

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive document preprocessing:
        1. Downscales massive phone camera scans (> 1800px) safely to prevent memory blowups.
        2. Bilateral filter preserves sharp handwriting/font edges while suppressing paper texture.
        """
        h, w = img.shape[:2]
        max_dim = 1800
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        smoothed = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        return smoothed

    def repair_gstin_slot(self, candidate: List[str]) -> str:
        """
        Reconciles real-world OCR and handwriting confusions to conform
        to Indian Statutory GSTIN Structure:
        - Slots 0,1: State Code (01-37)
        - Slots 2-6: PAN Alpha Prefix
        - Slots 7-10: PAN Numeric Sequence
        - Slot 11: PAN Alpha Suffix
        - Slot 12: Entity Count (usually 1-9)
        - Slot 13: Fixed Statutory Default ('Z')
        - Slot 14: Check Digit/Letter
        """
        # 1. State Code -> Digits
        for idx in [0, 1]:
            if candidate[idx] in ['O', 'D', 'Q', 'o']: candidate[idx] = '0'
            elif candidate[idx] in ['I', 'L', 'T', 'l', '|']: candidate[idx] = '1'
            elif candidate[idx] == 'Z': candidate[idx] = '2'

        # 2. PAN Alpha Prefix -> Alphabets
        for idx in range(2, 7):
            if candidate[idx] in ['0', 'O', 'o']: candidate[idx] = 'O'
            elif candidate[idx] in ['1', 'I', 'l', '|']: candidate[idx] = 'I'
            elif candidate[idx] == '5': candidate[idx] = 'S'
            elif candidate[idx] in ['8', '6']: candidate[idx] = 'B'

        # 3. PAN Numbers -> Digits
        for idx in range(7, 11):
            if candidate[idx] in ['O', 'D', 'Q', 'o']: candidate[idx] = '0'
            elif candidate[idx] in ['I', 'L', 'T', 'l', '|']: candidate[idx] = '1'
            elif candidate[idx] == 'S': candidate[idx] = '5'
            elif candidate[idx] == 'B': candidate[idx] = '8'

        # 4. PAN Suffix -> Alphabet
        if candidate[11] in ['6', '8', 'B']: candidate[11] = 'B'
        elif candidate[11] in ['0', 'O', 'o']: candidate[11] = 'O'
        elif candidate[11] in ['1', 'I']: candidate[11] = 'I'
        elif candidate[11] == '5': candidate[11] = 'S'

        # 5. Entity Code -> Usually digit '1'
        if candidate[12] in ['I', 'L', 'T', 'l', '|']: candidate[12] = '1'

        # 6. Fixed Statutory Slot 13 -> By Indian Law, always 'Z'
        # Fixes common OCR misreads: '€' (Euro), 'C', '2', '7'
        candidate[13] = 'Z'

        # 7. Check Code (Slot 14)
        if candidate[14] == 'S': candidate[14] = '5'

        return "".join(candidate)

    def find_gstin(self, text: str) -> str:
        """
        Space-agnostic, symbol-mapped sliding window extractor for Indian GSTINs.
        """
        # Step 1: Normalize symbols before alphanumeric stripping
        symbol_map = {
            '€': 'Z',
            '£': 'E',
            '$': 'S',
            '@': 'A',
            '©': 'C',
            '®': 'R'
        }
        for sym, rep in symbol_map.items():
            text = text.replace(sym, rep)

        # Step 2: Strip all non-alphanumeric chars (removes spaces, colons, hyphens)
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()

        # Step 3: Slide 15-character window across continuous token stream
        for i in range(len(cleaned) - 14):
            raw_window = list(cleaned[i:i+15])
            repaired = self.repair_gstin_slot(raw_window)

            # Strict Indian GSTIN validation regex:
            # State: 01-37 | PAN: 5 Alpha + 4 Digit + 1 Alpha | Entity: 1 Alphanum | Fixed 'Z' | Check Alphanum
            if re.match(r"^[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", repaired):
                return repaired

        return None

    def derive_pan_from_gstin(self, gstin: str) -> str:
        if gstin and len(gstin) == 15:
            return gstin[2:12]
        return None

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Unified ingestion pipeline supporting images (.png, .jpg, .jpeg) and PDFs."""
        if not file_path:
            raise ValueError("No file path provided.")

        is_pdf = file_path.lower().endswith(".pdf")
        if is_pdf:
            pages = self.extract_from_pdf(file_path)
            if not pages:
                raise ValueError(f"Unable to extract pages from PDF: {file_path}")
            images_to_process = pages
        else:
            img = cv2.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image file: {file_path}")
            images_to_process = [img]

        extracted_text_blocks = []

        for raw_img in images_to_process:
            processed_img = self.preprocess_image(raw_img)
            ocr_results = self.reader.readtext(processed_img, detail=0)
            page_text = " ".join(ocr_results)
            extracted_text_blocks.append(page_text)

        full_document_text = " ".join(extracted_text_blocks)
        gstin = self.find_gstin(full_document_text)
        pan = self.derive_pan_from_gstin(gstin)

        return {
            "document_id": file_path.split("/")[-1].split("\\")[-1],
            "total_pages": len(images_to_process),
            "extracted_fields": {
                "gstin": gstin,
                "pan": pan,
                "detected_date": None,
                "raw_text_snippet": full_document_text[:300]
            }
        }