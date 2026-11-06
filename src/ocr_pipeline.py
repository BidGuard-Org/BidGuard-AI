import fitz  # PyMuPDF
import cv2
import numpy as np
import easyocr
import re
from typing import Dict, Any, List

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CONFUSABLE_MAP: Dict[str, List[str]] = {
    '0': ['O', 'D', 'Q'],
    'O': ['0', 'Q', 'D'],
    'D': ['0', 'O', 'Q'],
    'Q': ['0', 'O', 'D'],
    '1': ['I', 'L', 'T', '|'],
    'I': ['1', 'L', 'T', '|'],
    'L': ['1', 'I', '|'],
    'T': ['1', 'I'],
    '|': ['1', 'I', 'L'],
    '2': ['Z', '7'],
    'Z': ['2', '7'],
    '7': ['Z', '2', '1'],
    '5': ['S'],
    'S': ['5'],
    '8': ['B', '6'],
    'B': ['8', '6'],
    '6': ['G', 'B', '8'],
    'G': ['6', 'B'],
}

def compute_gstin_checksum(gstin14: str) -> str:
    """Calculates ISO 7064 Mod 36,10 (Luhn Mod 36) check digit for Indian GSTIN."""
    total = 0
    for i, c in enumerate(gstin14):
        val = CHARS.index(c)
        factor = 1 if (i % 2 == 0) else 2
        prod = val * factor
        total += (prod // 36) + (prod % 36)
    check_code = (36 - (total % 36)) % 36
    return CHARS[check_code]

def is_valid_gstin(gstin: str, verify_checksum: bool = True) -> bool:
    """
    Validates Indian GSTIN format and optional ISO 7064 Mod 36,10 check digit.
    State Code: 01-38, 97, 99
    PAN: 5 Alpha + 4 Digits + 1 Alpha
    Entity: 1 Alphanum (1-9, A-Z)
    Default: 'Z'
    Check Code: Alphanum
    """
    gstin = gstin.strip().upper()
    if not re.match(r"^(0[1-9]|[1-2][0-9]|3[0-8]|97|99)[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", gstin):
        return False
    if not verify_checksum:
        return True
    try:
        return gstin[14] == compute_gstin_checksum(gstin[:14])
    except Exception:
        return False

def validate_gstin_checksum(gstin: str) -> Dict[str, Any]:
    """
    Validates Indian GSTIN format and ISO 7064 Mod 36,10 check digit.

    Clearly reports structural format or checksum failures and explicitly notes
    that this is an offline mathematical check, NOT a government-verified failure
    or live portal rejection.
    """
    if not gstin or not isinstance(gstin, str):
        return {
            "gstin": gstin,
            "is_valid": False,
            "status": "MISSING_INPUT",
            "reason": "GSTIN was not provided or is empty.",
            "is_government_verified": False,
            "verification_type": "Offline ISO 7064 Mod 36,10 Checksum"
        }

    clean_gstin = gstin.strip().upper()
    if not re.match(r"^(0[1-9]|[1-2][0-9]|3[0-8]|97|99)[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", clean_gstin):
        return {
            "gstin": clean_gstin,
            "is_valid": False,
            "status": "INVALID_FORMAT",
            "reason": "GSTIN format violates Indian statutory structure (State: 01-38/97/99, 10-char PAN, 1-char entity, 'Z' default, 1 check char).",
            "is_government_verified": False,
            "verification_type": "Offline ISO 7064 Mod 36,10 Checksum"
        }

    expected_check = compute_gstin_checksum(clean_gstin[:14])
    actual_check = clean_gstin[14]

    if actual_check != expected_check:
        return {
            "gstin": clean_gstin,
            "is_valid": False,
            "status": "CHECKSUM_FAILED",
            "expected_check_digit": expected_check,
            "actual_check_digit": actual_check,
            "reason": f"Mathematical checksum failed: check digit is '{actual_check}', expected '{expected_check}' under ISO 7064 Mod 36,10. This is an algorithmic check, not a government portal rejection.",
            "is_government_verified": False,
            "verification_type": "Offline ISO 7064 Mod 36,10 Checksum"
        }

    return {
        "gstin": clean_gstin,
        "is_valid": True,
        "status": "VALID",
        "expected_check_digit": expected_check,
        "actual_check_digit": actual_check,
        "reason": "GSTIN format and ISO 7064 Mod 36,10 checksum mathematically verified.",
        "is_government_verified": False,
        "verification_type": "Offline ISO 7064 Mod 36,10 Checksum"
    }

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
        - Slot 14: Check Digit/Letter (Alphanumeric 0-9, A-Z)
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

        # 5. Entity Code -> Alphanumeric 1-9, A-Z (fix common OCR pipe/l characters)
        if candidate[12] in ['l', '|']: candidate[12] = '1'

        # 6. Statutory Slot 13 -> Under Indian Law, statutorily 'Z'
        # Repair common OCR misreads while preventing blind replacement of arbitrary characters
        if candidate[13] in ['2', '7', 'C', 'c', 'z', '€', 'Γé¼']:
            candidate[13] = 'Z'

        # 7. Slot 14 is the check character; preserve valid alphanumeric characters without blind overwrite

        return "".join(candidate)

    def repair_gstin_with_checksum_permutations(self, raw_candidate: str) -> str | None:
        """
        Multi-pass mathematical repair for low-DPI OCR scans.
        Explores confusable character permutations constrained by ISO 7064 Mod 36,10.
        """
        if not raw_candidate or not isinstance(raw_candidate, str):
            return None

        clean_raw = re.sub(r"[^A-Za-z0-9]", "", raw_candidate).upper()
        if len(clean_raw) != 15:
            return None

        # 1. Check if the string already passes is_valid_gstin(..., verify_checksum=True)
        if is_valid_gstin(clean_raw, verify_checksum=True):
            return clean_raw

        # Run standard structural slot repair
        cand = list(self.repair_gstin_slot(list(clean_raw)))
        if len(cand) != 15:
            return None
        cand_str = "".join(cand)
        if is_valid_gstin(cand_str, verify_checksum=True):
            return cand_str

        # 2. Tier 1: Check-digit confusable resolution on slot 14 against compute_gstin_checksum(cand[:14])
        try:
            expected_chk = compute_gstin_checksum("".join(cand[:14]))
            actual_chk = cand[14]
            if actual_chk == expected_chk or actual_chk in CONFUSABLE_MAP.get(expected_chk, []):
                tier1_cand = "".join(cand[:14]) + expected_chk
                if is_valid_gstin(tier1_cand, verify_checksum=True):
                    return tier1_cand
        except Exception:
            pass

        # 3. Tier 2: Single-position confusable substitution on cand[:14] that yields a valid ISO 7064 Mod 36,10 checksum
        base14 = list(cand[:14])
        for idx in range(14):
            curr_char = base14[idx]
            alternatives = CONFUSABLE_MAP.get(curr_char, [])
            for alt in alternatives:
                # Slot syntax constraints
                if idx in (0, 1) and not alt.isdigit():
                    continue
                if 2 <= idx <= 6 and not alt.isalpha():
                    continue
                if 7 <= idx <= 10 and not alt.isdigit():
                    continue
                if idx == 11 and not alt.isalpha():
                    continue
                if idx == 13 and alt != 'Z':
                    continue

                trial = list(base14)
                trial[idx] = alt
                trial_prefix = "".join(trial)
                try:
                    chk = compute_gstin_checksum(trial_prefix)
                    if chk == cand[14] or cand[14] in CONFUSABLE_MAP.get(chk, []):
                        trial_cand = f"{trial_prefix}{chk}"
                        if is_valid_gstin(trial_cand, verify_checksum=True):
                            return trial_cand
                except Exception:
                    continue

        # 4. Tier 3: Non-destructive fallback — if checksum cannot be satisfied, return structurally valid 15-character string rather than dropping to None
        if is_valid_gstin(cand_str, verify_checksum=False):
            return cand_str

        return None

    def find_gstin(self, text: str, verify_checksum: bool = True) -> str | None:
        """
        Space-agnostic, symbol-mapped sliding window extractor for Indian GSTINs.
        """
        # Step 1: Normalize symbols before alphanumeric stripping
        symbol_map = {
            '€': 'Z',
            'Γé¼': 'Z',
            '£': 'E',
            '┬ú': 'E',
            '$': 'S',
            '@': 'A',
            '©': 'C',
            '┬⌐': 'C',
            '®': 'R',
            '┬«': 'R'
        }
        for sym, rep in symbol_map.items():
            text = text.replace(sym, rep)

        # Step 2: Strip all non-alphanumeric chars (removes spaces, colons, hyphens)
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()

        format_valid_fallback = None

        # Step 3: Slide 15-character window across continuous token stream
        for i in range(len(cleaned) - 14):
            raw_window = list(cleaned[i:i+15])
            raw_candidate = "".join(raw_window)
            # Preserve exact extracted value if already valid
            if is_valid_gstin(raw_candidate, verify_checksum=verify_checksum):
                return raw_candidate

            repaired = self.repair_gstin_with_checksum_permutations(raw_candidate)
            if repaired:
                if is_valid_gstin(repaired, verify_checksum=True):
                    return repaired
                elif format_valid_fallback is None and is_valid_gstin(repaired, verify_checksum=False):
                    format_valid_fallback = repaired

        # Fallback to format-only regex match only when verify_checksum is explicitly False
        if not verify_checksum:
            for i in range(len(cleaned) - 14):
                raw_window = list(cleaned[i:i+15])
                raw_candidate = "".join(raw_window)
                if is_valid_gstin(raw_candidate, verify_checksum=False):
                    return raw_candidate
                repaired = self.repair_gstin_with_checksum_permutations(raw_candidate)
                if repaired and is_valid_gstin(repaired, verify_checksum=False):
                    return repaired
            return format_valid_fallback

        return None

    def derive_pan_from_gstin(self, gstin: str | None) -> str | None:
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
            page_text = " ".join(str(result) for result in ocr_results)
            extracted_text_blocks.append(page_text)

        full_document_text = " ".join(extracted_text_blocks)
        gstin = self.find_gstin(full_document_text, verify_checksum=True)
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
