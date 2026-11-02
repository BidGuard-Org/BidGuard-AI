"""Universal Vision Preprocessor for Indian Statutory & Identity Documents.

Provides adaptive contrast enhancement (CLAHE), orientation correction / deskewing,
bilateral edge-preserving denoising, and high-resolution PDF rendering for EasyOCR.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Any
import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class DocumentPreprocessor:
    """Adaptive image and PDF preprocessor designed for diverse document qualities."""

    @staticmethod
    def resize_normalize(img: np.ndarray, target_min_dim: int = 1000, max_dim: int = 2200) -> np.ndarray:
        """
        Normalizes image dimensions:
        - Scales up small or compressed images (< 1000px) so fine text characters are legible to OCR.
        - Scales down massive phone camera captures (> 2200px) to prevent memory bottlenecks.
        """
        h, w = img.shape[:2]
        longest = max(h, w)
        shortest = min(h, w)

        if longest > max_dim:
            scale = max_dim / float(longest)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        elif shortest < target_min_dim:
            scale = target_min_dim / float(shortest)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        return img

    @staticmethod
    def enhance_contrast_clahe(img: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Compensates for uneven mobile flash, poor illumination, shadows, and faded print.
        """
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(img)

    @staticmethod
    def denoise_bilateral(img: np.ndarray) -> np.ndarray:
        """
        Bilateral filtering preserves text edge sharpness while removing background noise,
        paper grain, and compression artifacts.
        """
        return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=40)

    @staticmethod
    def estimate_skew_angle(gray_img: np.ndarray) -> float:
        """
        Estimates skew angle of text blocks using morphological line detection and minAreaRect.
        Returns angle in degrees (-45.0 to 45.0).
        """
        try:
            # Otsu thresholding after Gaussian blur
            blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Dilate text lines horizontally
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=2)

            # Find text contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            angles = []

            for c in contours:
                if cv2.contourArea(c) > 200:
                    rect = cv2.minAreaRect(c)
                    angle = rect[-1]
                    # Normalize angle
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                    if abs(angle) < 45:
                        angles.append(angle)

            if angles:
                median_angle = float(np.median(angles))
                return median_angle if abs(median_angle) > 0.5 else 0.0
        except Exception:
            pass

        return 0.0

    @classmethod
    def deskew_image(cls, img: np.ndarray) -> np.ndarray:
        """Rotates image to correct detected angular skew."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        angle = cls.estimate_skew_angle(gray)

        if abs(angle) > 0.75:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Recompute bounding dimensions to avoid cutting off rotated corners
            cos = np.abs(rot_mat[0, 0])
            sin = np.abs(rot_mat[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            rot_mat[0, 2] += (new_w / 2) - center[0]
            rot_mat[1, 2] += (new_h / 2) - center[1]
            return cv2.warpAffine(img, rot_mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return img

    @staticmethod
    def load_image_with_exif(source: Any) -> Optional[np.ndarray]:
        """
        Loads an image from disk while respecting and correcting for EXIF orientation metadata.
        Supports file path (str), PIL.Image, or raw bytes.
        Corrects rotated mobile camera captures (90°, 180°, 270°) before feeding to OCR.
        Falls back to cv2.imread if PIL is unavailable or fails.
        """
        try:
            from PIL import Image, ImageOps
            import io
            if isinstance(source, Image.Image):
                transposed = ImageOps.exif_transpose(source)
                if transposed.mode != "RGB":
                    transposed = transposed.convert("RGB")
                np_img = np.array(transposed)
                return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            elif isinstance(source, (bytes, bytearray)):
                with Image.open(io.BytesIO(source)) as pil_img:
                    transposed = ImageOps.exif_transpose(pil_img)
                    if transposed.mode != "RGB":
                        transposed = transposed.convert("RGB")
                    np_img = np.array(transposed)
                    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            else:
                with Image.open(source) as pil_img:
                    transposed = ImageOps.exif_transpose(pil_img)
                    if transposed.mode != "RGB":
                        transposed = transposed.convert("RGB")
                    np_img = np.array(transposed)
                    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        except Exception:
            try:
                if isinstance(source, str):
                    return cv2.imread(source)
                return None
            except Exception:
                return None

    @staticmethod
    def rotate_image(img: np.ndarray, angle_degrees: int) -> np.ndarray:
        """
        Rotates an image by exact statutory angles (90°, 180°, 270° clockwise).
        Leaves image unchanged if angle is 0 or unsupported.
        """
        if angle_degrees == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle_degrees == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle_degrees == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @classmethod
    def preprocess_image_pipeline(cls, img: np.ndarray) -> np.ndarray:
        """
        Executes end-to-end adaptive preprocessing:
        1. Dimension normalization (scaling low-res or downscaling high-res).
        2. Orientation / deskewing.
        3. CLAHE contrast adjustment.
        4. Bilateral edge-preserving smoothing.
        """
        img_norm = cls.resize_normalize(img)
        img_deskewed = cls.deskew_image(img_norm)
        img_enhanced = cls.enhance_contrast_clahe(img_deskewed)
        img_denoised = cls.denoise_bilateral(img_enhanced)
        return img_denoised

    @classmethod
    def render_pdf_page_to_image(cls, pdf_path: str, page_idx: int = 0, dpi: int = 300, page_number: Optional[int] = None) -> Optional[np.ndarray]:
        """Renders an individual PDF page into a BGR numpy array using PyMuPDF with safe resource cleanup."""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF rendering.")

        idx = page_number if page_number is not None else page_idx
        doc = fitz.open(pdf_path)
        try:
            if idx < 0 or idx >= len(doc):
                return None
            page = doc[idx]
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        finally:
            doc.close()

    @classmethod
    def render_pdf_to_images(cls, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """Converts all PDF pages into BGR numpy arrays using PyMuPDF with guaranteed handle cleanup."""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF rendering.")

        doc = fitz.open(pdf_path)
        pages = []
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    pages.append(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
                elif pix.n == 3:
                    pages.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    pages.append(img)
            return pages
        finally:
            doc.close()


# Module-level convenience functions
load_image_with_exif = DocumentPreprocessor.load_image_with_exif
rotate_image = DocumentPreprocessor.rotate_image
render_pdf_page_to_image = DocumentPreprocessor.render_pdf_page_to_image
render_pdf_to_images = DocumentPreprocessor.render_pdf_to_images
