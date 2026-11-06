"""
scripts/ocr_recognition.py: OCR Recognition Module (PaddleOCR / RapidOCR)
========================================================================
Encapsulates OCR transcription via RapidOCR / PaddleOCR engine.
Supports end-to-end detection+recognition or recognition-only mode.
Cleanses non-alphanumeric noise and classifies plate transcription integrity.
"""

from typing import Tuple, Optional, Any
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


class DualPathPaddleOCREngine:
    """
    OCR Recognition Module:
    Encapsulates OCR transcription via RapidOCR / PaddleOCR engine.
    Supports end-to-end detection+recognition or recognition-only mode.
    Cleanses non-alphanumeric noise and classifies plate transcription integrity.
    """
    def __init__(self, engine: Optional[Any] = None):
        if engine is not None:
            self.engine = engine
            print("[✓] [OCRRecognitionModule] Using preloaded RapidOCR Engine.")
        else:
            if RapidOCR is None:
                raise ImportError(
                    "[OCRRecognitionModule] 'rapidocr_onnxruntime' is not installed in the current environment. "
                    "Install it via 'pip install rapidocr_onnxruntime' or pass a preloaded engine."
                )
            print("[*] [OCRRecognitionModule] Initializing RapidOCR/PaddleOCR Engine...")
            self.engine = RapidOCR()
            print("[✓] [OCRRecognitionModule] OCR Engine loaded.")

    def transcribe(self, crop_bgr: np.ndarray, recognition_only: bool = False) -> Tuple[str, str, float]:
        """
        Transcribes text from a plate crop image.
        Returns:
            Tuple of (clean_text, status, avg_confidence).
            Statuses: COMPLETE_PLATE (4-12 chars), PARTIAL_RECOGNITION (1-3 chars),
                      REJECTED_LENGTH, EMPTY, EXCEPTION.
        """
        if not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return "NOT_DETECTED", "EMPTY", 0.0

        try:
            result, _ = self.engine(crop_bgr, use_det=(not recognition_only))
            if not result:
                return "NOT_DETECTED", "EMPTY", 0.0

            tokens, confs = [], []
            for item in result:
                raw_token = item[1].strip()
                clean_token = "".join([c for c in raw_token if c.isalnum()]).upper()
                if clean_token:
                    tokens.append(clean_token)
                    confs.append(float(item[2]))

            raw_candidate = "".join(tokens)
            clean_text = "".join([c for c in raw_candidate.strip().upper() if c.isalnum()])
            avg_conf = float(np.mean(confs)) if confs else 0.0

            if 4 <= len(clean_text) <= 12:
                return clean_text, "COMPLETE_PLATE", round(avg_conf, 3)
            elif 1 <= len(clean_text) < 4:
                return clean_text, "PARTIAL_RECOGNITION", round(avg_conf, 3)
            return "NOT_DETECTED", "REJECTED_LENGTH", 0.0
        except Exception:
            return "OCR_FAILED", "EXCEPTION", 0.0

    recognize = transcribe

