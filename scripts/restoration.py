"""
scripts/restoration.py: Classical Restoration Modules
=====================================================
Contains modular classical enhancement algorithms:
  - Non-Local Means (NLM) Denoising
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)
  - Unsharp Masking & Edge Sharpening
  - Morphological Residual Re-Inking
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class NLMDenoisingRestoration:
    """
    Restoration Module: Non-Local Means (NLM) Denoising.
    Filters high-frequency sensor noise while preserving structural plate edges.
    """
    def __init__(self, h: int = 4, template_window: int = 7, search_window: int = 15):
        self.h = h
        self.template_window = template_window
        self.search_window = search_window

    def apply(self, l_channel: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            l_channel, None, h=self.h,
            templateWindowSize=self.template_window,
            searchWindowSize=self.search_window
        )


class CLAHERestoration:
    """
    Restoration Module: Contrast Limited Adaptive Histogram Equalization.
    Enhances local luminance dynamic range while clipping over-amplification.
    """
    def __init__(self, clip_limit: float = 1.8, tile_grid_size: Tuple[int, int] = (4, 4)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def apply(self, l_channel: np.ndarray) -> np.ndarray:
        return self.clahe.apply(l_channel)


class UnsharpMaskSharpeningRestoration:
    """
    Restoration Module: Unsharp Masking & Edge Enhancement.
    Applies Gaussian blurring and weighted subtraction to sharpen character strokes.
    """
    def __init__(self, sigma: float = 0.8, weight_orig: float = 1.10, weight_blur: float = -0.10):
        self.sigma = sigma
        self.weight_orig = weight_orig
        self.weight_blur = weight_blur

    def apply(self, l_channel: np.ndarray) -> np.ndarray:
        blur_s = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=self.sigma)
        return cv2.addWeighted(l_channel, self.weight_orig, blur_s, self.weight_blur, 0)


class MorphologicalReinkingRestoration:
    """
    Restoration Module: Adaptive Morphological Residual Re-Inking.
    Estimates plate background via morphological closing and re-inks
    faded character strokes using difference residuals.
    """
    @staticmethod
    def _make_odd(n: int) -> int:
        return n if n % 2 != 0 else n + 1

    def apply(self, l_channel: np.ndarray, height: int, reink_factor: float = 0.40) -> np.ndarray:
        k_dim = self._make_odd(int(np.clip(0.35 * height, 5, 17)))
        k_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dim, k_dim))
        bg = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, k_bg)
        diff = np.maximum(bg.astype(np.float32) - l_channel.astype(np.float32), 0.0)
        p99_diff = np.percentile(diff, 99)
        diff_norm = np.clip((diff / max(1e-5, p99_diff)) * 255.0, 0, 255)
        re_inked = l_channel.astype(np.float32) - reink_factor * diff_norm
        p2, p98 = np.percentile(re_inked, (2, 98))
        return np.clip((re_inked - p2) / max(1e-5, (p98 - p2)) * 255.0, 0, 255).astype(np.uint8)


class ClassicalRestorationSuite:
    """
    Unified manager bundling all separate classical restoration modules.
    Provides utility methods to apply, ablate, or individually test each filter.
    """
    def __init__(
        self,
        nlm: Optional[NLMDenoisingRestoration] = None,
        clahe: Optional[CLAHERestoration] = None,
        sharpen: Optional[UnsharpMaskSharpeningRestoration] = None,
        morphology: Optional[MorphologicalReinkingRestoration] = None
    ):
        self.nlm = nlm or NLMDenoisingRestoration()
        self.clahe = clahe or CLAHERestoration()
        self.sharpen = sharpen or UnsharpMaskSharpeningRestoration()
        self.morphology = morphology or MorphologicalReinkingRestoration()

    @staticmethod
    def normalize_percentile(channel: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
        p1, p99 = np.percentile(channel, (p_low, p_high))
        return np.clip((channel - p1) / max(1e-5, (p99 - p1)) * 255.0, 0, 255).astype(np.uint8)


make_odd = MorphologicalReinkingRestoration._make_odd

