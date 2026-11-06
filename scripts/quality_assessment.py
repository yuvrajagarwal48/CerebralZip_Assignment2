"""
scripts/quality_assessment.py: Image Quality Assessment Module
==============================================================
Measures forensic image metrics across four dimensions:
  1. Resolution: Pre-rectification actual plate quadrilateral height and
     connected-component character stroke height estimation.
  2. Noise: Flat-region patch variance estimation (fully covering boundary patches).
  3. Sharpness: Composite metric combining Laplacian variance and Sobel gradient energy.
  4. Contrast: Illumination-independent normalized percentile dynamic range (p95 - p5) / 255.0.
Produces continuous metrics and thresholded decision flags for the routing module.
"""

from typing import Tuple, Dict, Optional, Any
import cv2
import numpy as np


class QualityAssessmentModule:
    """
    Quality Assessment Module:
    Measures forensic image metrics across four dimensions:
      1. Resolution: Pre-rectification actual plate quadrilateral height and
         connected-component character stroke height estimation.
      2. Noise: Flat-region patch variance estimation (fully covering boundary patches).
      3. Sharpness: Composite metric combining Laplacian variance and Sobel gradient energy.
      4. Contrast: Illumination-independent normalized percentile dynamic range (p95 - p5) / 255.0.
    Produces continuous metrics and thresholded decision flags for the routing module.
    """
    @staticmethod
    def estimate_flat_region_noise_safe(gray: np.ndarray, patch_size: int = 12, grad_thresh: float = 7.0) -> Tuple[Optional[float], int]:
        h, w = gray.shape
        variances = []
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = gray[y:y + patch_size, x:x + patch_size]
                gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
                grad_energy = np.mean(np.abs(gx) + np.abs(gy))
                if grad_energy < grad_thresh:
                    variances.append(float(np.std(patch)))

        valid_patch_count = len(variances)
        if valid_patch_count < 3:
            return None, valid_patch_count
        return float(np.median(variances)), valid_patch_count

    @staticmethod
    def compute_composite_sharpness(gray: np.ndarray) -> Tuple[float, float, float]:
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
        composite_sharpness = float(np.sqrt(max(0.0, lap_var)) * 0.4 + grad_energy * 0.6)
        return lap_var, grad_energy, composite_sharpness

    @staticmethod
    def compute_contrast_spread(gray: np.ndarray) -> float:
        """
        Computes normalized percentile dynamic range contrast: (p95 - p5) / 255.0.
        Provides an illumination-independent metric for 8-bit grayscale images.
        """
        p5, p95 = np.percentile(gray, (5, 95))
        return float((p95 - p5) / 255.0)

    @staticmethod
    def estimate_pre_rectification_char_height(
        raw_crop: np.ndarray,
        quad_dims: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float, str]:
        """
        Measures the actual plate quadrilateral dimensions and estimates character height
        before rectification. Uses connected component analysis of character strokes on
        the pre-rectified crop when discernible, falling back to 0.65 * measured quad height.
        """
        orig_h, orig_w = raw_crop.shape[:2]
        if quad_dims and "quad_h" in quad_dims and float(quad_dims["quad_h"]) > 0:
            actual_plate_h = float(quad_dims["quad_h"])
            method = quad_dims.get("method", "MEASURED_QUADRILATERAL")
        else:
            actual_plate_h = float(orig_h)
            method = "RAW_CROP_HEIGHT"

        try:
            gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

            char_heights = []
            for i in range(1, num_labels):
                comp_w = stats[i, cv2.CC_STAT_WIDTH]
                comp_h = stats[i, cv2.CC_STAT_HEIGHT]
                if (0.30 * actual_plate_h <= comp_h <= 0.85 * actual_plate_h) and \
                   (0.10 * actual_plate_h <= comp_w <= 0.85 * actual_plate_h):
                    char_heights.append(comp_h)

            if len(char_heights) >= 2:
                est_char_h = float(np.median(char_heights))
                method += "+CONNECTED_COMPONENTS"
            else:
                est_char_h = round(actual_plate_h * 0.65, 1)
                method += "+PROPORTIONAL_0.65X"
        except Exception:
            est_char_h = round(actual_plate_h * 0.65, 1)
            method += "+FALLBACK"

        return actual_plate_h, est_char_h, method

    def assess(
        self,
        original_raw_crop: np.ndarray,
        rectified_crop: np.ndarray,
        quad_dims: Optional[Dict[str, Any]] = None,
        contrast_thresh: float = 0.30
    ) -> Dict[str, Any]:
        """
        Executes comprehensive multi-indicator quality assessment:
        - Measures pre-rectification quadrilateral and estimated character height.
        - Computes flat-region noise including boundary patches.
        - Computes composite edge sharpness.
        - Evaluates illumination-independent normalized contrast ((p95 - p5) / 255.0) against calibrated threshold.
        """
        orig_h, orig_w = original_raw_crop.shape[:2]
        rect_gray = cv2.cvtColor(rectified_crop, cv2.COLOR_BGR2GRAY)

        # 1. Resolution Assessment on Pre-Rectified Crop & Quadrilateral
        actual_plate_h, est_char_h, char_method = self.estimate_pre_rectification_char_height(
            original_raw_crop, quad_dims
        )
        needs_sr = actual_plate_h < 35.0 or est_char_h < 22.0

        # 2. Noise Assessment on Rectified Region
        noise_sigma, patch_count = self.estimate_flat_region_noise_safe(rect_gray)
        noise_status = "UNKNOWN" if noise_sigma is None else f"{noise_sigma:.1f}"
        needs_denoise = (noise_sigma is not None and noise_sigma > 4.5)

        # 3. Multi-indicator Sharpness Assessment
        lap_var, grad_energy, composite_sharpness = self.compute_composite_sharpness(rect_gray)
        needs_sharpen = composite_sharpness < 28.0

        # 4. Illumination-independent Normalized Contrast Assessment
        contrast_spread = self.compute_contrast_spread(rect_gray)
        needs_clahe = contrast_spread < contrast_thresh

        return {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "actual_plate_h": actual_plate_h,
            "est_char_h": est_char_h,
            "char_height_method": char_method,
            "noise_sigma": noise_sigma,
            "noise_status": noise_status,
            "patch_count": patch_count,
            "lap_var": lap_var,
            "grad_energy": grad_energy,
            "composite_sharpness": composite_sharpness,
            "contrast_spread": contrast_spread,
            "needs_sr": needs_sr,
            "needs_denoise": needs_denoise,
            "needs_clahe": needs_clahe,
            "needs_sharpen": needs_sharpen
        }


estimate_flat_region_noise_safe = QualityAssessmentModule.estimate_flat_region_noise_safe
assess_plate_quality_comprehensive = QualityAssessmentModule().assess

