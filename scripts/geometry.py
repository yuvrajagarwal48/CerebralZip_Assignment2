"""
scripts/geometry.py: Plate Geometry & Rectification Module
==========================================================
Responsible for ROI extraction, bounding box padding, corner sorting,
quadrilateral perspective homography, Hough-based skew correction,
and geometric reference alignment for evaluation.
"""

from typing import Tuple, List, Dict, Optional, Any
import cv2
import numpy as np


class PlateGeometryModule:
    """
    Geometry Module:
    Responsible for ROI extraction, bounding box padding, corner sorting,
    quadrilateral perspective homography, Hough-based skew correction,
    and geometric reference alignment for evaluation.
    """
    def __init__(self):
        self.last_quad_dims: Dict[str, Any] = {
            "quad_h": 0.0,
            "quad_w": 0.0,
            "method": "UNINITIALIZED"
        }
        self.last_residual_skew: float = 0.0

    @staticmethod
    def extract_padded_roi(
        img_bgr: np.ndarray,
        box: List[int],
        pad_pct_w: float = 0.03,
        pad_pct_h: float = 0.04
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Applies proportional safety margins around detected bounding box
        and extracts the clamped ROI crop.
        """
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = box
        pw = int((x2 - x1) * pad_pct_w)
        ph = int((y2 - y1) * pad_pct_h)
        x1_pad, y1_pad = max(0, x1 - pw), max(0, y1 - ph)
        x2_pad, y2_pad = min(w, x2 + pw), min(h, y2 + ph)
        crop = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        return crop, (x1_pad, y1_pad, x2_pad, y2_pad)

    @staticmethod
    def order_quad_points_robust(pts: np.ndarray) -> np.ndarray:
        """
        Rigorously orders 4 quadrilateral points into [top-left, top-right, bottom-right, bottom-left].
        """
        pts = pts.reshape(4, 2).astype("float32")
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts_sorted = pts[np.argsort(angles)]
        tl_idx = np.argmin(np.sum(pts_sorted, axis=1))
        return np.roll(pts_sorted, -tl_idx, axis=0)

    def rectify(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], str]:
        """
        Performs 4-point perspective rectification or Hough line deskewing.
        Measures the actual plate quadrilateral before rectification and records dimensions.
        Returns:
            Tuple of (rectified_crop, transform_matrix_3x3, (target_w, target_h), status_string)
        """
        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 130)

        # Default fallback quadrilateral dimensions
        self.last_quad_dims = {
            "quad_h": float(h),
            "quad_w": float(w),
            "max_h": float(h),
            "max_w": float(w),
            "aspect_ratio": float(w / max(1e-5, h)),
            "method": "RAW_CROP_BOUNDS"
        }

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

        # Path A: Quadrilateral Homography Rectification
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.035 * peri, True)
            area = cv2.contourArea(approx)

            if len(approx) == 4 and cv2.isContourConvex(approx) and area > (0.15 * h * w):
                ordered_pts = self.order_quad_points_robust(approx)
                (tl, tr, br, bl) = ordered_pts

                width_top = np.linalg.norm(tr - tl)
                width_bottom = np.linalg.norm(br - bl)
                max_w = int(max(width_top, width_bottom))

                height_right = np.linalg.norm(tr - br)
                height_left = np.linalg.norm(tl - bl)
                max_h = int(max(height_right, height_left))

                aspect_ratio = max_w / max(1e-5, max_h)
                if 1.8 <= aspect_ratio <= 6.5 and max_w >= 45 and max_h >= 16:
                    target_h = 96
                    target_w = int(target_h * aspect_ratio)

                    # Record physical pre-rectification quadrilateral dimensions
                    mean_quad_h = float((height_left + height_right) / 2.0)
                    mean_quad_w = float((width_top + width_bottom) / 2.0)
                    self.last_quad_dims = {
                        "quad_h": mean_quad_h,
                        "quad_w": mean_quad_w,
                        "max_h": float(max_h),
                        "max_w": float(max_w),
                        "aspect_ratio": float(aspect_ratio),
                        "ordered_pts": ordered_pts,
                        "method": "MEASURED_QUADRILATERAL"
                    }

                    dst_pts = np.array([
                        [0, 0],
                        [target_w - 1, 0],
                        [target_w - 1, target_h - 1],
                        [0, target_h - 1]
                    ], dtype="float32")

                    H = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                    det = np.linalg.det(H)
                    if 1e-4 < abs(det) < 1e4:
                        warped = cv2.warpPerspective(
                            crop_bgr, H, (target_w, target_h),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=[255, 255, 255]
                        )
                        self.last_residual_skew = 0.0
                        return warped, H, (target_w, target_h), "HOMOGRAPHY_WARP"

        # Path B: Hough Deskewing
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(min(h, w) * 0.55))
        angles = []
        if lines is not None:
            for line in lines[:15]:
                theta = line[0][1]
                deg = np.rad2deg(theta) - 90
                if -30.0 < deg < 30.0:
                    angles.append(deg)

        if angles:
            median_angle = float(np.median(angles))
            if abs(median_angle) > 1.0:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                deskewed = cv2.warpAffine(
                    crop_bgr, M, (w, h), flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255]
                )
                M_3x3 = np.eye(3, dtype=np.float32)
                M_3x3[:2, :] = M
                self.last_residual_skew = 0.0
                return deskewed, M_3x3, (w, h), f"HOUGH_AFFINE_{median_angle:.1f}deg"
            else:
                self.last_residual_skew = abs(median_angle)
        else:
            self.last_residual_skew = 0.0

        # Path C: Original Preserved
        return crop_bgr, np.eye(3, dtype=np.float32), (w, h), "ORIGINAL_PRESERVED"

    @staticmethod
    def align_reference(
        crop_bgr: np.ndarray,
        M_geom: np.ndarray,
        target_size: Tuple[int, int],
        geom_status: str
    ) -> np.ndarray:
        """
        Transforms reference or raw crops using identical geometric mapping
        for geometrically valid PSNR/SSIM evaluation.
        """
        target_w, target_h = target_size
        if "HOMOGRAPHY" in geom_status:
            return cv2.warpPerspective(
                crop_bgr, M_geom, (target_w, target_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[255, 255, 255]
            )
        elif "HOUGH" in geom_status:
            M_affine = M_geom[:2, :]
            return cv2.warpAffine(
                crop_bgr, M_affine, (target_w, target_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=[255, 255, 255]
            )
        else:
            return cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


order_quad_points_robust = PlateGeometryModule.order_quad_points_robust
perspective_rectify_plate = PlateGeometryModule().rectify

