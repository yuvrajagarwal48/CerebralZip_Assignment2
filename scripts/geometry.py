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
        pad_pct_w: float = 0.06,
        pad_pct_h: float = 0.08
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
        Rotation-invariant algorithm that does not rely on sums/differences (x+y or y-x):
          1. Computes the centroid and sorts points cyclically (clockwise) by polar angle.
          2. Compares the two opposite edge pairs to identify the long (horizontal) edges vs short (vertical) edges.
          3. Distinguishes the top edge (smaller average y) from the bottom edge (larger average y).
          4. Orders vertices left-to-right along top edge (tl, tr) and bottom edge (bl, br).
        """
        pts = pts.reshape(4, 2).astype("float32")
        c = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts_sorted = pts[np.argsort(angles)]

        p0, p1, p2, p3 = pts_sorted
        e0_len = float(np.linalg.norm(p1 - p0))
        e1_len = float(np.linalg.norm(p2 - p1))
        e2_len = float(np.linalg.norm(p3 - p2))
        e3_len = float(np.linalg.norm(p0 - p3))

        # Plates are wider than they are tall: identify the pair of longer edges
        if (e0_len + e2_len) >= (e1_len + e3_len):
            pairA, pairB = (p0, p1), (p3, p2)
        else:
            pairA, pairB = (p1, p2), (p0, p3)

        yA = (pairA[0][1] + pairA[1][1]) / 2.0
        yB = (pairB[0][1] + pairB[1][1]) / 2.0
        top_pair, bot_pair = (pairA, pairB) if yA <= yB else (pairB, pairA)

        tl = top_pair[0] if top_pair[0][0] <= top_pair[1][0] else top_pair[1]
        tr = top_pair[1] if top_pair[0][0] <= top_pair[1][0] else top_pair[0]

        bl = bot_pair[0] if bot_pair[0][0] <= bot_pair[1][0] else bot_pair[1]
        br = bot_pair[1] if bot_pair[0][0] <= bot_pair[1][0] else bot_pair[0]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _evaluate_quad_candidate(
        approx: np.ndarray,
        crop_shape: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Scores an eligible 4-point contour based on:
        1. Contour Area & Proportion of the crop occupied
        2. Convexity & Solidity
        3. Four-corner approximation (orthogonality of corner angles)
        4. Aspect Ratio (measured naturally from the 4 corners)
        5. Parallel edge symmetry
        """
        h_crop, w_crop = crop_shape[:2]
        area_crop = float(h_crop * w_crop)
        area = float(cv2.contourArea(approx))

        # Filter out tiny noise contours or outer image boundaries
        if area < 0.15 * area_crop or area > 0.98 * area_crop:
            return None
        if not cv2.isContourConvex(approx):
            return None

        pts = PlateGeometryModule.order_quad_points_robust(approx)
        tl, tr, br, bl = pts

        width_top = float(np.linalg.norm(tr - tl))
        width_bot = float(np.linalg.norm(br - bl))
        height_r = float(np.linalg.norm(tr - br))
        height_l = float(np.linalg.norm(tl - bl))

        max_w = max(width_top, width_bot)
        max_h = max(height_r, height_l)
        mean_w = (width_top + width_bot) / 2.0
        mean_h = (height_r + height_l) / 2.0

        if mean_h < 4.0 or mean_w < 15.0:
            return None

        # Natural aspect ratio from four corners
        ar = mean_w / max(1e-4, mean_h)
        if ar < 1.4 or ar > 6.5:
            return None

        # 1. Occupancy proportion (target 40% - 90%)
        occupancy = area / area_crop
        s_occ = max(0.0, 1.0 - abs(occupancy - 0.70) / 0.35)

        # 2. Convexity & Solidity
        hull = cv2.convexHull(approx)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1e-4, hull_area)
        s_sol = min(1.0, solidity)

        # 3. Corner angle orthogonality (angles near 90 degrees)
        cos_angles = []
        for i in range(4):
            p_prev = pts[(i - 1) % 4]
            p_curr = pts[i]
            p_next = pts[(i + 1) % 4]
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))
            if norm1 > 1e-4 and norm2 > 1e-4:
                cos_angles.append(abs(float(np.dot(v1, v2) / (norm1 * norm2))))
            else:
                cos_angles.append(1.0)
        s_angle = max(0.0, 1.0 - float(np.mean(cos_angles)))

        # 4. Aspect ratio score (rewards standard plates ~ 2.0 to 5.0)
        s_ar = max(0.0, 1.0 - abs(ar - 3.7) / 2.2)

        # 5. Parallel edge symmetry
        s_sym = max(0.0, 1.0 - (abs(width_top - width_bot) + abs(height_l - height_r)) / (mean_w + mean_h))

        total_score = 0.25 * s_occ + 0.20 * s_sol + 0.25 * s_angle + 0.15 * s_ar + 0.15 * s_sym
        return {
            "score": total_score,
            "ordered_pts": pts,
            "target_w": int(round(max_w)),
            "target_h": int(round(max_h)),
            "mean_w": mean_w,
            "mean_h": mean_h,
            "aspect_ratio": ar,
            "occupancy": occupancy
        }

    def rectify(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], str]:
        """
        Performs 4-point perspective rectification or Hough line deskewing.
        Ranks all eligible candidate contours using area, convexity, corner orthogonality,
        aspect ratio, and occupancy.
        Calculates output dimensions directly from the 4 corners, preserving natural aspect ratio.
        Returns:
            Tuple of (rectified_crop, transform_matrix_3x3, (target_w, target_h), status_string)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr, np.eye(3, dtype=np.float32), (0, 0), "EMPTY_CROP"

        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Default fallback quadrilateral dimensions
        self.last_quad_dims = {
            "quad_h": float(h),
            "quad_w": float(w),
            "max_h": float(h),
            "max_w": float(w),
            "aspect_ratio": float(w / max(1e-5, h)),
            "method": "RAW_CROP_BOUNDS"
        }

        # Multi-modal edge and binarization representations for robust boundary discovery
        edge_maps = [
            cv2.Canny(blurred, 30, 120),
            cv2.Canny(blurred, 50, 150)
        ]
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge_maps.append(otsu)
        edge_maps.append(cv2.bitwise_not(otsu))

        candidates: List[Dict[str, Any]] = []

        for em in edge_maps:
            cnts, _ = cv2.findContours(em, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                if peri < 15:
                    continue
                # 1. Sweep adaptive epsilon factors for 4-corner approximation
                for eps_pct in [0.015, 0.025, 0.035, 0.05, 0.07]:
                    approx = cv2.approxPolyDP(c, eps_pct * peri, True)
                    if len(approx) == 4:
                        res = self._evaluate_quad_candidate(approx, crop_bgr.shape)
                        if res and res["score"] >= 0.40:
                            candidates.append(res)

                # 2. Minimum Area Oriented Bounding Quadrilateral
                rect = cv2.minAreaRect(c)
                box_pts = cv2.boxPoints(rect).astype(np.int32)
                res = self._evaluate_quad_candidate(box_pts, crop_bgr.shape)
                if res and res["score"] >= 0.40:
                    candidates.append(res)

        # Rank all eligible contours by composite score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        if candidates:
            best = candidates[0]
            ordered_pts = best["ordered_pts"]
            ar = best["aspect_ratio"]

            # Calculate natural dimensions directly from the 4 corners
            tw = max(16, best["target_w"])
            th = max(16, best["target_h"])

            # Resolution scaling: if quadrilateral is low-res, scale both dimensions
            # proportionally to guarantee sufficient pixel density for OCR,
            # strictly preserving the corner-derived physical aspect ratio (no fixed 4:1 stretch)
            if th < 32:
                scale = 48.0 / float(th)
                tw = int(round(tw * scale))
                th = int(round(th * scale))

            self.last_quad_dims = {
                "quad_h": float(best["mean_h"]),
                "quad_w": float(best["mean_w"]),
                "max_h": float(best["target_h"]),
                "max_w": float(best["target_w"]),
                "aspect_ratio": float(ar),
                "ordered_pts": ordered_pts,
                "score": float(best["score"]),
                "method": "RANKED_QUADRILATERAL"
            }

            dst_pts = np.array([
                [0, 0],
                [tw - 1, 0],
                [tw - 1, th - 1],
                [0, th - 1]
            ], dtype=np.float32)

            H = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            det = np.linalg.det(H)
            if 1e-4 < abs(det) < 1e4:
                warped = cv2.warpPerspective(
                    crop_bgr, H, (tw, th),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=[255, 255, 255]
                )
                self.last_residual_skew = 0.0
                return warped, H, (tw, th), f"HOMOGRAPHY_WARP(score={best['score']:.2f}, AR={ar:.2f})"

        # Path B: Hough Deskewing fallback
        edges = cv2.Canny(blurred, 30, 130)
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

        # Path C: Original Preserved safely without distortion
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

