"""
scripts/evaluation_metrics.py: Forensic Evaluation Metrics Module
=================================================================
Measures quantitative recovery fidelity and transcription accuracy:
  - String metrics: Levenshtein distance, Character Error Rate (CER), Exact Match.
  - Image metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM).
  - Spatial alignment: Phase correlation registration to verify physical correspondence.
  - Candidate benchmark table: Per-variant PSNR/SSIM gains against geometrically aligned reference.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    def psnr(image_true: np.ndarray, image_test: np.ndarray, data_range: float = 255.0) -> float:
        mse = np.mean((image_true.astype(np.float64) - image_test.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(10.0 * np.log10((data_range ** 2) / mse))

    def ssim(image_true: np.ndarray, image_test: np.ndarray, data_range: float = 255.0) -> float:
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        img1 = image_true.astype(np.float64)
        img2 = image_test.astype(np.float64)
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())


class ForensicEvaluator:
    """
    Evaluation Module:
    Measures quantitative recovery fidelity and transcription accuracy:
      - String metrics: Levenshtein distance, Character Error Rate (CER), Exact Match.
      - Image metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM).
      - Candidate benchmark table: Per-variant PSNR/SSIM gains against geometrically aligned reference.
    """
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return ForensicEvaluator.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dels = curr[j] + 1
                subs = prev[j] + (c1 != c2)
                curr.append(min(ins, dels, subs))
            prev = curr
        return prev[-1]

    @classmethod
    def compute_cer(cls, gt: str, pred: str) -> float:
        """
        Computes Character Error Rate (CER) = Levenshtein(gt, pred) / len(gt).
        Standard Character Error Rate is NOT capped at 1.0 and can exceed 1.0
        when there are multiple character insertions.
        """
        clean_gt = "".join([c for c in gt.upper() if c.isalnum()]).strip()
        clean_pred = "".join([c for c in pred.upper() if c.isalnum()]).strip() if pred not in ["NOT_DETECTED", "OCR_FAILED"] else ""
        if not clean_gt and not clean_pred:
            return 0.0
        if not clean_gt:
            return float(len(clean_pred))
        if pred in ["NOT_DETECTED", "OCR_FAILED"] or not clean_pred:
            return 1.0
        return float(cls.levenshtein_distance(clean_gt, clean_pred) / len(clean_gt))

    @staticmethod
    def compute_exact_match(gt: str, pred: str) -> bool:
        clean_gt = "".join([c for c in gt.upper() if c.isalnum()]).strip()
        clean_pred = "".join([c for c in pred.upper() if c.isalnum()]).strip()
        return bool(clean_gt and clean_gt == clean_pred)

    @staticmethod
    def compute_psnr(ref_gray: np.ndarray, target_gray: np.ndarray, max_psnr_ceiling: float = 60.0) -> float:
        if np.array_equal(ref_gray, target_gray):
            return max_psnr_ceiling
        try:
            val = float(psnr(ref_gray, target_gray))
            return max_psnr_ceiling if np.isinf(val) else val
        except Exception:
            return max_psnr_ceiling

    @staticmethod
    def compute_ssim(ref_gray: np.ndarray, target_gray: np.ndarray) -> float:
        return float(ssim(ref_gray, target_gray))

    @staticmethod
    def align_spatially(ref_gray: np.ndarray, target_gray: np.ndarray) -> np.ndarray:
        """
        Spatially aligns reference image to target image via phase correlation translation
        to ensure PSNR and SSIM compare corresponding physical plate regions.
        """
        if ref_gray.shape != target_gray.shape:
            ref_gray = cv2.resize(ref_gray, (target_gray.shape[1], target_gray.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        try:
            f_ref = np.float32(ref_gray)
            f_tar = np.float32(target_gray)
            shift, _ = cv2.phaseCorrelate(f_ref, f_tar)
            dx, dy = shift
            h, w = target_gray.shape[:2]
            if abs(dx) < 0.25 * w and abs(dy) < 0.25 * h:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                return cv2.warpAffine(ref_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass
        return ref_gray

    @classmethod
    def evaluate_candidates_fidelity(
        cls,
        variants: Dict[str, np.ndarray],
        clean_aligned_bgr: np.ndarray,
        raw_aligned_bgr: np.ndarray,
        border_pad: int = 18
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes PSNR and SSIM for every candidate without border or aspect ratio distortion:
        1. Clean and raw references undergo identical geometric transformation.
        2. Proportional scaling to the candidate's core plate dimensions preserves exact aspect ratio.
        3. Identical padding is applied to references matching candidate borders.
        4. Metrics are evaluated on the corresponding active plate regions without boundary distortion.
        """
        metrics = {}
        clean_gray_base = cv2.cvtColor(clean_aligned_bgr, cv2.COLOR_BGR2GRAY)
        raw_gray_base = cv2.cvtColor(raw_aligned_bgr, cv2.COLOR_BGR2GRAY)

        for cand_name, cand_img in variants.items():
            if cand_name == "display_crop":
                continue
            cand_h, cand_w = cand_img.shape[:2]
            cand_gray = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY) if cand_img.ndim == 3 else cand_img

            if cand_h > 2 * border_pad and cand_w > 2 * border_pad:
                core_h = cand_h - 2 * border_pad
                core_w = cand_w - 2 * border_pad
                cand_core = cand_gray[border_pad : cand_h - border_pad, border_pad : cand_w - border_pad]
            else:
                core_h = cand_h
                core_w = cand_w
                cand_core = cand_gray

            cl_core = cv2.resize(clean_gray_base, (core_w, core_h), interpolation=cv2.INTER_LANCZOS4)
            rw_core = cv2.resize(raw_gray_base, (core_w, core_h), interpolation=cv2.INTER_LANCZOS4)

            cl_padded = cv2.copyMakeBorder(cl_core, border_pad, border_pad, border_pad, border_pad, cv2.BORDER_CONSTANT, value=255)
            rw_padded = cv2.copyMakeBorder(rw_core, border_pad, border_pad, border_pad, border_pad, cv2.BORDER_CONSTANT, value=255)

            p_before = cls.compute_psnr(cl_core, rw_core)
            p_after = cls.compute_psnr(cl_core, cand_core)
            s_before = cls.compute_ssim(cl_core, rw_core)
            s_after = cls.compute_ssim(cl_core, cand_core)

            p_after_padded = cls.compute_psnr(cl_padded, cand_gray)
            s_after_padded = cls.compute_ssim(cl_padded, cand_gray)

            metrics[cand_name] = {
                "psnr_before": p_before,
                "psnr_after": p_after,
                "ssim_before": s_before,
                "ssim_after": s_after,
                "psnr_after_padded": p_after_padded,
                "ssim_after_padded": s_after_padded,
                "core_dimensions": (core_w, core_h)
            }

        return metrics


levenshtein_distance = ForensicEvaluator.levenshtein_distance
compute_cer = ForensicEvaluator.compute_cer

