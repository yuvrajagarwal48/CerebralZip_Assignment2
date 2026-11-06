"""
scripts/routing.py: Adaptive Enhancement Routing Module
========================================================
Consumes Quality Assessment metrics and decides which restoration stages
and resolution scalings to trigger. Produces a multi-candidate enhancement ladder:
  - Rectified_Unprocessed (unmolested baseline)
  - SR_4X_Native and SR_2X_Downsampled (if needs_sr is true)
  - Adaptive_Targeted (NLM + CLAHE + Sharpen applied conditionally)
  - Adaptive_Residual (Morphological re-inking applied conditionally)
"""

from typing import Tuple, Dict, List, Optional, Any
import cv2
import numpy as np

from scripts.restoration import ClassicalRestorationSuite
from scripts.super_resolution import RealESRGANEngine


class AdaptiveEnhancementRouter:
    """
    Routing Module:
    Consumes Quality Assessment metrics and decides which restoration stages
    and resolution scalings to trigger. Produces a multi-candidate enhancement ladder.
    """
    def __init__(self, restoration_suite: Optional[ClassicalRestorationSuite] = None):
        self.restoration = restoration_suite or ClassicalRestorationSuite()

    def generate_ladder(
        self,
        rectified_bgr: np.ndarray,
        qa: Dict[str, Any],
        sr_engine: RealESRGANEngine,
        border_pad: int = 18,
        border_color: List[int] = [255, 255, 255]
    ) -> Tuple[Dict[str, np.ndarray], str]:
        actions = []

        # Candidate 1: Completely Unprocessed Rectified Crop (baseline)
        cand_unprocessed = cv2.copyMakeBorder(
            rectified_bgr, border_pad, border_pad, border_pad, border_pad,
            cv2.BORDER_CONSTANT, value=border_color
        )

        # Resolution Enhancement Decision
        cand_sr_4x = None
        cand_sr_2x = None
        if qa["needs_sr"]:
            # Run 4x model ONCE and derive 2x output by downsampling
            img_4x, label_4x = sr_engine.upscale(rectified_bgr, target_scale=4.0)
            rh, rw = rectified_bgr.shape[:2]
            tw_2x, th_2x = int(rw * 2.0), int(rh * 2.0)
            img_2x = cv2.resize(img_4x, (tw_2x, th_2x), interpolation=cv2.INTER_LANCZOS4)
            label_2x = "REAL_ESRGAN_4X_DOWNSAMPLED_TO_2.0X"

            cand_sr_4x = cv2.copyMakeBorder(img_4x, border_pad, border_pad, border_pad, border_pad, cv2.BORDER_CONSTANT, value=border_color)
            cand_sr_2x = cv2.copyMakeBorder(img_2x, border_pad, border_pad, border_pad, border_pad, cv2.BORDER_CONSTANT, value=border_color)
            actions.append(label_2x)
            work_base = img_2x
        else:
            work_base = rectified_bgr.copy()
            actions.append("NATIVE_SCALE")

        # LAB Space Decomposition
        lab = cv2.cvtColor(work_base, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)

        # Classical Denoising Decision
        if qa["needs_denoise"]:
            l_filtered = self.restoration.nlm.apply(l)
            actions.append("NLM_FILTERED")
        else:
            l_filtered = l

        # Classical CLAHE Decision
        if qa["needs_clahe"]:
            l_contrast = self.restoration.clahe.apply(l_filtered)
            actions.append("CLAHE_APPLIED")
        else:
            l_contrast = l_filtered
            actions.append("CLAHE_BYPASSED")

        l_norm = self.restoration.normalize_percentile(l_contrast, 1.0, 99.0)

        # Classical Sharpening Decision
        if qa["needs_sharpen"]:
            l_sharp = self.restoration.sharpen.apply(l_norm)
            actions.append("SHARPEN_APPLIED")
        else:
            l_sharp = l_norm
            actions.append("SHARPEN_BYPASSED")

        cand_adaptive_clean = cv2.copyMakeBorder(
            cv2.cvtColor(l_sharp, cv2.COLOR_GRAY2BGR),
            border_pad, border_pad, border_pad, border_pad,
            cv2.BORDER_CONSTANT, value=border_color
        )

        # Adaptive Morphological Residual Decision
        cand_residual = None
        if qa["needs_clahe"] and qa["needs_sharpen"]:
            sh = work_base.shape[0]
            res_norm = self.restoration.morphology.apply(l_norm, height=sh)
            cand_residual = cv2.copyMakeBorder(
                cv2.cvtColor(res_norm, cv2.COLOR_GRAY2BGR),
                border_pad, border_pad, border_pad, border_pad,
                cv2.BORDER_CONSTANT, value=border_color
            )
            actions.append("RESIDUAL_REINK")

        # Compile Candidates
        candidates = {
            "Rectified_Unprocessed": cand_unprocessed,
            "Adaptive_Targeted": cand_adaptive_clean,
            "display_crop": cv2.cvtColor(l_sharp, cv2.COLOR_GRAY2BGR)
        }
        if cand_sr_2x is not None:
            candidates["SR_2X_Downsampled"] = cand_sr_2x
        if cand_sr_4x is not None:
            candidates["SR_4X_Native"] = cand_sr_4x
        if cand_residual is not None:
            candidates["Adaptive_Residual"] = cand_residual

        return candidates, " + ".join(actions)


def generate_genuinely_adaptive_ladder(rectified_bgr: np.ndarray, qa: dict, sr_engine: RealESRGANEngine) -> Tuple[Dict[str, np.ndarray], str]:
    router = AdaptiveEnhancementRouter()
    return router.generate_ladder(rectified_bgr, qa, sr_engine)

