"""
scripts/ocr_candidate_selection.py: OCR Candidate Selection Module
==================================================================
Aggregates candidates across enhancement ladder variants, checks multi-variant agreement,
analyzes character-level positional ambiguities, and selects the consensus prediction.
"""

from typing import Tuple, List, Dict, Any
from collections import Counter
import numpy as np

from scripts.ocr_recognition import DualPathPaddleOCREngine


class OCRCandidateSelectionModule:
    """
    OCR Candidate Selection Module:
    1. Collects transcriptions across all enhancement ladder variants + raw degraded crop.
    2. Validates candidates, giving priority to complete plate strings.
    3. Analyzes multi-variant agreement frequency across candidates.
    4. Detects character-level positional ambiguities when disagreements occur.
    5. Selects the most confident consensus prediction.
    """
    @staticmethod
    def collect_candidates(
        variants: Dict[str, np.ndarray],
        raw_crop: np.ndarray,
        ocr_engine: DualPathPaddleOCREngine
    ) -> List[Dict[str, Any]]:
        candidate_pool = []

        # 1. Baseline Raw Unrectified Crop
        t_raw, s_raw, c_raw = ocr_engine.transcribe(raw_crop, recognition_only=False)
        candidate_pool.append({"variant": "Raw_Degraded_Original", "text": t_raw, "conf": c_raw, "status": s_raw})

        # 2. Rectified Unprocessed
        if "Rectified_Unprocessed" in variants:
            t_unproc, s_unproc, c_unproc = ocr_engine.transcribe(variants["Rectified_Unprocessed"], recognition_only=False)
            candidate_pool.append({"variant": "Rectified_Unprocessed", "text": t_unproc, "conf": c_unproc, "status": s_unproc})

        # 3. Adaptive Targeted
        if "Adaptive_Targeted" in variants:
            t_adp, s_adp, c_adp = ocr_engine.transcribe(variants["Adaptive_Targeted"], recognition_only=False)
            candidate_pool.append({"variant": "Adaptive_Targeted", "text": t_adp, "conf": c_adp, "status": s_adp})

        # 4. Optional Super-Resolution Variants
        if "SR_2X_Downsampled" in variants:
            t_2x, s_2x, c_2x = ocr_engine.transcribe(variants["SR_2X_Downsampled"], recognition_only=False)
            candidate_pool.append({"variant": "SR_2X_Downsampled", "text": t_2x, "conf": c_2x, "status": s_2x})

        if "SR_4X_Native" in variants:
            t_4x, s_4x, c_4x = ocr_engine.transcribe(variants["SR_4X_Native"], recognition_only=False)
            candidate_pool.append({"variant": "SR_4X_Native", "text": t_4x, "conf": c_4x, "status": s_4x})

        # 5. Optional Adaptive Residual Re-Inking
        if "Adaptive_Residual" in variants:
            t_res, s_res, c_res = ocr_engine.transcribe(variants["Adaptive_Residual"], recognition_only=False)
            candidate_pool.append({"variant": "Adaptive_Residual", "text": t_res, "conf": c_res, "status": s_res})

        return candidate_pool

    @staticmethod
    def select(candidate_pool: List[Dict[str, Any]]) -> Tuple[str, float, str, List[str]]:
        valid = [c for c in candidate_pool if c["text"] not in ["NOT_DETECTED", "OCR_FAILED", ""]]
        if not valid:
            return "NOT_DETECTED", 0.0, "ALL_FAILED", []

        complete_cands = [c for c in valid if c["status"] == "COMPLETE_PLATE"]
        eval_pool = complete_cands if complete_cands else valid

        counts = Counter([c["text"] for c in eval_pool])
        best_text, best_count = counts.most_common(1)[0]
        best_confs = [c["conf"] for c in eval_pool if c["text"] == best_text]
        avg_conf = float(np.mean(best_confs))

        if best_count >= 2:
            return best_text, round(avg_conf, 3), f"MULTI_VARIANT_AGREEMENT ({best_count}/{len(eval_pool)})", []

        # If only Raw succeeded while all enhanced candidates collapsed
        raw_match = next((c for c in eval_pool if c["variant"] == "Raw_Degraded_Original"), None)
        if raw_match and len(eval_pool) == 1:
            return raw_match["text"], raw_match["conf"], "RAW_BASELINE_ACCEPTED", []

        # Ambiguity Check: Disagreements between candidates of identical length
        lengths = [len(c["text"]) for c in eval_pool]
        common_len, _ = Counter(lengths).most_common(1)[0]
        aligned = [c["text"] for c in eval_pool if len(c["text"]) == common_len]

        ambiguities = []
        if len(aligned) >= 2:
            for idx in range(common_len):
                col = [t[idx] for t in aligned]
                col_c = Counter(col)
                if len(col_c) > 1:
                    ambiguities.append(f"pos_{idx}:{dict(col_c)}")

        top_cand = max(eval_pool, key=lambda x: x["conf"])
        status = "UNCERTAIN_DISAGREEMENT" if ambiguities else "SINGLE_SOURCE_CANDIDATE"
        return top_cand["text"], top_cand["conf"], status, ambiguities


validate_and_select_ocr_candidate = OCRCandidateSelectionModule.select

