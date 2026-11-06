#!/usr/bin/env python3
"""
evaluation.py: Comprehensive Forensic License Plate Evaluation Framework
========================================================================
Performs rigorous component-wise evaluation, end-to-end benchmarking, character-level
OCR confusion analysis, and a 3-way progressive baseline comparison across real-world data:

Key Evaluation Modules:
1. DETECTION & LOCALIZATION EVALUATION:
   - Precision @ 0.50 IoU, Recall @ 0.50 IoU, F1-Score @ 0.50 IoU
   - True AP@0.50 (101-point COCO interpolation)
   - True mAP@0.50:0.95 (COCO standard IoU thresholds 0.50 to 0.95, step 0.05)
   - Mean IoU across all ground-truth plates (misses counted as 0.0)
   - Mean IoU on successfully detected plates
   - Detection failure rate (% of unlocalized plates)
   - Scale stratification (Small < 32², Medium 32²-96², Large >= 96² px)

2. GEOMETRIC RECTIFICATION BENCHMARK:
   - Residual skew angle after deskewing (degrees)
   - Aspect ratio deviation from standard Indian plate geometry (%)

3. THREE-WAY BASELINE COMPARISON:
   - Baseline 1: Original GT Plate Crop -> OCR (unenhanced crop to OCR)
   - Baseline 2: Enhanced GT Plate Crop -> OCR (adaptive restoration on GT crop to OCR)
   - Baseline 3: Full Autonomous Pipeline -> OCR (scene deblur -> YOLO -> rectify -> enhance -> OCR)

4. ADVANCED OCR & CHARACTER CONFUSION ANALYSIS:
   - Exact Match Accuracy (%)
   - Mean (Macro) Character Error Rate (CER)
   - Normalized Macro Character Accuracy (%)
   - Corpus (Micro) CER & Character Accuracy
   - Wagner-Fischer character alignment: Substitutions, Insertions, Deletions
   - Top character confusion pairs
   - Character-position accuracy curve across plate indices (0..9)

5. RUNTIME & THROUGHPUT PROFILING:
   - Component-wise latency breakdown (ms) and percentage of pipeline runtime
   - Average frame latency (ms) and throughput (FPS)

Strict Methodological Guarantees:
- Consistent evaluation denominator (N) across all detection, OCR, and baseline metrics.
- Unreadable images, localization failures, and empty OCR outputs are penalized with CER=1.0 and Exact Match=False.
- All configuration and file paths are HARDCODED (zero CLI arguments required).
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from scripts.detection import AutonomousPlateDetector
    from scripts.deblurring import DeblurringModule
    from scripts.super_resolution import RealESRGANEngine
    from scripts.geometry import PlateGeometryModule
    from scripts.quality_assessment import QualityAssessmentModule
    from scripts.restoration import ClassicalRestorationSuite
    from scripts.routing import AdaptiveEnhancementRouter
    from scripts.ocr_recognition import DualPathPaddleOCREngine
    from scripts.ocr_candidate_selection import OCRCandidateSelectionModule
    from scripts.evaluation_metrics import ForensicEvaluator
    from scripts.pipeline import MainImagePipeline
except ImportError:
    try:
        from forensic_pipeline import (
            AutonomousPlateDetector,
            DeblurringModule,
            RealESRGANEngine,
            PlateGeometryModule,
            QualityAssessmentModule,
            ClassicalRestorationSuite,
            AdaptiveEnhancementRouter,
            DualPathPaddleOCREngine,
            OCRCandidateSelectionModule,
            ForensicEvaluator,
            MainImagePipeline
        )
    except ImportError:
        pass


# =====================================================================
# HARDCODED CONFIGURATION & PATHS (NO CLI ARGUMENTS)
# =====================================================================
EVAL_MANIFEST_PATH = "./data/dataset_prepared/manifest.json"
EVAL_OUTPUT_DIR = "./results/evaluation"
EVAL_LIMIT_SAMPLES = None       # Set to integer (e.g. 10) for testing, or None for entire dataset
EVAL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 1. DETECTION & LOCALIZATION EVALUATION
# =====================================================================
class DetectionEvaluator:
    """
    Evaluates license plate detection and localization performance:
      - Intersection over Union (IoU)
      - Precision, Recall, and F1 at 0.50 IoU
      - True AP@0.50 and COCO-style mAP@0.50:0.95
      - Detection failure rate (% of unlocalized plates)
      - Stratification by plate size (Small < 32², Medium 32²-96², Large >= 96² px)
    """
    @staticmethod
    def compute_iou(boxA: Optional[List[Union[int, float]]], boxB: Optional[List[Union[int, float]]]) -> float:
        """Computes standard Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        if boxA is None or boxB is None:
            return 0.0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, float(xB - xA))
        inter_h = max(0.0, float(yB - yA))
        inter_area = inter_w * inter_h

        boxA_area = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
        boxB_area = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))
        union_area = boxA_area + boxB_area - inter_area

        if union_area <= 1e-6:
            return 0.0
        return float(inter_area / union_area)

    @classmethod
    def classify_plate_size(cls, box: List[int]) -> str:
        """Classifies plate area according to standard COCO object size scales."""
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area < 32 * 32:
            return "small"
        elif area <= 96 * 96:
            return "medium"
        else:
            return "large"

    @classmethod
    def evaluate_detections(
        cls,
        detector: Union[AutonomousPlateDetector, List[AutonomousPlateDetector]],
        test_cases: List[Dict[str, Any]],
        conf_operating_point: float = 0.25
    ) -> Dict[str, Any]:
        """
        Runs dedicated YOLO detection evaluation with COCO-style AP and size stratification.
        Supports multi-GPU detector lists for accelerated 2xT4 inference.
        """
        total_gt = len(test_cases)
        if total_gt == 0:
            return {}

        detectors = detector if isinstance(detector, list) else [detector]
        num_workers = len(detectors)

        records = [None] * total_gt
        ious_all = [0.0] * total_gt
        ious_detected = []
        size_stats = defaultdict(lambda: {"total": 0, "detected": 0, "tp_05": 0, "ious": []})
        missing_images = 0

        def process_det(item):
            idx, case = item
            det = detectors[idx % num_workers]
            dev = getattr(det, "device", None)
            if dev is not None and dev.type == "cuda":
                torch.cuda.set_device(dev)

            img_path = case["image_path"]
            gt_box = case["gt_bbox"]
            plate_scale = cls.classify_plate_size(gt_box)

            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                rec = {
                    "score": 0.0,
                    "matched": False,
                    "gt_box": gt_box,
                    "pred_box": None,
                    "iou": 0.0,
                    "scale": plate_scale
                }
                return idx, plate_scale, 0.0, None, 0.0, False, True, rec

            pred_box, conf = det.detect(img)
            if pred_box is not None:
                iou = cls.compute_iou(pred_box, gt_box)
                rec = {
                    "score": float(conf),
                    "matched": True,
                    "gt_box": gt_box,
                    "pred_box": pred_box,
                    "iou": iou,
                    "scale": plate_scale
                }
                return idx, plate_scale, float(conf), pred_box, iou, True, False, rec
            else:
                rec = {
                    "score": 0.0,
                    "matched": False,
                    "gt_box": gt_box,
                    "pred_box": None,
                    "iou": 0.0,
                    "scale": plate_scale
                }
                return idx, plate_scale, 0.0, None, 0.0, False, False, rec

        if num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_det, (i, c)): i for i, c in enumerate(test_cases)}
                for f in as_completed(futures):
                    idx, p_scale, conf, p_box, iou, is_det, is_miss, rec = f.result()
                    records[idx] = rec
                    ious_all[idx] = iou
                    size_stats[p_scale]["total"] += 1
                    if is_miss:
                        missing_images += 1
                    elif is_det:
                        ious_detected.append(iou)
                        size_stats[p_scale]["detected"] += 1
                        size_stats[p_scale]["ious"].append(iou)
                        if iou >= 0.50:
                            size_stats[p_scale]["tp_05"] += 1
        else:
            for i, case in enumerate(test_cases):
                idx, p_scale, conf, p_box, iou, is_det, is_miss, rec = process_det((i, case))
                records[idx] = rec
                ious_all[idx] = iou
                size_stats[p_scale]["total"] += 1
                if is_miss:
                    missing_images += 1
                elif is_det:
                    ious_detected.append(iou)
                    size_stats[p_scale]["detected"] += 1
                    size_stats[p_scale]["ious"].append(iou)
                    if iou >= 0.50:
                        size_stats[p_scale]["tp_05"] += 1

        # Calculate Operating-Point Metrics at IoU >= 0.50
        tp_05 = sum(1 for r in records if r["matched"] and r["score"] >= conf_operating_point and r["iou"] >= 0.50)
        fp_05 = sum(1 for r in records if r["matched"] and r["score"] >= conf_operating_point and r["iou"] < 0.50)
        fn_05 = total_gt - tp_05

        prec_05 = float(tp_05 / (tp_05 + fp_05)) if (tp_05 + fp_05) > 0 else 0.0
        rec_05 = float(tp_05 / total_gt)
        f1_05 = float(2 * (prec_05 * rec_05) / (prec_05 + rec_05)) if (prec_05 + rec_05) > 0 else 0.0

        # Calculate True COCO-Style AP across multiple IoU thresholds
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        ap_scores = []
        map_dict = {}

        # Sort all candidate records by detection confidence descending
        sorted_records = sorted(records, key=lambda x: x["score"], reverse=True)

        for thresh in iou_thresholds:
            tp_flags = []
            for r in sorted_records:
                if r["matched"] and r["iou"] >= thresh:
                    tp_flags.append(1)
                else:
                    tp_flags.append(0)

            tp_cumsum = np.cumsum(tp_flags)
            fp_cumsum = np.cumsum([1 - f for f in tp_flags])

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
            recalls = tp_cumsum / total_gt

            # 101-point COCO interpolation
            recall_levels = np.linspace(0.0, 1.0, 101)
            interpolated_precisions = []
            for rl in recall_levels:
                prec_above = precisions[recalls >= rl]
                if len(prec_above) > 0:
                    interpolated_precisions.append(np.max(prec_above))
                else:
                    interpolated_precisions.append(0.0)

            ap = float(np.mean(interpolated_precisions))
            ap_scores.append(ap)
            map_dict[f"ap_{thresh:.2f}"] = round(ap, 4)

        true_ap_05 = ap_scores[0]
        true_map_05_95 = float(np.mean(ap_scores))

        # Size stratification metrics
        scale_metrics = {}
        for scale, stats in size_stats.items():
            tot = stats["total"]
            det = stats["detected"]
            tp = stats["tp_05"]
            scale_metrics[scale] = {
                "test_count": tot,
                "detected_count": det,
                "recall_at_0.50": round(float(tp / tot), 4) if tot > 0 else 0.0,
                "mean_iou_all": round(float(sum(stats["ious"]) / tot), 4) if tot > 0 else 0.0,
                "mean_iou_detected": round(float(sum(stats["ious"]) / det), 4) if det > 0 else 0.0
            }

        unlocalized_count = total_gt - len(ious_detected)
        det_failure_rate = float(unlocalized_count / total_gt)

        return {
            "total_samples": total_gt,
            "successfully_loaded": total_gt - missing_images,
            "missing_or_corrupt_images": missing_images,
            "precision_at_0.50": round(prec_05, 4),
            "recall_at_0.50": round(rec_05, 4),
            "f1_score_at_0.50": round(f1_05, 4),
            "ap_at_0.50": round(true_ap_05, 4),
            "map_0.50_0.95": round(true_map_05_95, 4),
            "ap_thresholds": map_dict,
            "mean_iou_all": round(float(np.mean(ious_all)), 4) if ious_all else 0.0,
            "mean_iou_detected_only": round(float(np.mean(ious_detected)), 4) if ious_detected else 0.0,
            "detection_failure_rate": round(det_failure_rate, 4),
            "ious_all": ious_all,
            "size_stratification": scale_metrics
        }


# =====================================================================
# 2. GEOMETRIC RECTIFICATION EVALUATION
# =====================================================================
class GeometryEvaluator:
    """Evaluates perspective rectification and deskewing."""
    @staticmethod
    def evaluate_rectification(
        geometry_module: PlateGeometryModule,
        test_cases: List[Dict[str, Any]],
        target_aspect_ratio: float = 4.0
    ) -> Dict[str, Any]:
        residual_skews = []
        aspect_ratio_errors = []

        for case in test_cases:
            img = cv2.imread(case["image_path"])
            if img is None or img.size == 0:
                continue

            gt_box = case["gt_bbox"]
            plate_crop, _ = geometry_module.extract_padded_roi(img, gt_box)
            rectified, _, (tw, th), status = geometry_module.rectify(plate_crop)

            skew = getattr(geometry_module, "last_residual_skew", 0.0)
            residual_skews.append(abs(skew))

            measured_aspect = tw / max(1, th)
            ar_error = abs(measured_aspect - target_aspect_ratio) / target_aspect_ratio
            aspect_ratio_errors.append(ar_error)

        return {
            "mean_residual_skew_deg": round(float(np.mean(residual_skews)), 2) if residual_skews else 0.0,
            "mean_aspect_ratio_error": round(float(np.mean(aspect_ratio_errors)), 4) if aspect_ratio_errors else 0.0
        }


# =====================================================================
# 3. ADVANCED OCR & CHARACTER CONFUSION EVALUATOR
# =====================================================================
class OCREvaluator:
    """
    Computes exact match, CER, character accuracy, Wagner-Fischer character alignment,
    and positional accuracy curves.
    """
    @staticmethod
    def align_characters_wagner_fischer(gt: str, pred: str) -> List[Tuple[str, str, str]]:
        """
        Aligns ground truth and predicted strings using Wagner-Fischer dynamic programming.
        Returns list of (operation, gt_char, pred_char) where operation in ('match', 'sub', 'ins', 'del').
        """
        m, n = len(gt), len(pred)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if gt[i - 1] == pred[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost # substitution or match
                )

        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and gt[i - 1] == pred[j - 1]:
                alignment.append(("match", gt[i - 1], pred[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                alignment.append(("sub", gt[i - 1], pred[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                alignment.append(("del", gt[i - 1], "-"))
                i -= 1
            else:
                alignment.append(("ins", "-", pred[j - 1]))
                j -= 1

        alignment.reverse()
        return alignment

    @classmethod
    def evaluate_transcriptions(
        cls,
        eval_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates OCR metrics across evaluated samples with strict denominator accounting.
        """
        total_cases = len(eval_cases)
        if total_cases == 0:
            return {}

        exact_matches = 0
        cer_values = []
        char_accuracies = []
        total_edit_distance = 0
        total_gt_characters = 0

        substitution_counter = Counter()
        deletion_counter = Counter()
        insertion_counter = Counter()
        positional_correct = defaultdict(int)
        positional_total = defaultdict(int)

        for item in eval_cases:
            gt = item["gt_text"]
            pred = item["pred_text"]

            clean_gt = "".join([c for c in gt.upper() if c.isalnum()]).strip()
            clean_pred = "".join([c for c in pred.upper() if c.isalnum()]).strip() if pred not in ["NOT_DETECTED", "OCR_FAILED", "LOCALIZATION_FAILED"] else ""

            is_match = (clean_gt == clean_pred) and (len(clean_gt) > 0)
            if is_match:
                exact_matches += 1

            # Standard Levenshtein edit distance
            edit_dist = ForensicEvaluator.levenshtein_distance(clean_gt, clean_pred)
            total_edit_distance += edit_dist
            total_gt_characters += len(clean_gt)

            # Sample CER: edit_dist / len(clean_gt)
            cer = float(edit_dist / len(clean_gt)) if len(clean_gt) > 0 else (1.0 if clean_pred else 0.0)
            cer_values.append(cer)
            char_accuracies.append(max(0.0, 1.0 - cer))

            # Alignment analysis
            if clean_gt:
                alignment = cls.align_characters_wagner_fischer(clean_gt, clean_pred)
                gt_idx = 0
                for op, g, p in alignment:
                    if op == "sub":
                        substitution_counter[f"'{g}' -> '{p}'"] += 1
                        positional_total[gt_idx] += 1
                        gt_idx += 1
                    elif op == "del":
                        deletion_counter[f"'{g}'"] += 1
                        positional_total[gt_idx] += 1
                        gt_idx += 1
                    elif op == "ins":
                        insertion_counter[f"'{p}'"] += 1
                    elif op == "match":
                        positional_correct[gt_idx] += 1
                        positional_total[gt_idx] += 1
                        gt_idx += 1

        exact_match_rate = float(exact_matches / total_cases)
        mean_cer = float(np.mean(cer_values))
        macro_char_acc = float(np.mean(char_accuracies))
        corpus_cer = float(total_edit_distance / total_gt_characters) if total_gt_characters > 0 else 0.0
        corpus_char_acc = max(0.0, 1.0 - corpus_cer)

        position_curve = {}
        max_idx = min(10, max(positional_total.keys()) + 1 if positional_total else 10)
        for pos in range(max_idx):
            tot = positional_total[pos]
            corr = positional_correct[pos]
            position_curve[f"index_{pos}"] = round(float(corr / tot), 4) if tot > 0 else 0.0

        return {
            "total_evaluated_samples": total_cases,
            "exact_matches": exact_matches,
            "exact_match_rate": round(exact_match_rate, 4),
            "mean_cer": round(mean_cer, 4),
            "macro_char_accuracy": round(macro_char_acc, 4),
            "corpus_cer": round(corpus_cer, 4),
            "corpus_char_accuracy": round(corpus_char_acc, 4),
            "total_gt_characters": total_gt_characters,
            "total_edit_distance": total_edit_distance,
            "positional_accuracy_curve": position_curve,
            "top_substitutions": dict(substitution_counter.most_common(8)),
            "top_deletions": dict(deletion_counter.most_common(5)),
            "top_insertions": dict(insertion_counter.most_common(5))
        }


# =====================================================================
# 4. THREE-WAY BASELINE COMPARISON ENGINE
# =====================================================================
class BaselineComparisonEngine:
    """
    Executes a 3-way progressive comparison to measure exact modular impact:
      - Baseline 1: Original GT Plate Crop -> OCR (pure unenhanced crop to OCR)
      - Baseline 2: Enhanced GT Plate Crop -> OCR (isolated enhancement on GT crop to OCR)
      - Baseline 3: Full Autonomous Pipeline -> OCR (full scene -> detection -> rectify -> enhance -> OCR)
    """
    @classmethod
    def run_baselines(
        cls,
        pipeline: Union[MainImagePipeline, List[MainImagePipeline]],
        test_cases: List[Dict[str, Any]],
        border_pad: int = 18
    ) -> Dict[str, Any]:
        pipelines = pipeline if isinstance(pipeline, list) else [pipeline]
        num_workers = len(pipelines)
        total_cases = len(test_cases)
        print(f"\n[*] Running 3-Way Baseline Comparison across {total_cases} test samples using {num_workers} pipeline worker(s)...")

        b1_records = [None] * total_cases
        b2_records = [None] * total_cases
        b3_records = [None] * total_cases

        def process_sample(item):
            idx, case = item
            pipe = pipelines[idx % num_workers]
            dev = getattr(pipe, "device", None)
            if dev is not None and dev.type == "cuda":
                torch.cuda.set_device(dev)

            img_path = case["image_path"]
            gt_text = case["ground_truth_text"]
            gt_box = case["gt_bbox"]

            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                # Handle unreadable image: penalize all baselines consistently
                return (idx,
                        {"gt_text": gt_text, "pred_text": "NOT_DETECTED"},
                        {"gt_text": gt_text, "pred_text": "NOT_DETECTED"},
                        {"gt_text": gt_text, "pred_text": "NOT_DETECTED"})

            # ---------------------------------------------------------
            # Baseline 1: Original GT Plate Crop -> OCR
            # ---------------------------------------------------------
            gx1, gy1, gx2, gy2 = gt_box
            gt_crop = img[gy1:gy2, gx1:gx2]
            if gt_crop.size > 0:
                padded_crop = cv2.copyMakeBorder(
                    gt_crop, border_pad, border_pad, border_pad, border_pad,
                    cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )
                b1_text, _, _ = pipe.ocr.transcribe(padded_crop)
            else:
                b1_text = "NOT_DETECTED"
            b1_res = {"gt_text": gt_text, "pred_text": b1_text}

            # ---------------------------------------------------------
            # Baseline 2: Enhanced GT Plate Crop -> OCR
            # ---------------------------------------------------------
            if gt_crop.size > 0:
                rectified_crop, _, _, _ = pipe.geometry.rectify(gt_crop)
                qa_metrics = pipe.qa.assess(gt_crop, rectified_crop)
                variants, _ = pipe.router.generate_ladder(rectified_crop, qa_metrics, pipe.sr_engine)
                candidate_pool = pipe.selector.collect_candidates(variants, gt_crop, pipe.ocr)
                b2_text, _, _, _ = pipe.selector.select(candidate_pool)
            else:
                b2_text = "NOT_DETECTED"
            b2_res = {"gt_text": gt_text, "pred_text": b2_text}

            # ---------------------------------------------------------
            # Baseline 3: Full Autonomous Pipeline -> OCR
            # ---------------------------------------------------------
            try:
                res = pipe.run(input_bgr=img, reference_clean_bgr=None, track_latency=False)
                b3_text = res["final_text"]
            except Exception:
                b3_text = "NOT_DETECTED"
            b3_res = {"gt_text": gt_text, "pred_text": b3_text}

            return (idx, b1_res, b2_res, b3_res)

        if num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            completed = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_sample, (i, case)): i for i, case in enumerate(test_cases)}
                for f in as_completed(futures):
                    idx, r1, r2, r3 = f.result()
                    b1_records[idx] = r1
                    b2_records[idx] = r2
                    b3_records[idx] = r3
                    completed += 1
                    if completed % 25 == 0 or completed == total_cases:
                        print(f"  -> Evaluated {completed}/{total_cases} baseline cases ({completed/total_cases*100:.1f}%)")
        else:
            for i, case in enumerate(test_cases):
                idx, r1, r2, r3 = process_sample((i, case))
                b1_records[idx] = r1
                b2_records[idx] = r2
                b3_records[idx] = r3
                if (i + 1) % 25 == 0 or (i + 1) == total_cases:
                    print(f"  -> Evaluated {i + 1}/{total_cases} baseline cases ({(i + 1)/total_cases*100:.1f}%)")

        # Evaluate OCR transcriptions across all 3 baselines using identical denominators
        b1_metrics = OCREvaluator.evaluate_transcriptions(b1_records)
        b2_metrics = OCREvaluator.evaluate_transcriptions(b2_records)
        b3_metrics = OCREvaluator.evaluate_transcriptions(b3_records)

        return {
            "baseline_1_raw_gt_crop": {
                "name": "Original GT Plate Crop -> OCR",
                "exact_match_rate": b1_metrics["exact_match_rate"],
                "mean_cer": b1_metrics["mean_cer"],
                "macro_char_accuracy": b1_metrics["macro_char_accuracy"],
                "corpus_cer": b1_metrics["corpus_cer"],
                "corpus_char_accuracy": b1_metrics["corpus_char_accuracy"]
            },
            "baseline_2_enhanced_gt_crop": {
                "name": "Enhanced GT Plate Crop -> OCR",
                "exact_match_rate": b2_metrics["exact_match_rate"],
                "mean_cer": b2_metrics["mean_cer"],
                "macro_char_accuracy": b2_metrics["macro_char_accuracy"],
                "corpus_cer": b2_metrics["corpus_cer"],
                "corpus_char_accuracy": b2_metrics["corpus_char_accuracy"]
            },
            "baseline_3_full_pipeline": {
                "name": "Full Autonomous Pipeline -> OCR",
                "exact_match_rate": b3_metrics["exact_match_rate"],
                "mean_cer": b3_metrics["mean_cer"],
                "macro_char_accuracy": b3_metrics["macro_char_accuracy"],
                "corpus_cer": b3_metrics["corpus_cer"],
                "corpus_char_accuracy": b3_metrics["corpus_char_accuracy"]
            }
        }


# =====================================================================
# 5. MASTER FORENSIC PIPELINE EVALUATOR
# =====================================================================
class ForensicPipelineEvaluator:
    """
    Coordinates end-to-end evaluation, 3-way baseline benchmarks,
    character error analysis, and diagnostic report generation.
    """
    def __init__(
        self,
        pipeline: Optional[MainImagePipeline] = None,
        manifest_path: str = EVAL_MANIFEST_PATH,
        output_dir: str = EVAL_OUTPUT_DIR,
        device: Optional[torch.device] = None,
        devices: Optional[List[torch.device]] = None
    ):
        self.device = device or torch.device(EVAL_DEVICE)
        self.output_dir = Path(output_dir)
        self.manifest_path = Path(manifest_path)
        self.plots_dir = self.output_dir / "plots"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        num_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if devices is not None and len(devices) > 0:
            self.devices = devices
        elif num_avail >= 2:
            self.devices = [torch.device(f"cuda:{i}") for i in range(min(num_avail, 2))]
        else:
            self.devices = [self.device]

        print(f"[*] Initializing Evaluation Suite on Primary Device: {self.device}")
        if len(self.devices) > 1:
            print(f"[⚡] Multi-GPU Evaluation Active ({len(self.devices)}x GPUs): {[str(d) for d in self.devices]}")
            self.pipelines = []
            for d in self.devices:
                if pipeline is not None and getattr(pipeline, "device", None) == d:
                    self.pipelines.append(pipeline)
                else:
                    self.pipelines.append(MainImagePipeline(device=d))
            self.pipeline = self.pipelines[0]
        else:
            self.pipeline = pipeline or MainImagePipeline(device=self.device)
            self.pipelines = [self.pipeline]

        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads dataset test cases from manifest.json."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"[Evaluation] Manifest file does not exist: {self.manifest_path.resolve()}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        print(f"[✓] Loaded {len(cases)} test cases from manifest.")
        return cases

    def run_full_evaluation(self, limit_samples: Optional[int] = EVAL_LIMIT_SAMPLES) -> Dict[str, Any]:
        """Executes full standardized evaluation across the dataset."""
        test_cases = self.dataset
        if limit_samples is not None and limit_samples > 0:
            print(f"[*] Limiting evaluation to first {limit_samples} samples.")
            test_cases = test_cases[:limit_samples]

        total_samples = len(test_cases)
        start_time = time.perf_counter()

        # -------------------------------------------------------------
        # Stage 1: Plate Detection & Localization Evaluation
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[1/4] EVALUATING DETECTION & LOCALIZATION MODULE (YOLOv8)...")
        print("=" * 70)
        detectors = [p.detector for p in self.pipelines]
        det_metrics = DetectionEvaluator.evaluate_detections(detectors, test_cases)
        ious_all = det_metrics.pop("ious_all", [])

        # -------------------------------------------------------------
        # Stage 2: Geometric Rectification Evaluation
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[2/4] EVALUATING GEOMETRIC RECTIFICATION MODULE...")
        print("=" * 70)
        geom_metrics = GeometryEvaluator.evaluate_rectification(self.pipeline.geometry, test_cases)

        # -------------------------------------------------------------
        # Stage 3: Three-Way Baseline Comparison
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[3/4] EXECUTING THREE-WAY BASELINE COMPARISON...")
        print("=" * 70)
        baseline_metrics = BaselineComparisonEngine.run_baselines(self.pipelines, test_cases)

        # -------------------------------------------------------------
        # Stage 4: End-to-End Pipeline Evaluation & Latency Profiling
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[4/4] EVALUATING END-TO-END PIPELINE & CHARACTER CONFUSION...")
        print("=" * 70)
        e2e_records = [None] * total_samples
        stage_latencies = defaultdict(list)
        total_latencies = []
        num_workers = len(self.pipelines)

        def process_e2e(item):
            idx, case = item
            pipe = self.pipelines[idx % num_workers]
            dev = getattr(pipe, "device", None)
            if dev is not None and dev.type == "cuda":
                torch.cuda.set_device(dev)

            img_path = case["image_path"]
            gt_text = case["ground_truth_text"]

            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                return idx, {
                    "sample_id": case.get("sample_id", idx),
                    "gt_text": gt_text,
                    "pred_text": "NOT_DETECTED"
                }, 0.0, {}

            t_start = time.perf_counter()
            res = pipe.run(
                input_bgr=img,
                reference_clean_bgr=None,
                apply_nafnet=True,
                track_latency=True
            )
            frame_ms = (time.perf_counter() - t_start) * 1000.0
            timings = res.get("timings", {})
            return idx, {
                "sample_id": case.get("sample_id", idx),
                "gt_text": gt_text,
                "pred_text": res["final_text"]
            }, frame_ms, timings

        if num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            completed = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_e2e, (i, case)): i for i, case in enumerate(test_cases)}
                for f in as_completed(futures):
                    idx, rec, frame_ms, timings = f.result()
                    e2e_records[idx] = rec
                    if frame_ms > 0:
                        total_latencies.append(frame_ms)
                    for stg, sec in timings.items():
                        stage_latencies[stg].append(sec * 1000.0)
                    completed += 1
                    if completed % 25 == 0 or completed == total_samples:
                        print(f"  -> Processed {completed}/{total_samples} pipeline samples ({completed/total_samples*100:.1f}%)")
        else:
            for i, case in enumerate(test_cases):
                idx, rec, frame_ms, timings = process_e2e((i, case))
                e2e_records[idx] = rec
                if frame_ms > 0:
                    total_latencies.append(frame_ms)
                for stg, sec in timings.items():
                    stage_latencies[stg].append(sec * 1000.0)
                if (i + 1) % 25 == 0 or (i + 1) == total_samples:
                    print(f"  -> Processed {i + 1}/{total_samples} pipeline samples ({(i + 1)/total_samples*100:.1f}%)")

        ocr_metrics = OCREvaluator.evaluate_transcriptions(e2e_records)

        # Latency calculations
        mean_frame_ms = float(np.mean(total_latencies)) if total_latencies else 0.0
        fps = float(1000.0 / mean_frame_ms) if mean_frame_ms > 0 else 0.0
        latency_breakdown = {
            stage: round(float(np.mean(vals)), 2)
            for stage, vals in stage_latencies.items()
        }

        total_wall_sec = round(time.perf_counter() - start_time, 2)

        # Assemble master evaluation summary
        summary = {
            "evaluation_metadata": {
                "manifest_path": str(self.manifest_path),
                "total_test_samples": total_samples,
                "evaluated_samples": total_samples,
                "device": str(self.device),
                "total_wall_clock_sec": total_wall_sec,
                "mean_frame_latency_ms": round(mean_frame_ms, 2),
                "throughput_fps": round(fps, 2)
            },
            "detection_evaluation": det_metrics,
            "geometry_evaluation": geom_metrics,
            "baseline_comparison": baseline_metrics,
            "end_to_end_evaluation": {
                "exact_match_rate": ocr_metrics["exact_match_rate"],
                "mean_cer": ocr_metrics["mean_cer"],
                "macro_char_accuracy": ocr_metrics["macro_char_accuracy"],
                "corpus_cer": ocr_metrics["corpus_cer"],
                "corpus_char_accuracy": ocr_metrics["corpus_char_accuracy"],
                "total_gt_characters": ocr_metrics["total_gt_characters"],
                "total_edit_distance": ocr_metrics["total_edit_distance"],
                "top_substitutions": ocr_metrics["top_substitutions"],
                "top_deletions": ocr_metrics["top_deletions"],
                "top_insertions": ocr_metrics["top_insertions"]
            },
            "character_position_curve": ocr_metrics["positional_accuracy_curve"],
            "latency_profiling": {
                "mean_total_ms": round(mean_frame_ms, 2),
                "fps": round(fps, 2),
                "stage_latencies_ms": latency_breakdown
            }
        }

        # -------------------------------------------------------------
        # Generate Reports and Diagnostic Plots
        # -------------------------------------------------------------
        self.generate_reports_and_plots(summary, ious_all)

        return summary

    def generate_reports_and_plots(self, summary: Dict[str, Any], ious_all: List[float]):
        """Generates JSON, Markdown report, and diagnostic visualization figures."""
        # 1. Export JSON Summary
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[✓] Saved Master Evaluation JSON to: {json_path.resolve()}")

        # 2. Export Diagnostic Plots
        self.plot_baseline_comparison(summary["baseline_comparison"])
        self.plot_positional_accuracy(summary["character_position_curve"])
        self.plot_iou_distribution(ious_all)

        # 3. Export Comprehensive Markdown Report
        md_path = self.output_dir / "EVALUATION_REPORT.md"
        report_md = self.generate_markdown_report(summary)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[✓] Saved Comprehensive Markdown Report to: {md_path.resolve()}")
        print(f"[✓] Saved Diagnostic Plots to: {self.plots_dir.resolve()}")

    def plot_baseline_comparison(self, baseline_metrics: Dict[str, Any]):
        """Plots grouped bar chart of the 3-Way Baseline Comparison."""
        labels = ["Baseline 1\nRaw GT Crop", "Baseline 2\nEnhanced GT Crop", "Baseline 3\nFull Pipeline"]
        exact_matches = [
            baseline_metrics["baseline_1_raw_gt_crop"]["exact_match_rate"] * 100,
            baseline_metrics["baseline_2_enhanced_gt_crop"]["exact_match_rate"] * 100,
            baseline_metrics["baseline_3_full_pipeline"]["exact_match_rate"] * 100
        ]
        char_accs = [
            baseline_metrics["baseline_1_raw_gt_crop"]["macro_char_accuracy"] * 100,
            baseline_metrics["baseline_2_enhanced_gt_crop"]["macro_char_accuracy"] * 100,
            baseline_metrics["baseline_3_full_pipeline"]["macro_char_accuracy"] * 100
        ]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        rects1 = ax.bar(x - width/2, exact_matches, width, label='Exact Match Accuracy (%)', color='#2E86C1')
        rects2 = ax.bar(x + width/2, char_accs, width, label='Character Accuracy (%)', color='#28B463')

        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title('Three-Way Baseline Comparison: Impact of Enhancement & Pipeline', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "baseline_comparison.png", dpi=200)
        plt.close(fig)

    def plot_positional_accuracy(self, pos_curve: Dict[str, float]):
        """Plots character recognition accuracy across plate index positions."""
        indices = list(range(len(pos_curve)))
        accuracies = [pos_curve[f"index_{i}"] * 100 for i in indices]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(indices, accuracies, marker='o', linewidth=2.5, markersize=7, color='#8E44AD')
        ax.set_xlabel('Character Index Position on License Plate (0 to 9)', fontsize=11)
        ax.set_ylabel('Recognition Accuracy (%)', fontsize=11)
        ax.set_title('License Plate Positional Accuracy Curve', fontsize=13, fontweight='bold')
        ax.set_xticks(indices)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.5)

        for i, acc in enumerate(accuracies):
            ax.annotate(f'{acc:.1f}%', (i, acc), textcoords="offset points", xytext=(0, 6),
                        ha='center', fontsize=8.5, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "positional_accuracy_curve.png", dpi=200)
        plt.close(fig)

    def plot_iou_distribution(self, ious: List[float]):
        """Plots YOLO IoU overlap distribution across ground-truth plates."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(ious, bins=20, range=(0.0, 1.0), color='#34495E', edgecolor='white', alpha=0.85)
        mean_iou = float(np.mean(ious)) if ious else 0.0
        ax.axvline(mean_iou, color='#E74C3C', linestyle='dashed', linewidth=2, label=f'Mean IoU ({mean_iou:.3f})')
        ax.set_xlabel('Intersection over Union (IoU)', fontsize=11)
        ax.set_ylabel('Frequency (Count)', fontsize=11)
        ax.set_title('YOLO License Plate Localization IoU Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "detection_iou_distribution.png", dpi=200)
        plt.close(fig)

    def generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Constructs publication-quality Markdown report."""
        meta = summary["evaluation_metadata"]
        det = summary["detection_evaluation"]
        geom = summary["geometry_evaluation"]
        base = summary["baseline_comparison"]
        e2e = summary["end_to_end_evaluation"]
        lat = summary["latency_profiling"]

        md = []
        md.append("# Forensic License Plate Super-Resolution, Restoration & OCR")
        md.append("## Comprehensive Standardized Pipeline Evaluation Report\n")
        md.append("---\n")

        # Executive Summary
        md.append("### Executive Performance Summary")
        md.append(f"- **Total Test Samples Evaluated**: **{meta['total_test_samples']} samples** (Denominator N = {meta['total_test_samples']})")
        md.append(f"- **End-to-End Exact Match Accuracy**: **{e2e['exact_match_rate'] * 100:.1f}%**")
        md.append(f"- **Mean (Macro) Character Error Rate (CER)**: **{e2e['mean_cer']:.3f}**")
        md.append(f"- **Normalized Macro Character Accuracy**: **{e2e['macro_char_accuracy'] * 100:.1f}%**")
        md.append(f"- **Corpus (Micro) CER**: **{e2e['corpus_cer']:.3f}**")
        md.append(f"- **Corpus (Micro) Character Accuracy**: **{e2e['corpus_char_accuracy'] * 100:.1f}%**")
        md.append(f"- **YOLO Detection Recall @ 0.50 IoU**: **{det['recall_at_0.50'] * 100:.1f}%** (True AP@0.50 = {det['ap_at_0.50']:.3f})")
        md.append(f"- **Mean Localization IoU**: **{det['mean_iou_all']:.3f}** (All plates)")
        md.append(f"- **Average Pipeline Latency**: **{lat['mean_total_ms']:.1f} ms / frame** (~**{lat['fps']:.1f} FPS** on {meta['device']})\n")
        md.append("---\n")

        # 1. Detection
        md.append("### 1. License Plate Detection & Localization (YOLOv8)")
        md.append("| Metric | Value | Description |")
        md.append("| :--- | :---: | :--- |")
        md.append(f"| **Precision @ 0.50 IoU** | {det['precision_at_0.50'] * 100:.1f}% | TP / (TP + FP) at IoU >= 0.50 |")
        md.append(f"| **Recall @ 0.50 IoU** | {det['recall_at_0.50'] * 100:.1f}% | TP / Total Manifest at IoU >= 0.50 |")
        md.append(f"| **F1-Score @ 0.50 IoU** | {det['f1_score_at_0.50']:.3f} | Harmonic mean of detection precision and recall |")
        md.append(f"| **True AP @ 0.50** | {det['ap_at_0.50']:.3f} | Area under Precision-Recall curve at IoU 0.50 (101-point COCO interpolation) |")
        md.append(f"| **True mAP @ 0.50:0.95** | {det['map_0.50_0.95']:.3f} | COCO mAP averaged across IoU thresholds 0.50 to 0.95 |")
        md.append(f"| **Mean IoU (All Plates)** | {det['mean_iou_all']:.3f} | Average box overlap across all GT plates (misses counted as 0.0) |")
        md.append(f"| **Mean IoU (Detected Only)** | {det['mean_iou_detected_only']:.3f} | Average box overlap conditioned on localization |")
        md.append(f"| **Detection Failure Rate** | {det['detection_failure_rate'] * 100:.1f}% | Percentage of images where plate was unlocalized |\n")

        md.append("**Detection Stratification by Plate Size:**")
        md.append("| Scale Bucket | Area Criteria | Test Count | Recall @ 0.50 | Mean IoU (All) | Mean IoU (Det) |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for scale, sdata in det["size_stratification"].items():
            criteria = "< 32² px" if scale == "small" else ("32² - 96² px" if scale == "medium" else ">= 96² px")
            md.append(f"| **{scale.capitalize()}** | {criteria} | {sdata['test_count']} | {sdata['recall_at_0.50'] * 100:.1f}% | {sdata['mean_iou_all']:.3f} | {sdata['mean_iou_detected']:.3f} |")
        md.append("\n---\n")

        # 2. Geometric Rectification
        md.append("### 2. Geometric Rectification & Deskewing")
        md.append("| Metric | Measured Value | Target Standard |")
        md.append("| :--- | :---: | :---: |")
        md.append(f"| **Mean Residual Skew Angle** | **{geom['mean_residual_skew_deg']:.2f}°** | < 1.0° |")
        md.append(f"| **Mean Aspect Ratio Error** | **{geom['mean_aspect_ratio_error'] * 100:.1f}%** | < 15.0% |\n")
        md.append("---\n")

        # 3. Baseline Comparison
        md.append("### 3. Three-Way Baseline Comparison: Progressive Enhancement Impact")
        md.append("| Evaluation Mode | Description | Exact Match | Mean CER | Char Accuracy | Corpus CER |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for bkey, bdata in base.items():
            md.append(f"| **{bdata['name'].split('->')[0].strip()}** | {bdata['name']} | **{bdata['exact_match_rate'] * 100:.1f}%** | **{bdata['mean_cer']:.3f}** | **{bdata['macro_char_accuracy'] * 100:.1f}%** | {bdata['corpus_cer']:.3f} |")
        md.append("\n---\n")

        # 4. OCR Confusion
        md.append("### 4. OCR Error & Character Confusion Analysis")
        md.append("#### Top Character Substitution Pairs")
        md.append("| Substitution (Actual -> Predicted) | Occurrences | Typical Root Cause |")
        md.append("| :---: | :---: | :--- |")
        for pair, cnt in list(e2e["top_substitutions"].items())[:6]:
            md.append(f"| **{pair}** | {cnt} | Morphological character geometry similarity |")

        md.append("\n#### Alignment Error Breakdown")
        subs_sum = sum(e2e["top_substitutions"].values())
        dels_sum = sum(e2e["top_deletions"].values())
        inss_sum = sum(e2e["top_insertions"].values())
        md.append(f"- **Total Edit Distance (Errors)**: {e2e['total_edit_distance']} character errors across {e2e['total_gt_characters']} ground-truth characters")
        md.append(f"- **Top Deletions**: {', '.join([f'{k} ({v}x)' for k, v in e2e['top_deletions'].items()]) if e2e['top_deletions'] else 'None'}")
        md.append(f"- **Top Insertions**: {', '.join([f'{k} ({v}x)' for k, v in e2e['top_insertions'].items()]) if e2e['top_insertions'] else 'None'}\n")
        md.append("---\n")

        # 5. Runtime Profile
        md.append("### 5. Runtime & Throughput Profile")
        md.append(f"- **Average Frame Latency**: **{lat['mean_total_ms']:.1f} ms**")
        md.append(f"- **System Throughput**: **{lat['fps']:.1f} frames / second**\n")
        md.append("| Pipeline Stage | Mean Execution Time (ms) |")
        md.append("| :--- | :---: |")
        for stg, ms_val in lat["stage_latencies_ms"].items():
            clean_name = stg.replace("_sec", "").replace("_", " ").title()
            md.append(f"| **{clean_name}** | {ms_val:.2f} ms |")

        md.append("\n---\n*Report generated autonomously by ForensicPipelineEvaluator.*")
        return "\n".join(md)


def main():
    device = torch.device(EVAL_DEVICE)
    print("=" * 70)
    print("FORENSIC PIPELINE EVALUATION SUITE")
    print("=" * 70)
    print(f"[*] Manifest Path : {os.path.abspath(EVAL_MANIFEST_PATH)}")
    print(f"[*] Output Dir    : {os.path.abspath(EVAL_OUTPUT_DIR)}")
    print(f"[*] Compute Device: {device}")
    print(f"[*] Sample Limit  : {'All' if EVAL_LIMIT_SAMPLES is None else EVAL_LIMIT_SAMPLES}")
    print("=" * 70 + "\n")

    evaluator = ForensicPipelineEvaluator(
        manifest_path=EVAL_MANIFEST_PATH,
        output_dir=EVAL_OUTPUT_DIR,
        device=device
    )

    evaluator.run_full_evaluation(limit_samples=EVAL_LIMIT_SAMPLES)


if __name__ == "__main__":
    main()
