"""
scripts/pipeline.py: Main Forensic Image Pipeline Coordinator
==============================================================
Central orchestrator that coordinates all 10 specialized modules in a structured,
traceable workflow. Supports module swapping, module-wise latency profiling,
and diagnostic inspection payloads.
"""

import time
from typing import Optional, Dict, Any
import cv2
import torch
import torch.nn as nn
import numpy as np

from scripts.detection import AutonomousPlateDetector
from scripts.deblurring import DeblurringModule
from scripts.super_resolution import RealESRGANEngine
from scripts.geometry import PlateGeometryModule
from scripts.quality_assessment import QualityAssessmentModule
from scripts.routing import AdaptiveEnhancementRouter
from scripts.ocr_recognition import DualPathPaddleOCREngine
from scripts.ocr_candidate_selection import OCRCandidateSelectionModule
from scripts.evaluation_metrics import ForensicEvaluator


class MainImagePipeline:
    """
    Main Image Pipeline:
    Coordinates and orchestrates all individual modules in a structured,
    traceable workflow. Allows assessing module-wise execution latencies,
    swapping modules, and retrieving diagnostic visualization payloads.
    """
    def __init__(
        self,
        detector: Optional[AutonomousPlateDetector] = None,
        deblurrer: Optional[DeblurringModule] = None,
        sr_engine: Optional[RealESRGANEngine] = None,
        geometry_module: Optional[PlateGeometryModule] = None,
        qa_module: Optional[QualityAssessmentModule] = None,
        router: Optional[AdaptiveEnhancementRouter] = None,
        ocr_engine: Optional[DualPathPaddleOCREngine] = None,
        candidate_selector: Optional[OCRCandidateSelectionModule] = None,
        evaluator: Optional[ForensicEvaluator] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] [MainPipeline] Initializing Pipeline Coordinator on Device: {self.device}...")

        self.detector = detector or AutonomousPlateDetector(device=self.device)
        self.deblurrer = deblurrer or DeblurringModule(self.device)
        self.sr_engine = sr_engine or RealESRGANEngine(self.device)
        self.geometry = geometry_module or PlateGeometryModule()
        self.qa = qa_module or QualityAssessmentModule()
        self.router = router or AdaptiveEnhancementRouter()
        self.ocr = ocr_engine or DualPathPaddleOCREngine()
        self.selector = candidate_selector or OCRCandidateSelectionModule()
        self.evaluator = evaluator or ForensicEvaluator()

        print("[✓] [MainPipeline] Pipeline Coordinator fully assembled and ready.")

    def run(
        self,
        input_bgr: np.ndarray,
        reference_clean_bgr: Optional[np.ndarray] = None,
        apply_nafnet: bool = True,
        track_latency: bool = True
    ) -> Dict[str, Any]:
        """
        Coordinates full forensic pipeline on an input image.
        """
        timings = {
            "deblurring_sec": 0.0,
            "detection_sec": 0.0,
            "geometry_sec": 0.0,
            "qa_sec": 0.0,
            "routing_and_restoration_sec": 0.0,
            "ocr_and_selection_sec": 0.0,
            "evaluation_sec": 0.0
        }
        h, w = input_bgr.shape[:2]

        # STAGE 1: Deblurring Module (Full-Frame Learned Motion Deblurring)
        t0 = time.perf_counter()
        if apply_nafnet:
            try:
                scene_restored = self.deblurrer.deblur(input_bgr)
            except Exception as e:
                print(f"[!] [MainPipeline] Deblurring bypassed due to error: {e}")
                scene_restored = input_bgr.copy()
        else:
            scene_restored = input_bgr.copy()
        timings["deblurring_sec"] = time.perf_counter() - t0

        # STAGE 2: Detection Module (YOLO Plate Localization)
        t0 = time.perf_counter()
        detected_box, det_conf = self.detector.detect(scene_restored)
        if detected_box is None and apply_nafnet:
            # Fallback: check if plate is detectable on original input_bgr
            detected_box, det_conf = self.detector.detect(input_bgr)
            if detected_box is not None:
                scene_restored = input_bgr.copy()
        timings["detection_sec"] = time.perf_counter() - t0

        if detected_box is None:
            return {
                "final_text": "NOT_DETECTED",
                "final_conf": 0.0,
                "detection_bbox": None,
                "loc_status": "LOCALIZATION_FAILED",
                "timings": timings,
                "vis": None
            }

        loc_status = f"YOLO_DETECTED(conf={det_conf:.2f})"

        # STAGE 3: Geometry Module (ROI Extraction & Perspective Rectification)
        t0 = time.perf_counter()
        plate_crop, (x1, y1, x2, y2) = self.geometry.extract_padded_roi(scene_restored, detected_box)
        raw_crop = input_bgr[y1:y2, x1:x2]
        clean_crop = reference_clean_bgr[y1:y2, x1:x2] if reference_clean_bgr is not None else None

        rectified_crop, M_geom, (target_w, target_h), geom_status = self.geometry.rectify(plate_crop)
        timings["geometry_sec"] = time.perf_counter() - t0

        # STAGE 4: Quality Assessment Module (Comprehensive Multi-Indicator QA)
        t0 = time.perf_counter()
        qa_metrics = self.qa.assess(raw_crop, rectified_crop, quad_dims=self.geometry.last_quad_dims)
        timings["qa_sec"] = time.perf_counter() - t0

        # STAGE 5: Routing Module & Classical Restoration Modules
        t0 = time.perf_counter()
        variants, action_summary = self.router.generate_ladder(rectified_crop, qa_metrics, self.sr_engine)
        timings["routing_and_restoration_sec"] = time.perf_counter() - t0

        # STAGE 6: OCR Recognition & Candidate Selection Modules
        t0 = time.perf_counter()
        candidate_pool = self.selector.collect_candidates(variants, raw_crop, self.ocr)
        final_text, final_conf, certainty_status, ambiguities = self.selector.select(candidate_pool)
        timings["ocr_and_selection_sec"] = time.perf_counter() - t0

        raw_ocr_entry = next((c for c in candidate_pool if c["variant"] == "Raw_Degraded_Original"), None)
        raw_ocr_text = raw_ocr_entry["text"] if raw_ocr_entry else "NOT_DETECTED"
        raw_ocr_conf = raw_ocr_entry["conf"] if raw_ocr_entry else 0.0
        winning_variant = next((c["variant"] for c in candidate_pool if c["text"] == final_text), (candidate_pool[0]["variant"] if candidate_pool else "Rectified_Unprocessed"))

        # STAGE 7: Evaluation Module (Fidelity & Metric Calculation)
        t0 = time.perf_counter()
        candidate_metrics = {}
        best_metric = {"psnr_before": 0.0, "psnr_after": 0.0, "ssim_before": 0.0, "ssim_after": 0.0}

        if clean_crop is not None:
            clean_aligned = self.geometry.align_reference(clean_crop, M_geom, (target_w, target_h), geom_status)
            raw_aligned = self.geometry.align_reference(raw_crop, M_geom, (target_w, target_h), geom_status)
            candidate_metrics = self.evaluator.evaluate_candidates_fidelity(variants, clean_aligned, raw_aligned, border_pad=18)
            best_metric = candidate_metrics.get("Adaptive_Targeted", candidate_metrics.get("Rectified_Unprocessed", {}))
        timings["evaluation_sec"] = time.perf_counter() - t0

        display_crop = variants.get("display_crop", rectified_crop)
        latency_breakdown = {k: v * 1000.0 for k, v in timings.items()}
        latency_breakdown["total_pipeline_ms"] = sum(timings.values()) * 1000.0

        return {
            "final_text": final_text,
            "final_conf": final_conf,
            "raw_ocr_text": raw_ocr_text,
            "raw_ocr_conf": raw_ocr_conf,
            "certainty_status": certainty_status,
            "selection_strategy": certainty_status,
            "winning_candidate": winning_variant,
            "detection_bbox": (x1, y1, x2, y2),
            "ambiguities": ambiguities,
            "qa_metrics": qa_metrics,
            "action_summary": action_summary,
            "geom_status": geom_status,
            "loc_status": loc_status,
            "candidate_metrics": candidate_metrics,
            "best_metric": best_metric,
            "candidate_pool": candidate_pool,
            "timings": timings if track_latency else None,
            "latency_breakdown": latency_breakdown if track_latency else None,
            "vis": {
                "original_image": input_bgr.copy(),
                "scene_restored": scene_restored,
                "raw_crop": raw_crop,
                "rectified_crop": rectified_crop,
                "display_crop": display_crop,
                "candidate_variants": variants,
                "bbox": (x1, y1, x2, y2)
            }
        }


def run_pipeline_on_image(
    input_bgr: np.ndarray,
    reference_clean_bgr: np.ndarray,
    detector: AutonomousPlateDetector,
    nafnet_model: nn.Module,
    sr_engine: RealESRGANEngine,
    paddle_engine: DualPathPaddleOCREngine,
    device: torch.device,
    apply_nafnet: bool = True
) -> dict:
    """
    Backward-compatible functional API wrapping the modular MainImagePipeline.
    """
    deblurrer = DeblurringModule(device=device, model=nafnet_model)

    pipeline = MainImagePipeline(
        detector=detector,
        deblurrer=deblurrer,
        sr_engine=sr_engine,
        ocr_engine=paddle_engine,
        device=device
    )
    return pipeline.run(
        input_bgr=input_bgr,
        reference_clean_bgr=reference_clean_bgr,
        apply_nafnet=apply_nafnet
    )

