"""
scripts package: Modular Forensic License Plate Pipeline
=========================================================
A clean, modular architecture for forensic vehicle license plate localization,
learned deblurring, super-resolution, geometric rectification, quality-adaptive
routing, OCR transcription, evaluation, and diagnostic visualization.
"""

from scripts.detection import AutonomousPlateDetector
from scripts.deblurring import DeblurringModule, NAFNet, load_official_nafnet
from scripts.super_resolution import RealESRGANEngine, StandaloneRRDBNet
from scripts.geometry import PlateGeometryModule, order_quad_points_robust, perspective_rectify_plate
from scripts.quality_assessment import QualityAssessmentModule, assess_plate_quality_comprehensive
from scripts.restoration import (
    ClassicalRestorationSuite,
    NLMDenoisingRestoration,
    CLAHERestoration,
    UnsharpMaskSharpeningRestoration,
    MorphologicalReinkingRestoration,
    make_odd
)
from scripts.routing import AdaptiveEnhancementRouter, generate_genuinely_adaptive_ladder
from scripts.ocr_recognition import DualPathPaddleOCREngine
from scripts.ocr_candidate_selection import OCRCandidateSelectionModule, validate_and_select_ocr_candidate
from scripts.evaluation_metrics import ForensicEvaluator, levenshtein_distance, compute_cer
from scripts.pipeline import MainImagePipeline, run_pipeline_on_image
from scripts.visualization import render_visual_card, run_visual_pipeline, visualize_single_image
from scripts.video_pipeline import (
    VisualFeatureExtractor,
    calculate_sharpness,
    compute_plate_clarity,
    merge_duplicate_vehicles,
    extract_unique_vehicles_no_ocr,
    filter_and_deduplicate_vehicles,
    display_final_vehicle_ocr_summary,
    ForensicVideoPipeline,
    process_video_and_recognize_plates
)
from scripts.data_prep import prepare_unified_dataset
from scripts.evaluation import ForensicPipelineEvaluator, DetectionEvaluator, BaselineComparisonEngine, OCREvaluator

__all__ = [
    "AutonomousPlateDetector",
    "DeblurringModule",
    "NAFNet",
    "load_official_nafnet",
    "RealESRGANEngine",
    "StandaloneRRDBNet",
    "PlateGeometryModule",
    "order_quad_points_robust",
    "perspective_rectify_plate",
    "QualityAssessmentModule",
    "assess_plate_quality_comprehensive",
    "ClassicalRestorationSuite",
    "NLMDenoisingRestoration",
    "CLAHERestoration",
    "UnsharpMaskSharpeningRestoration",
    "MorphologicalReinkingRestoration",
    "make_odd",
    "AdaptiveEnhancementRouter",
    "generate_genuinely_adaptive_ladder",
    "DualPathPaddleOCREngine",
    "OCRCandidateSelectionModule",
    "validate_and_select_ocr_candidate",
    "ForensicEvaluator",
    "levenshtein_distance",
    "compute_cer",
    "MainImagePipeline",
    "run_pipeline_on_image",
    "render_visual_card",
    "run_visual_pipeline",
    "visualize_single_image",
    "VisualFeatureExtractor",
    "calculate_sharpness",
    "compute_plate_clarity",
    "merge_duplicate_vehicles",
    "extract_unique_vehicles_no_ocr",
    "filter_and_deduplicate_vehicles",
    "display_final_vehicle_ocr_summary",
    "ForensicVideoPipeline",
    "process_video_and_recognize_plates",
    "prepare_unified_dataset",
    "ForensicPipelineEvaluator",
    "DetectionEvaluator",
    "BaselineComparisonEngine",
    "OCREvaluator"
]

