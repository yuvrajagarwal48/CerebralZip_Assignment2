#!/usr/bin/env python3
"""
forensic_pipeline.py: Modular Forensic License Plate Pipeline Facade
=====================================================================
Central entry point for the forensic pipeline. Re-exports all components from
the 'scripts' package for clean modularity and seamless backward compatibility.

Modules available:
  - AutonomousPlateDetector           (scripts.detection)
  - DeblurringModule, NAFNet           (scripts.deblurring)
  - RealESRGANEngine, StandaloneRRDBNet (scripts.super_resolution)
  - PlateGeometryModule               (scripts.geometry)
  - QualityAssessmentModule           (scripts.quality_assessment)
  - ClassicalRestorationSuite          (scripts.restoration)
  - AdaptiveEnhancementRouter         (scripts.routing)
  - DualPathPaddleOCREngine           (scripts.ocr_recognition)
  - OCRCandidateSelectionModule       (scripts.ocr_candidate_selection)
  - ForensicEvaluator                 (scripts.evaluation_metrics)
  - MainImagePipeline                 (scripts.pipeline)
  - render_visual_card                (scripts.visualization)
  - run_visual_pipeline               (scripts.visualization)
  - visualize_single_image            (scripts.visualization)
"""

import os
from scripts import (
    AutonomousPlateDetector,
    DeblurringModule,
    NAFNet,
    load_official_nafnet,
    RealESRGANEngine,
    StandaloneRRDBNet,
    PlateGeometryModule,
    order_quad_points_robust,
    perspective_rectify_plate,
    QualityAssessmentModule,
    assess_plate_quality_comprehensive,
    ClassicalRestorationSuite,
    NLMDenoisingRestoration,
    CLAHERestoration,
    UnsharpMaskSharpeningRestoration,
    MorphologicalReinkingRestoration,
    make_odd,
    AdaptiveEnhancementRouter,
    generate_genuinely_adaptive_ladder,
    DualPathPaddleOCREngine,
    OCRCandidateSelectionModule,
    validate_and_select_ocr_candidate,
    ForensicEvaluator,
    levenshtein_distance,
    compute_cer,
    MainImagePipeline,
    run_pipeline_on_image,
    render_visual_card,
    run_visual_pipeline,
    visualize_single_image,
    VisualFeatureExtractor,
    calculate_sharpness,
    compute_plate_clarity,
    merge_duplicate_vehicles,
    extract_unique_vehicles_no_ocr,
    filter_and_deduplicate_vehicles,
    display_final_vehicle_ocr_summary,
    ForensicVideoPipeline,
    process_video_and_recognize_plates,
    prepare_unified_dataset,
    ForensicPipelineEvaluator,
    DetectionEvaluator,
    BaselineComparisonEngine,
    OCREvaluator
)

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


if __name__ == "__main__":
    manifest = "./data/dataset_prepared/manifest.json"
    if os.path.exists(manifest):
        run_visual_pipeline(
            manifest_path=manifest,
            num_samples=5,
            variant_index=0,
            save_dir="./results/visualizations"
        )
    else:
        print(f"[*] Pipeline initialized successfully. Manifest not found at: '{manifest}'.")
