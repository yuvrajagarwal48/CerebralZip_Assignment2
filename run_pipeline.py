#!/usr/bin/env python3
"""
run_pipeline.py: Master Pipeline Execution Entrypoint
=====================================================
Unified interface to execute all pipeline stages with sensible hardcoded defaults:
  1. Image Mode: Single image or batch forensic super-resolution & OCR.
  2. Video Mode: End-to-end traffic video ANPR, tracking, Re-ID deduplication.
  3. Evaluation Mode: Standardized 3-way baseline benchmarks & AP/CER metrics.
  4. Prep Mode: Unified real-world dataset ingestion and manifest generation.
  5. App Mode: Launches the Streamlit interactive dashboard.

Usage Examples:
  python run_pipeline.py                    # Default: Runs image pipeline on sample
  python run_pipeline.py --mode image       # Run image pipeline on default sample
  python run_pipeline.py --mode video       # Run video pipeline on traffic footage
  python run_pipeline.py --mode eval        # Run 3-way baseline evaluation
  python run_pipeline.py --mode prep        # Prepare/consolidate dataset
  python run_pipeline.py --mode app         # Launch Streamlit web app
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import cv2
import torch

from scripts.pipeline import MainImagePipeline
from scripts.video_pipeline import ForensicVideoPipeline, DEFAULT_VEHICLE_MODEL, DEFAULT_VIDEO_OUTPUT_DIR
from scripts.evaluation import ForensicPipelineEvaluator, EVAL_MANIFEST_PATH, EVAL_OUTPUT_DIR
from scripts.data_prep import prepare_unified_dataset, DATA_INPUT_DIR, DATA_OUTPUT_DIR
from scripts.visualization import render_visual_card

# Default Paths
DEFAULT_SAMPLE_IMAGE = "data/dataset_prepared/images/AN1.jpg"
DEFAULT_SAMPLE_VIDEO = "data/videos/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"


def run_image(image_path: str = DEFAULT_SAMPLE_IMAGE, output_dir: str = "results/visualizations"):
    """Runs end-to-end forensic restoration and OCR on a single image."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FORENSIC IMAGE PIPELINE EXECUTION")
    print("=" * 70)
    print(f"[*] Compute Device : {device}")
    
    # Fallback to any available image in dataset if default doesn't exist
    if not os.path.exists(image_path):
        cands = list(Path("data/dataset_prepared/images").glob("*.jpg")) + list(Path("data/dataset_prepared/images").glob("*.png"))
        if cands:
            image_path = str(cands[0])
            print(f"[!] Target not found. Using available sample: {image_path}")
        else:
            raise FileNotFoundError(f"Image not found at '{image_path}'. Run data prep first.")

    print(f"[*] Input Image    : {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image at '{image_path}'")

    pipeline = MainImagePipeline(device=device)
    result = pipeline.run(input_bgr=img_bgr, reference_clean_bgr=None, track_latency=True)

    print("\n" + "=" * 50)
    print("RECOGNITION RESULTS:")
    print("=" * 50)
    print(f"Recognized Plate Number : '{result['final_text']}'")
    print(f"Selection Strategy      : {result['selection_strategy']}")
    print(f"Winning Ladder Variant  : {result['winning_candidate']}")
    print(f"Plate Bounding Box      : {result['detection_bbox']}")
    print(f"Total Pipeline Latency  : {result['latency_breakdown']['total_pipeline_ms']:.1f} ms")
    print("=" * 50)

    # Save 4-panel diagnostic visual card
    os.makedirs(output_dir, exist_ok=True)
    save_card_path = os.path.join(output_dir, f"visual_card_{Path(image_path).stem}.png")
    render_visual_card(
        pipeline_result=result,
        gt_text="",
        sample_id=Path(image_path).stem,
        tag="Single_Run",
        save_path=save_card_path,
        show_fig=False
    )
    print(f"\n[✓] Diagnostic Visual Card saved to -> {save_card_path}")
    return result


def run_video(video_path: str = DEFAULT_SAMPLE_VIDEO, output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR, max_frames: int = 250):
    """Runs video tracking, deep re-ID deduplication, and forensic OCR."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FORENSIC VIDEO ANPR PIPELINE EXECUTION")
    print("=" * 70)
    print(f"[*] Compute Device : {device}")

    # Fallback to any available video if default doesn't exist
    if not os.path.exists(video_path):
        vids = list(Path("data/videos").glob("*.mp4")) + list(Path("data/videos").glob("*.avi"))
        if vids:
            video_path = str(vids[0])
            print(f"[!] Target video not found. Using available video: {video_path}")
        else:
            raise FileNotFoundError(f"Video not found at '{video_path}'. Place an MP4 in data/videos/.")

    print(f"[*] Input Video    : {video_path}")
    print(f"[*] Output Dir     : {output_dir}")
    print(f"[*] Frame Limit    : {max_frames}")

    video_pipe = ForensicVideoPipeline(device=device)
    verified_cars = video_pipe.process_video(
        video_path=video_path,
        output_dir=output_dir,
        vehicle_model_name=DEFAULT_VEHICLE_MODEL,
        max_frames=max_frames,
        save_visual_summary=True,
        show_summary=False
    )

    print("\n" + "=" * 50)
    print(f"VIDEO ANPR SUMMARY: Found {len(verified_cars)} unique verified vehicle(s):")
    print("=" * 50)
    for idx, car in enumerate(verified_cars, 1):
        print(f"  [{idx}] Track ID: {car['track_id']} | Plate OCR: '{car['ocr_text']}' | Best Score: {car['clarity_score']:.1f}")
    print(f"\n[✓] Results saved to -> {output_dir}/verified_vehicles_anpr_summary.png")
    return verified_cars


def run_evaluation(limit_samples: int = 5, output_dir: str = EVAL_OUTPUT_DIR):
    """Runs standardized evaluation suite and 3-way progressive baselines."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STANDARDIZED BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Sample Limit   : {limit_samples if limit_samples else 'All'}")
    print(f"[*] Output Dir     : {output_dir}")

    evaluator = ForensicPipelineEvaluator(
        manifest_path=EVAL_MANIFEST_PATH,
        output_dir=output_dir,
        device=device
    )
    summary = evaluator.run_full_evaluation(limit_samples=limit_samples)
    print(f"\n[✓] Evaluation completed. Master report -> {output_dir}/EVALUATION_REPORT.md")
    return summary


def run_prep():
    """Consolidates raw dataset into collision-free unified dataset."""
    print("=" * 70)
    print("UNIFIED DATASET PREPARATION")
    print("=" * 70)
    return prepare_unified_dataset(input_dir=DATA_INPUT_DIR, output_dir=DATA_OUTPUT_DIR)


def run_app():
    """Launches Streamlit dashboard."""
    print("[*] Launching Streamlit Forensic Application...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)


def run_compare(output_dir: str = "results/comparative_cards"):
    """Runs 15-sample comparative forensic evaluation and saves composite cards."""
    from scripts.generate_comparative_cards import generate_all_15_comparative_cards
    return generate_all_15_comparative_cards(output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Forensic License Plate Recognition Pipeline Coordinator")
    parser.add_argument(
        "--mode",
        choices=["image", "video", "eval", "prep", "app", "compare"],
        default="image",
        help="Pipeline execution mode (default: image)"
    )
    parser.add_argument("--image", type=str, default=DEFAULT_SAMPLE_IMAGE, help="Path to input image for image mode")
    parser.add_argument("--video", type=str, default=DEFAULT_SAMPLE_VIDEO, help="Path to input video for video mode")
    parser.add_argument("--limit", type=int, default=5, help="Sample limit for eval mode or frames for video mode")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    if args.mode == "image":
        out = args.output or "results/visualizations"
        run_image(image_path=args.image, output_dir=out)
    elif args.mode == "video":
        out = args.output or DEFAULT_VIDEO_OUTPUT_DIR
        run_video(video_path=args.video, output_dir=out, max_frames=args.limit)
    elif args.mode == "eval":
        out = args.output or EVAL_OUTPUT_DIR
        run_evaluation(limit_samples=args.limit, output_dir=out)
    elif args.mode == "prep":
        run_prep()
    elif args.mode == "app":
        run_app()
    elif args.mode == "compare":
        out = args.output or "results/comparative_cards"
        run_compare(output_dir=out)


if __name__ == "__main__":
    main()

