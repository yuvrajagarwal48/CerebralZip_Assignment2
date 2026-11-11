#!/usr/bin/env python3
"""
scripts/generate_step_progression_cards.py: Step-by-Step Pipeline Progression Visualizer
========================================================================================
Runs the forensic pipeline on the exact 15 images used in comparative cards,
extracting and visualizing intermediate output at EVERY processing step in a single
comprehensive diagnostic figure per sample:
  Row 1 (Scene & Detection Pipeline):
    - Step 1: Original Image (Raw input scene / traffic frame)
    - Step 2: NAFNet Deblurred Scene (Full-frame learned motion deblurring + YOLO box)
    - Step 3: YOLO Plate Crop (Localized raw bounding box ROI)
    - Step 4: Geometric Homography Rectification (Quadrilateral deskewing, 4:1 canvas)
  Row 2 (Restoration, Super-Resolution & OCR Pipeline):
    - Step 5: Real-ESRGAN Super-Resolution (Deep RRDBNet FP32 upscaling)
    - Step 6: Adaptive Contrast & Denoising (CIE LAB space NLM & CLAHE equalization)
    - Step 7: Final Enhanced Plate (Unsharp masking stroke sharpening)
    - Step 8: Final Forensic Result (OCR transcription, ground truth, metrics & badge)

Outputs are saved to results/step_by_step_visualizations/.
Also generates step_progression_summary.json and STEP_BY_STEP_REPORT.md.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from scripts.pipeline import MainImagePipeline
from scripts.evaluation_metrics import ForensicEvaluator

DEFAULT_OUTPUT_DIR = "results/step_by_step_visualizations"
DEFAULT_MANIFEST_PATH = "data/dataset_prepared/manifest.json"

# The exact 15 images utilized in comparative cards
TARGET_SAMPLE_FILES = [
    # 8 Real-world dataset images
    "data/dataset_prepared/images/AR13.jpg",
    "data/dataset_prepared/images/AN1.jpg",
    "data/dataset_prepared/images/AN2.jpg",
    "data/dataset_prepared/images/AN4.jpg",
    "data/dataset_prepared/images/AN5.jpg",
    "data/dataset_prepared/images/AN6.jpg",
    "data/dataset_prepared/images/AN7.jpg",
    "data/dataset_prepared/images/AN10.jpg",
    # 7 Traffic video ANPR candidate frames
    "results/video_anpr/raw_video_candidates/car_1_frame_88.jpg",
    "results/video_anpr/raw_video_candidates/car_3_frame_221.jpg",
    "results/video_anpr/raw_video_candidates/car_4_frame_260.jpg",
    "results/video_anpr/raw_video_candidates/car_6_frame_265.jpg",
    "results/video_anpr/raw_video_candidates/car_7_frame_306.jpg",
    "results/video_anpr/raw_video_candidates/car_8_frame_315.jpg",
    "results/video_anpr/raw_video_candidates/car_10_frame_335.jpg",
]


def load_manifest_gt(manifest_path: str = DEFAULT_MANIFEST_PATH) -> Dict[str, str]:
    """Loads ground truth map {filename: ground_truth_text} if manifest exists."""
    gt_map = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                fn = item.get("filename")
                txt = item.get("ground_truth_text") or item.get("text")
                if fn and txt:
                    gt_map[fn] = txt
                    gt_map[Path(fn).stem] = txt
        except Exception:
            pass
    return gt_map


def compute_plate_metrics(raw_crop: np.ndarray, enhanced_crop: np.ndarray) -> Dict[str, float]:
    """Computes PSNR, SSIM, and Laplacian sharpness gain between raw and enhanced plates."""
    if raw_crop is None or enhanced_crop is None or raw_crop.size == 0 or enhanced_crop.size == 0:
        return {"psnr": 0.0, "ssim": 0.0, "sharpness_before": 0.0, "sharpness_after": 0.0, "sharpness_gain_pct": 0.0}

    th, tw = enhanced_crop.shape[:2]
    raw_aligned = cv2.resize(raw_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    try:
        psnr_val = float(peak_signal_noise_ratio(raw_aligned, enhanced_crop, data_range=255))
    except Exception:
        psnr_val = 0.0

    try:
        ssim_val = float(structural_similarity(raw_aligned, enhanced_crop, channel_axis=2, data_range=255))
    except Exception:
        ssim_val = 0.0

    var_before = float(cv2.Laplacian(cv2.cvtColor(raw_aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    var_after = float(cv2.Laplacian(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    gain_pct = ((var_after - var_before) / max(1.0, var_before)) * 100.0 if var_before > 0 else 0.0

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "sharpness_before": var_before,
        "sharpness_after": var_after,
        "sharpness_gain_pct": gain_pct
    }


def extract_pipeline_stages(pipeline: MainImagePipeline, img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Executes the pipeline stage-by-stage and retains all intermediate visual outputs.
    """
    # Stage 1: Input Original
    orig_bgr = img_bgr.copy()

    # Stage 2: NAFNet Motion Deblurring
    try:
        scene_naf = pipeline.deblurrer.deblur(img_bgr)
    except Exception as e:
        print(f"[!] Deblurring fallback: {e}")
        scene_naf = img_bgr.copy()

    # Rigorous non-black validity check
    if scene_naf is None or (scene_naf.mean() < 3.0 and img_bgr.mean() >= 5.0):
        print("[!] NAFNet produced degraded or black output; reverting to original scene.")
        scene_naf = img_bgr.copy()

    # Stage 3: YOLO Detection
    box, conf = pipeline.detector.detect(scene_naf)
    if box is None:
        box, conf = pipeline.detector.detect(img_bgr)
        if box is not None:
            scene_naf = img_bgr.copy()

    scene_annotated = scene_naf.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(scene_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            scene_annotated, f"PLATE {conf:.2f}", (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
        )
        plate_crop, _ = pipeline.geometry.extract_padded_roi(scene_naf, box)
        raw_crop = img_bgr[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, img_bgr.shape[1], img_bgr.shape[0]
        plate_crop = img_bgr.copy()
        raw_crop = img_bgr.copy()

    # Stage 4: Geometric Homography Rectification
    rect_crop, M, dims, status = pipeline.geometry.rectify(plate_crop)

    # Stage 5: Real-ESRGAN Deep Super-Resolution (2x scale for display)
    sr_crop, sr_lbl = pipeline.sr_engine.upscale(rect_crop, target_scale=2.0)

    # Stage 6: Quality-Adaptive CIE LAB Contrast & Denoising
    qa = pipeline.qa.assess(raw_crop, rect_crop)
    lab = cv2.cvtColor(sr_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_nlm = pipeline.router.restoration.nlm.apply(l) if qa["needs_denoise"] else l
    l_clahe = pipeline.router.restoration.clahe.apply(l_nlm) if qa["needs_clahe"] else l_nlm
    l_norm = pipeline.router.restoration.normalize_percentile(l_clahe, 1.0, 99.0)
    clahe_bgr = cv2.cvtColor(l_norm, cv2.COLOR_GRAY2BGR)

    # Stage 7: Unsharp Masking & Stroke Sharpening (Final Enhanced Plate)
    l_sharp = pipeline.router.restoration.sharpen.apply(l_norm)
    enhanced_crop = cv2.cvtColor(l_sharp, cv2.COLOR_GRAY2BGR)

    # Stage 8: OCR Recognition & Consensus Selection
    full_res = pipeline.run(input_bgr=img_bgr, track_latency=True)

    metrics = compute_plate_metrics(raw_crop, enhanced_crop)

    return {
        "step1_original": orig_bgr,
        "step2_nafnet": scene_annotated,
        "step3_raw_crop": raw_crop,
        "step4_rectified": rect_crop,
        "step5_super_resolved": sr_crop,
        "step6_clahe_denoise": clahe_bgr,
        "step7_enhanced": enhanced_crop,
        "pipeline_res": full_res,
        "qa_metrics": qa,
        "metrics": metrics,
        "box": box,
        "rect_status": status,
        "quad_dims": pipeline.geometry.last_quad_dims
    }


def render_step_progression_card(
    idx: int,
    image_path: str,
    stages: Dict[str, Any],
    gt_text: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders an 8-panel step-by-step pipeline progression figure:
      Row 1: Step 1 (Original) -> Step 2 (NAFNet) -> Step 3 (YOLO Crop) -> Step 4 (Homography Rectified)
      Row 2: Step 5 (Real-ESRGAN SR) -> Step 6 (CLAHE & Denoise) -> Step 7 (Final Enhanced Plate) -> Step 8 (Final OCR Card)
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5))
    fig.patch.set_facecolor("#0a0f1d")

    fname = os.path.basename(image_path)
    source_tag = "Dataset Image" if "dataset" in image_path.lower() else "Video ANPR Frame"
    fig.suptitle(
        f"Forensic Sample #{idx:02d}: {fname} ({source_tag}) — End-to-End Stage Progression",
        fontsize=14, fontweight="bold", color="#f8fafc", y=0.985
    )

    rect_ar = stages.get("quad_dims", {}).get("aspect_ratio")
    if rect_ar is None and stages.get("step4_rectified") is not None:
        rh, rw = stages["step4_rectified"].shape[:2]
        rect_ar = float(rw) / max(1e-4, float(rh))
    step4_sub = f"Corner-Derived Homography (AR={rect_ar:.2f})" if rect_ar else "Corner-Derived Homography"

    stage_titles = [
        # Row 1
        ("Step 1: Original Image", "Raw Source Capture", "#38bdf8"),
        ("Step 2: NAFNet Deblurred Scene", "GoPro Learned Deblurring + YOLO", "#818cf8"),
        ("Step 3: YOLO Plate Localization", "Raw ROI Bounding Box Crop", "#fbbf24"),
        ("Step 4: Geometric Deskewing", step4_sub, "#a7f3d0"),
        # Row 2
        ("Step 5: Real-ESRGAN x4 (2x Scaled)", "RRDBNet Deep Super-Resolution (FP32)", "#f472b6"),
        ("Step 6: Adaptive Denoise & Contrast", "CIE LAB Space (NLM + CLAHE)", "#fb923c"),
        ("Step 7: Final Enhanced Plate", "Unsharp Mask Stroke Sharpening", "#4ade80"),
        ("Step 8: Final Forensic Result", "Autonomous OCR & Validation", "#38bdf8"),
    ]

    # Images to plot
    img_list = [
        stages["step1_original"],
        stages["step2_nafnet"],
        stages["step3_raw_crop"],
        stages["step4_rectified"],
        stages["step5_super_resolved"],
        stages["step6_clahe_denoise"],
        stages["step7_enhanced"]
    ]

    # Plot Steps 1 through 7
    for k in range(7):
        r = k // 4
        c = k % 4
        ax = axes[r, c]
        im = img_list[k]
        t_main, t_sub, color = stage_titles[k]

        if im is not None and im.size > 0:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            h, w = im.shape[:2]
            res_str = f"{w}x{h} px"
        else:
            ax.text(0.5, 0.5, "Stage Unavailable", color="orange", ha="center", va="center")
            res_str = "N/A"

        ax.set_title(f"{t_main}\n({t_sub} — {res_str})", fontsize=10.5, fontweight="bold", color=color, pad=7)
        ax.axis("off")

        # Frame border
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.2)

    # Plot Step 8: Final OCR & Quantitative Result Card
    ax_card = axes[1, 3]
    ax_card.set_facecolor("#131c31")
    for spine in ax_card.spines.values():
        spine.set_color("#38bdf8")
        spine.set_linewidth(2.0)

    res = stages["pipeline_res"]
    metrics = stages["metrics"]
    pred_text = res.get("final_text", "NOT_DETECTED")
    raw_ocr = res.get("raw_ocr_text", "NOT_DETECTED")
    winner = res.get("winning_candidate", "Unknown")
    strategy = res.get("selection_strategy", "Unknown")
    conf = res.get("final_conf", 0.0)
    latency_ms = res.get("latency_breakdown", {}).get("total_pipeline_ms", 0.0)

    gt_clean = "".join([ch for ch in (gt_text or "").upper() if ch.isalnum()]).strip()
    is_exact = ForensicEvaluator.compute_exact_match(gt_clean, pred_text) if gt_clean else None
    cer_score = ForensicEvaluator.compute_cer(gt_clean, pred_text) if gt_clean else None

    if is_exact is True:
        status_txt = "EXACT MATCH ✓"
    elif is_exact is False:
        status_txt = f"CER: {cer_score:.2f} (Partial)"
    else:
        status_txt = "AUTONOMOUS READ"

    card_str = (
        f"════════════════════════════════════\n"
        f"   STEP 8: RECOGNITION & METRICS   \n"
        f"════════════════════════════════════\n"
        f" RECOGNIZED OCR : {pred_text}\n"
        f" RAW CROP OCR   : {raw_ocr}\n"
    )
    if gt_clean:
        card_str += f" GROUND TRUTH   : {gt_clean}\n"
    card_str += (
        f" STATUS         : {status_txt}\n"
        f" OCR CONFIDENCE : {conf * 100:.1f}%\n"
        f"────────────────────────────────────\n"
        f" FORENSIC METRICS (Before vs After):\n"
        f" • PSNR Gain    : {metrics['psnr']:.2f} dB\n"
        f" • SSIM Index   : {metrics['ssim']:.4f}\n"
        f" • Sharpness Var: {metrics['sharpness_before']:.1f} -> {metrics['sharpness_after']:.1f}\n"
        f" • Relative Gain: {metrics['sharpness_gain_pct']:+.1f}%\n"
        f"────────────────────────────────────\n"
        f" WINNING LADDER : {winner}\n"
        f" STRATEGY       : {strategy}\n"
        f" TOTAL LATENCY  : {latency_ms:.1f} ms\n"
        f"════════════════════════════════════"
    )

    ax_card.text(
        0.05, 0.50, card_str,
        color="#f8fafc", fontsize=8.8, family="monospace", va="center", linespacing=1.28
    )
    ax_card.set_title("Step 8: Final Forensic Result\n(RapidOCR Consensus Transcription)", fontsize=10.5, fontweight="bold", color="#38bdf8", pad=7)
    ax_card.set_xticks([])
    ax_card.set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    return fig


def generate_all_step_progression_cards(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_files: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Executes the pipeline stage-by-stage on the 15 comparative images,
    producing an 8-panel step-by-step progression card for each sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("STEP-BY-STEP FORENSIC PIPELINE PROGRESSION GENERATOR")
    print("=" * 78)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Output Directory: {output_dir}")

    # Gather targets
    target_files = []
    candidates = sample_files or TARGET_SAMPLE_FILES

    for fpath in candidates:
        if os.path.isfile(fpath):
            target_files.append(fpath)

    # Fallback to discover additional images if some defaults are missing
    if len(target_files) < 15:
        more_ds = sorted(glob.glob("data/dataset_prepared/images/*.jpg"))
        for p in more_ds:
            if p not in target_files:
                target_files.append(p)
            if len(target_files) == 15:
                break

    if len(target_files) < 15:
        more_vid = sorted(glob.glob("results/video_anpr/raw_video_candidates/*.jpg"))
        for p in more_vid:
            if p not in target_files:
                target_files.append(p)
            if len(target_files) == 15:
                break

    target_files = target_files[:15]
    print(f"[*] Processing all {len(target_files)} target images across all 8 pipeline stages...")

    pipeline = MainImagePipeline(device=device)
    gt_map = load_manifest_gt()

    records = []

    print("\n" + "=" * 78)
    print(f"{'#':<3} | {'Sample Name':<28} | {'OCR Result':<14} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Sharpness Gain'}")
    print("-" * 78)

    for i, img_path in enumerate(target_files, 1):
        fname = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"[-] [Error] Failed to read {img_path}. Skipping.")
            continue

        # Extract all intermediate steps
        stages = extract_pipeline_stages(pipeline, img_bgr)
        metrics = stages["metrics"]
        res = stages["pipeline_res"]

        # Lookup ground truth
        gt_text = gt_map.get(fname, gt_map.get(Path(fname).stem, None))

        # Save composite step progression card
        card_filename = f"step_progression_{i:02d}_{Path(fname).stem}.png"
        card_save_path = os.path.join(output_dir, card_filename)

        render_step_progression_card(
            idx=i,
            image_path=img_path,
            stages=stages,
            gt_text=gt_text,
            save_path=card_save_path
        )

        rec = {
            "sample_index": i,
            "filename": fname,
            "image_path": img_path,
            "card_image_path": card_save_path,
            "ocr_text": res.get("final_text", "NOT_DETECTED"),
            "ground_truth": gt_text,
            "winning_candidate": res.get("winning_candidate", "Unknown"),
            "psnr_db": metrics["psnr"],
            "ssim": metrics["ssim"],
            "sharpness_before": metrics["sharpness_before"],
            "sharpness_after": metrics["sharpness_after"],
            "sharpness_gain_pct": metrics["sharpness_gain_pct"],
            "latency_ms": res.get("latency_breakdown", {}).get("total_pipeline_ms", 0.0)
        }
        records.append(rec)

        print(f"[{i:02d}] {fname[:28]:<28} | {rec['ocr_text']:<14} | {metrics['psnr']:6.2f} dB  | {metrics['ssim']:6.4f} | {metrics['sharpness_gain_pct']:+8.1f}%")

    print("=" * 78)

    # Save summary JSON
    summary_json_path = os.path.join(output_dir, "step_progression_summary.json")
    avg_psnr = float(np.mean([r["psnr_db"] for r in records])) if records else 0.0
    avg_ssim = float(np.mean([r["ssim"] for r in records])) if records else 0.0
    avg_sharpness = float(np.mean([r["sharpness_gain_pct"] for r in records])) if records else 0.0

    summary_payload = {
        "num_samples_processed": len(records),
        "mean_psnr_db": avg_psnr,
        "mean_ssim": avg_ssim,
        "mean_sharpness_gain_pct": avg_sharpness,
        "samples": records
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Save Markdown report
    report_md_path = os.path.join(output_dir, "STEP_BY_STEP_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Forensic Step-by-Step Pipeline Progression Report\n\n")
        f.write("A granular 8-stage visualization report detailing intermediate visual transformations for all 15 comparative samples.\n\n")
        f.write("### Pipeline Stages Captured in Each Figure:\n")
        f.write("1. **Original Image**: Raw source capture from vehicle photo or traffic video.\n")
        f.write("2. **NAFNet Deblurred Scene**: Full-scene learned motion deblurring with YOLO plate detection box.\n")
        f.write("3. **YOLO Plate Crop**: Isolated bounding box ROI before enhancement.\n")
        f.write("4. **Homography Rectification**: Perspective deskewing using ranked candidate contours with corner-derived dimensions.\n")
        f.write("5. **Real-ESRGAN x4 (2x Scaled)**: Deep RRDBNet super-resolution in FP32.\n")
        f.write("6. **Adaptive Denoise & Contrast**: CIE LAB space NLM denoising and CLAHE dynamic range expansion.\n")
        f.write("7. **Final Enhanced Plate**: Unsharp mask stroke sharpening.\n")
        f.write("8. **Final Forensic Result**: RapidOCR consensus recognition, ground truth, PSNR, SSIM, and confidence.\n\n")
        f.write("## 🔬 Sample Progression Table\n\n")
        f.write("| # | Filename | Recognized OCR | Ground Truth | PSNR (dB) | SSIM | Sharpness Gain | Step Progression Figure |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in records:
            gt_val = r["ground_truth"] or "N/A"
            card_rel = os.path.basename(r["card_image_path"])
            f.write(f"| {r['sample_index']:02d} | `{r['filename']}` | **{r['ocr_text']}** | `{gt_val}` | {r['psnr_db']:.2f} | {r['ssim']:.4f} | {r['sharpness_gain_pct']:+.1f}% | [`{card_rel}`](./{card_rel}) |\n")

    print(f"\n[✓] Successfully generated {len(records)} step progression cards in -> {output_dir}")
    print(f"[✓] Summary JSON saved to -> {summary_json_path}")
    print(f"[✓] Markdown Report saved to -> {report_md_path}")

    return records


if __name__ == "__main__":
    generate_all_step_progression_cards()

