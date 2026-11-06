#!/usr/bin/env python3
"""
scripts/generate_comparative_cards.py: 15-Sample Comparative Forensic Visualizer
================================================================================
Takes 15 representative vehicle images from dataset images and video ANPR footage,
processes them through the forensic pipeline, and generates a single composite
diagnostic image for each sample containing:
  - Panel 1: Before Image (Vehicle / Scene with localized plate boundary)
  - Panel 2: Extracted Plate (Raw ROI bounding box crop before enhancement)
  - Panel 3: After Plate (Deep restored, deskewed & super-resolved plate crop)
  - Panel 4: OCR Result & Quantitative Improvements (PSNR, SSIM, Sharpness Gain)

All 15 composite images are saved to results/comparative_cards/.
Also generates comparative_summary.json and COMPARATIVE_REPORT.md.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

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

DEFAULT_OUTPUT_DIR = "results/comparative_cards"
DEFAULT_MANIFEST_PATH = "data/dataset_prepared/manifest.json"

# Curated 15 representative sample files (8 from dataset images, 7 from video ANPR)
DEFAULT_SAMPLE_FILES = [
    # Real-world dataset images (varying lighting, angles, plate geometries)
    "data/dataset_prepared/images/AR13.jpg",
    "data/dataset_prepared/images/AN1.jpg",
    "data/dataset_prepared/images/AN2.jpg",
    "data/dataset_prepared/images/AN4.jpg",
    "data/dataset_prepared/images/AN5.jpg",
    "data/dataset_prepared/images/AN6.jpg",
    "data/dataset_prepared/images/AN7.jpg",
    "data/dataset_prepared/images/AN10.jpg",
    # Video ANPR candidate vehicle frames (motion blur, tracking crops, varying resolutions)
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


def compute_metrics(raw_crop: np.ndarray, enhanced_crop: np.ndarray) -> Dict[str, float]:
    """
    Calculates PSNR, SSIM, and Laplacian sharpness gain between the raw
    extracted plate (Before) and the restored/super-resolved plate (After).
    """
    if raw_crop is None or enhanced_crop is None or raw_crop.size == 0 or enhanced_crop.size == 0:
        return {"psnr": 0.0, "ssim": 0.0, "sharpness_before": 0.0, "sharpness_after": 0.0, "sharpness_gain_pct": 0.0}

    # Match dimensions for fair structural comparison
    th, tw = enhanced_crop.shape[:2]
    raw_aligned = cv2.resize(raw_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    # Compute PSNR & SSIM across color channels
    try:
        psnr_val = float(peak_signal_noise_ratio(raw_aligned, enhanced_crop, data_range=255))
    except Exception:
        psnr_val = 0.0

    try:
        ssim_val = float(structural_similarity(raw_aligned, enhanced_crop, channel_axis=2, data_range=255))
    except Exception:
        ssim_val = 0.0

    # Compute Laplacian variance of sharpness
    gray_before = cv2.cvtColor(raw_aligned, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2GRAY)
    var_before = float(cv2.Laplacian(gray_before, cv2.CV_64F).var())
    var_after = float(cv2.Laplacian(gray_after, cv2.CV_64F).var())
    gain_pct = ((var_after - var_before) / max(1.0, var_before)) * 100.0 if var_before > 0 else 0.0

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "sharpness_before": var_before,
        "sharpness_after": var_after,
        "sharpness_gain_pct": gain_pct
    }


def render_composite_card(
    idx: int,
    image_path: str,
    orig_bgr: np.ndarray,
    raw_crop: np.ndarray,
    enhanced_crop: np.ndarray,
    bbox: Optional[List[int]],
    ocr_result: Dict[str, Any],
    metrics: Dict[str, float],
    gt_text: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders a 4-panel forensic diagnostic comparison in a single image:
      Panel 1: Before Image (Vehicle / Scene with bounding box)
      Panel 2: Extracted Plate (Before Enhancement)
      Panel 3: After Plate (Enhanced & Super-Resolved)
      Panel 4: Forensic Recognition & Quantitative Improvement Card
    """
    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.2),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.35]}
    )
    fig.patch.set_facecolor("#0f172a")

    source_tag = "Dataset Scene" if "dataset" in image_path.lower() else "Video Footage"
    fname = os.path.basename(image_path)
    fig.suptitle(
        f"Forensic Sample #{idx:02d}: {fname} ({source_tag})",
        fontsize=14, fontweight="bold", color="#f8fafc", y=0.98
    )

    # -------------------------------------------------------------
    # Panel 1: Before Image (Full Vehicle / Scene with Bounding Box)
    # -------------------------------------------------------------
    scene_disp = orig_bgr.copy()
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(scene_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            scene_disp, "PLATE ROI", (x1, max(y1 - 6, 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
        )
    axes[0].imshow(cv2.cvtColor(scene_disp, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"1. Before Image (Source Frame)\nResolution: {orig_bgr.shape[1]}x{orig_bgr.shape[0]}",
        fontsize=11, fontweight="bold", color="#93c5fd", pad=8
    )
    axes[0].axis("off")

    # -------------------------------------------------------------
    # Panel 2: Extracted Plate (Raw ROI Bounding Box Crop)
    # -------------------------------------------------------------
    if raw_crop is not None and raw_crop.size > 0:
        axes[1].imshow(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"2. Extracted Plate (Before)\nRaw Crop: {raw_crop.shape[1]}x{raw_crop.shape[0]} px",
            fontsize=11, fontweight="bold", color="#cbd5e1", pad=8
        )
    else:
        axes[1].text(0.5, 0.5, "Crop Unavailable", color="orange", ha="center", va="center")
    axes[1].axis("off")

    # -------------------------------------------------------------
    # Panel 3: After Plate (Enhanced & Super-Resolved)
    # -------------------------------------------------------------
    if enhanced_crop is not None and enhanced_crop.size > 0:
        axes[2].imshow(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB))
        axes[2].set_title(
            f"3. After Plate (Enhanced)\nRestored & SR: {enhanced_crop.shape[1]}x{enhanced_crop.shape[0]} px",
            fontsize=11, fontweight="bold", color="#34d399", pad=8
        )
    else:
        axes[2].text(0.5, 0.5, "Enhancement Unavailable", color="orange", ha="center", va="center")
    axes[2].axis("off")

    # -------------------------------------------------------------
    # Panel 4: Recognition & Quantitative Improvement Card
    # -------------------------------------------------------------
    axes[3].set_facecolor("#1e293b")
    for spine in axes[3].spines.values():
        spine.set_color("#3b82f6")
        spine.set_linewidth(1.5)

    pred_text = ocr_result.get("final_text", "NOT_DETECTED")
    raw_ocr = ocr_result.get("raw_ocr_text", "NOT_DETECTED")
    winning_cand = ocr_result.get("winning_candidate", "Unknown")
    strategy = ocr_result.get("selection_strategy", "Unknown")
    conf = ocr_result.get("final_conf", 0.0)
    latency_ms = ocr_result.get("latency_breakdown", {}).get("total_pipeline_ms", 0.0)

    gt_clean = "".join([c for c in (gt_text or "").upper() if c.isalnum()]).strip()
    is_exact = ForensicEvaluator.compute_exact_match(gt_clean, pred_text) if gt_clean else None
    cer_score = ForensicEvaluator.compute_cer(gt_clean, pred_text) if gt_clean else None

    # Status color
    if is_exact is True:
        status_txt = "EXACT MATCH ✓"
        status_color = "#4ade80"
    elif is_exact is False:
        status_txt = f"PARTIAL MATCH (CER: {cer_score:.2f})"
        status_color = "#facc15"
    else:
        status_txt = "AUTONOMOUS READ"
        status_color = "#60a5fa"

    card_text = (
        f"4. FINAL OCR & METRICS\n"
        f"─────────────────────────────────────\n"
        f" RECOGNIZED OCR : {pred_text}\n"
        f" RAW CROP OCR   : {raw_ocr}\n"
    )
    if gt_clean:
        card_text += f" GROUND TRUTH   : {gt_clean}\n"
    card_text += (
        f" STATUS         : {status_txt}\n"
        f" OCR CONFIDENCE : {conf * 100:.1f}%\n"
        f"─────────────────────────────────────\n"
        f" QUANTITATIVE FORENSIC GAIN:\n"
        f" • PSNR (Before vs After) : {metrics['psnr']:.2f} dB\n"
        f" • SSIM Structural Index : {metrics['ssim']:.4f}\n"
        f" • Sharpness Var (Lapl.)  : {metrics['sharpness_before']:.1f} -> {metrics['sharpness_after']:.1f}\n"
        f" • Sharpness Relative Gain: {metrics['sharpness_gain_pct']:+.1f}%\n"
        f"─────────────────────────────────────\n"
        f" STRATEGY : {strategy}\n"
        f" WINNER   : {winning_cand}\n"
        f" LATENCY  : {latency_ms:.1f} ms"
    )

    axes[3].text(
        0.05, 0.50, card_text,
        color="#f1f5f9", fontsize=9.2, family="monospace", va="center", linespacing=1.35
    )
    axes[3].set_title("4. Quantitative Evaluation", fontsize=11, fontweight="bold", color="#38bdf8", pad=8)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    return fig


def generate_all_15_comparative_cards(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    sample_files: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Processes 15 images through the forensic pipeline, calculates PSNR and SSIM,
    and produces 15 individual composite comparison cards in results/comparative_cards/.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("FORENSIC 15-SAMPLE COMPARATIVE DEMONSTRATION GENERATOR")
    print("=" * 75)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Output Directory: {output_dir}")

    # Gather 15 samples
    target_files = []
    candidates = sample_files or DEFAULT_SAMPLE_FILES

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
    print(f"[*] Total target images to process: {len(target_files)}")

    # Initialize Pipeline
    pipeline = MainImagePipeline(device=device)
    gt_map = load_manifest_gt()

    records = []

    print("\n" + "=" * 75)
    print(f"{'#':<3} | {'Sample Name':<28} | {'OCR Result':<14} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Sharpness Gain'}")
    print("-" * 75)

    for i, img_path in enumerate(target_files, 1):
        fname = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"[-] [Error] Failed to read {img_path}. Skipping.")
            continue

        res = pipeline.run(input_bgr=img_bgr, track_latency=True)
        vis = res.get("vis")

        raw_crop = vis.get("raw_crop") if vis else None
        enh_crop = None
        if vis:
            if vis.get("display_crop") is not None and getattr(vis.get("display_crop"), "size", 0) > 0:
                enh_crop = vis.get("display_crop")
            elif vis.get("rectified_crop") is not None and getattr(vis.get("rectified_crop"), "size", 0) > 0:
                enh_crop = vis.get("rectified_crop")

        # Fallback if detection failed
        if raw_crop is None:
            raw_crop = img_bgr
        if enh_crop is None:
            enh_crop = raw_crop

        bbox = res.get("detection_bbox")

        # Calculate PSNR and SSIM between before (raw crop) and after (enhanced crop)
        metrics = compute_metrics(raw_crop, enh_crop)

        # Lookup ground truth
        gt_text = gt_map.get(fname, gt_map.get(Path(fname).stem, None))

        # Save single composite image for this sample
        card_filename = f"comparative_card_{i:02d}_{Path(fname).stem}.png"
        card_save_path = os.path.join(output_dir, card_filename)

        render_composite_card(
            idx=i,
            image_path=img_path,
            orig_bgr=img_bgr,
            raw_crop=raw_crop,
            enhanced_crop=enh_crop,
            bbox=bbox,
            ocr_result=res,
            metrics=metrics,
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

    print("=" * 75)

    # Save summary JSON
    summary_json_path = os.path.join(output_dir, "comparative_summary.json")
    avg_psnr = float(np.mean([r["psnr_db"] for r in records])) if records else 0.0
    avg_ssim = float(np.mean([r["ssim"] for r in records])) if records else 0.0
    avg_sharpness_gain = float(np.mean([r["sharpness_gain_pct"] for r in records])) if records else 0.0

    summary_payload = {
        "num_samples_processed": len(records),
        "mean_psnr_db": avg_psnr,
        "mean_ssim": avg_ssim,
        "mean_sharpness_gain_pct": avg_sharpness_gain,
        "samples": records
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # Save Markdown report
    report_md_path = os.path.join(output_dir, "COMPARATIVE_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Forensic 15-Sample Comparative Evaluation Report\n\n")
        f.write("A quantitative and visual before/after analysis across 15 real-world vehicle images and video ANPR frames.\n\n")
        f.write("## 📊 Summary Metrics\n\n")
        f.write(f"- **Total Samples Processed**: {len(records)}\n")
        f.write(f"- **Mean PSNR (dB)**: {avg_psnr:.2f} dB\n")
        f.write(f"- **Mean SSIM**: {avg_ssim:.4f}\n")
        f.write(f"- **Mean Laplacian Sharpness Gain**: {avg_sharpness_gain:+.1f}%\n\n")
        f.write("## 🔬 Sample-by-Sample Breakdown\n\n")
        f.write("| # | Sample Filename | Recognized OCR | Ground Truth | PSNR (dB) | SSIM | Sharpness Gain | Composite Card |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in records:
            gt_val = r["ground_truth"] or "N/A"
            card_rel = os.path.basename(r["card_image_path"])
            f.write(f"| {r['sample_index']:02d} | `{r['filename']}` | **{r['ocr_text']}** | `{gt_val}` | {r['psnr_db']:.2f} | {r['ssim']:.4f} | {r['sharpness_gain_pct']:+.1f}% | [`{card_rel}`](./{card_rel}) |\n")

    print(f"\n[✓] Successfully generated {len(records)} comparative cards in -> {output_dir}")
    print(f"[✓] Summary JSON saved to -> {summary_json_path}")
    print(f"[✓] Markdown Report saved to -> {report_md_path}")

    return records


if __name__ == "__main__":
    generate_all_15_comparative_cards()
