"""
scripts/visualization.py: Dedicated Forensic Image Visualization Engine
========================================================================
Provides modular, publication-quality 4-panel forensic diagnostic cards:
  [1. Original Image] -> [2. Extracted Plate] -> [3. Post-Processed Plate] -> [4. Final Result]

Can be invoked as a library function or run standalone via CLI.
"""

import os
import json
from typing import Optional, Dict, Any, List
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.pipeline import MainImagePipeline
from scripts.evaluation_metrics import ForensicEvaluator


def render_visual_card(
    pipeline_result: Dict[str, Any],
    gt_text: str = "",
    sample_id: Any = 0,
    tag: str = "real_image",
    save_path: Optional[str] = None,
    show_fig: bool = False
) -> plt.Figure:
    """
    Renders a 4-panel diagnostic figure from pipeline output:
      Panel 1: Original Image with plate detection bounding box.
      Panel 2: Extracted Plate (Raw ROI crop).
      Panel 3: Post-Processed Plate (Rectified, Restored & SR).
      Panel 4: Final Result & Forensic Summary Card.
    """
    res = pipeline_result
    vis = res.get("vis")
    if vis is None:
        raise ValueError("[Visualization] Pipeline result does not contain visualization payload 'vis'.")

    gt_clean = "".join([c for c in gt_text.upper() if c.isalnum()]).strip()
    is_exact = ForensicEvaluator.compute_exact_match(gt_clean, res["final_text"]) if gt_clean else False
    cer_score = ForensicEvaluator.compute_cer(gt_clean, res["final_text"]) if gt_clean else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    try:
        fig.canvas.manager.set_window_title(f"Forensic Case #{sample_id} - {tag}")
    except Exception:
        pass

    # Panel 1: Original Image with Plate Detection
    orig_img = vis.get("original_image", vis.get("scene_restored")).copy()
    x1, y1, x2, y2 = vis["bbox"]
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        orig_img, "PLATE ROI", (x1, max(y1 - 6, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
    )
    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"1. Original Image\n({res['loc_status']})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Extracted Plate (Raw ROI Crop)
    axes[1].imshow(cv2.cvtColor(vis["raw_crop"], cv2.COLOR_BGR2RGB))
    raw_display = res['raw_ocr_text'] if res['raw_ocr_text'] else "NONE"
    axes[1].set_title(f"2. Extracted Plate\nRaw Crop OCR: '{raw_display}'", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Post-Processed Plate (Rectified, Restored & SR)
    post_processed_plate = vis.get("display_crop", vis.get("rectified_crop"))
    axes[2].imshow(cv2.cvtColor(post_processed_plate, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. Post-Processed Plate\n({res['action_summary']})", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Final Result & Forensic Summary Card
    axes[3].set_facecolor("#161616")
    color = "#00ff88" if is_exact else ("#ff4444" if res["final_text"] == "NOT_DETECTED" else "#ffb300")

    qa = res["qa_metrics"]
    bm = res.get("best_metric", {})
    amb_str = f"\nAmbiguities  : {', '.join(res['ambiguities'])}" if res.get("ambiguities") else ""
    report = (
        f"4. FINAL FORENSIC RESULT\n"
        f"──────────────────────────\n"
        f"Ground Truth : {gt_clean if gt_clean else 'N/A'}\n"
        f"Predicted OCR: {res['final_text']}\n"
        f"Exact Match  : {is_exact} (CER: {cer_score:.3f})\n"
        f"Mean OCR Conf: {res['final_conf'] * 100:.1f}%\n"
        f"Status       : {res['certainty_status']}{amb_str}\n"
        f"──────────────────────────\n"
        f"Plate Height : {qa.get('actual_plate_h', qa.get('orig_h', 0)):.1f} px | CharH: {qa.get('est_char_h', 0):.1f} px\n"
        f"Contrast     : {qa.get('contrast_spread', 0.0):.2f} | Noise: {qa.get('noise_status', 'N/A')}\n"
        f"PSNR Gain    : {bm.get('psnr_before', 0.0):.1f} -> {bm.get('psnr_after', 0.0):.1f} dB\n"
        f"SSIM Gain    : {bm.get('ssim_before', 0.0):.3f} -> {bm.get('ssim_after', 0.0):.3f}\n"
        f"Outcome      : {'EXACT MATCH' if is_exact else ('NOT DETECTED' if res['final_text'] == 'NOT_DETECTED' else 'PARTIAL RECOVERY')}"
    )
    axes[3].text(0.05, 0.5, report, color=color, fontsize=8.4, family="monospace", va="center")
    axes[3].set_title("4. Final Result", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[✓] [Visualization] Diagnostic card saved: {save_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return fig


def visualize_single_image(
    image_path: str,
    gt_text: str = "",
    reference_clean_path: Optional[str] = None,
    save_path: Optional[str] = None,
    pipeline: Optional[MainImagePipeline] = None,
    device: Optional[torch.device] = None,
    apply_nafnet: bool = True
) -> Dict[str, Any]:
    """
    Runs the pipeline on a standalone image and generates the 4-panel diagnostic card.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[Visualization] Could not read image at: {image_path}")

    clean_img = cv2.imread(reference_clean_path) if reference_clean_path else None

    if pipeline is None:
        pipeline = MainImagePipeline(device=device)

    res = pipeline.run(input_bgr=img, reference_clean_bgr=clean_img, apply_nafnet=apply_nafnet)
    if res.get("vis") is not None:
        render_visual_card(
            pipeline_result=res,
            gt_text=gt_text,
            sample_id=os.path.splitext(os.path.basename(image_path))[0],
            tag="single_image",
            save_path=save_path
        )
    return res


def run_visual_pipeline(
    manifest_path: str = "./data/dataset_prepared/manifest.json",
    num_samples: int = 6,
    variant_index: int = 0,
    save_dir: Optional[str] = "./forensic_visual_outputs",
    pipeline: Optional[MainImagePipeline] = None
) -> List[Dict[str, Any]]:
    """
    Loads dataset manifest, runs the modular forensic pipeline on N samples,
    computes exact match and CER, generates diagnostic 4-panel figures, and logs metrics.
    """
    if not os.path.exists(manifest_path):
        for fallback in [
            "./data/dataset_prepared/manifest.json",
            "./data/forensic_dataset_prepared/forensic_manifest.json"
        ]:
            if os.path.exists(fallback):
                print(f"[*] Manifest '{manifest_path}' not found, falling back to '{fallback}'")
                manifest_path = fallback
                break

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"[FATAL] Manifest path not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Batch Visual Pipeline on Device: {device}")

    if pipeline is None:
        pipeline = MainImagePipeline(device=device)

    cases_to_run = catalog[:num_samples]
    print(f"[*] Processing {len(cases_to_run)} samples with adaptive enhancement...\n")

    results = []

    for idx, case in enumerate(cases_to_run):
        sample_id = case["sample_id"]
        gt_text = "".join([c for c in case["ground_truth_text"].upper() if c.isalnum()]).strip()
        clean_path = case.get("clean_file", case.get("image_path"))
        clean_img = cv2.imread(clean_path) if clean_path else None

        variants = case.get("degraded_variants", [])
        if not variants:
            deg_type = case.get("category", "real_image")
            deg_img = clean_img
        else:
            selected_variant = variants[min(variant_index, len(variants) - 1)]
            deg_type = selected_variant.get("degradation_type", "Unknown")
            deg_img = cv2.imread(selected_variant["file_path"])

        if deg_img is None:
            continue

        print(f"[{idx + 1}/{len(cases_to_run)}] Sample #{sample_id} | Type: {deg_type} | GT: {gt_text}")

        res = pipeline.run(
            input_bgr=deg_img,
            reference_clean_bgr=clean_img,
            apply_nafnet=True
        )

        vis = res.get("vis", None)
        if vis is None:
            print(f"    [!] Localization failed on Sample #{sample_id}.")
            continue

        save_path = os.path.join(save_dir, f"visual_card_{sample_id}_{deg_type}.png") if save_dir else None
        render_visual_card(
            pipeline_result=res,
            gt_text=gt_text,
            sample_id=sample_id,
            tag=deg_type,
            save_path=save_path,
            show_fig=False
        )
        results.append(res)

    print(f"\n[✓] Completed visualization batch: {len(results)}/{len(cases_to_run)} cards generated.")
    return results


# =========================================================================
# HARDCODED CONFIGURATION & PATHS (NO CLI ARGUMENTS)
# =========================================================================
VIS_IMAGE_PATH = None           # Path to single image (e.g. "./data/dataset_prepared/images/AN1.jpg"), or None for batch
VIS_GT_TEXT = ""
VIS_MANIFEST_PATH = "./data/dataset_prepared/manifest.json"
VIS_NUM_SAMPLES = 5
VIS_OUTPUT_DIR = "./results/visualizations"


def main():
    if VIS_IMAGE_PATH:
        print(f"[*] Visualizing single image: {VIS_IMAGE_PATH}")
        save_name = f"visual_{os.path.splitext(os.path.basename(VIS_IMAGE_PATH))[0]}.png"
        save_dest = os.path.join(VIS_OUTPUT_DIR, save_name)
        visualize_single_image(VIS_IMAGE_PATH, gt_text=VIS_GT_TEXT, save_path=save_dest)
    else:
        run_visual_pipeline(
            manifest_path=VIS_MANIFEST_PATH,
            num_samples=VIS_NUM_SAMPLES,
            save_dir=VIS_OUTPUT_DIR
        )


if __name__ == "__main__":
    main()

