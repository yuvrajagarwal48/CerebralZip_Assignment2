"""
scripts/video_pipeline.py: Vehicle Tracking, Visual Deduplication & Forensic Video Pipeline
========================================================================================
Integrates video vehicle extraction, deep visual re-ID deduplication (ResNet-18),
and end-to-end connection to the Main Forensic Image Pipeline.

Handles:
1. Vehicle Tracking via YOLOv8 (classes=[2, 5, 7] - car, bus, truck) + ByteTrack.
2. License Plate Localization inside vehicle crops with forensic clarity scoring.
3. Tracklet Merging & Broken Track Deduplication via 512-D L2 ResNet-18 embeddings.
4. "No-Plate" Image Filtering: Rejects false-positive vehicle crops where the forensic
   pipeline confirms no license plate or where OCR yields no valid text.
5. "Duplicate Plate" Filtering: Cross-validates recognized license plates across vehicles
   to guarantee that each physical vehicle/plate is unique, retaining the clearest observation.
6. Publication-grade final summary montage displaying only verified cars with their
   rectified plate crops and detected OCR text.
"""

import os
import gc
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
import urllib.request
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Internal pipeline imports (with fallback for notebook/standalone execution)
try:
    from scripts.pipeline import MainImagePipeline
    from scripts.detection import AutonomousPlateDetector
    from scripts.evaluation_metrics import ForensicEvaluator
except ImportError:
    try:
        from forensic_pipeline import MainImagePipeline, AutonomousPlateDetector, ForensicEvaluator
    except ImportError:
        pass


# =========================================================================
# 1. Visual Feature Extractor (Pretrained Deep Re-ID Backbone - No OCR)
# =========================================================================
class VisualFeatureExtractor:
    """
    Extracts L2-normalized 512-D deep appearance feature embeddings using
    a pretrained ResNet-18 backbone stripped of its classification head.
    Used for visual re-identification and tracklet deduplication.
    """
    def __init__(self, device: torch.device):
        self.device = device
        weights = models.ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights)
        # Strip final classification head to output raw 512-D visual vectors
        self.encoder = nn.Sequential(*list(base_model.children())[:-1]).to(device)
        self.encoder.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def get_embedding(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Extracts an L2-normalized 512-D visual feature embedding with AMP acceleration."""
        if bgr_crop is None or bgr_crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = self.encoder(tensor).squeeze().cpu().numpy()
        else:
            feat = self.encoder(tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


# =========================================================================
# 2. Model Loader & Metric Helpers
# =========================================================================
def get_license_plate_model(weights_path: str = "plate_model.pt") -> Any:
    """
    Loads license plate detector model. Tries:
    1. Local weights_path if it exists.
    2. HuggingFace dedicated detector ('joker5914/yolov8n-license-plate').
    3. Official ultralytics anpr demo model fallback.
    """
    if os.path.exists(weights_path):
        return YOLO(weights_path)

    # Prefer dedicated HuggingFace plate detector if available
    try:
        hf_weights = hf_hub_download(repo_id="joker5914/yolov8n-license-plate", filename="best.pt")
        return YOLO(hf_weights)
    except Exception:
        pass

    # Fallback to ultralytics demo weights
    print("[*] Downloading plate detector weights...")
    url = "https://ultralytics.com/assets/anpr-demo-model.pt"
    try:
        urllib.request.urlretrieve(url, weights_path)
        print("[+] Download complete.")
        return YOLO(weights_path)
    except Exception as e:
        raise RuntimeError(f"[-] Plate detector model load failed: {e}")


def calculate_sharpness(img: np.ndarray) -> float:
    """Calculates variance of Laplacian as a proxy for image sharpness."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_plate_clarity(plate_crop: np.ndarray, conf: float) -> float:
    """
    Computes forensic sharpness & resolution clarity score.
    Favors high-resolution, sharp, high-confidence plates with standard aspect ratios.
    """
    if plate_crop is None or plate_crop.size == 0:
        return 0.0
    h, w = plate_crop.shape[:2]
    area = w * h
    sharpness = calculate_sharpness(plate_crop)
    aspect = float(w) / max(1, h)
    aspect_weight = 1.0 if (2.4 <= aspect <= 5.0) else (0.5 if (1.8 <= aspect < 2.4) else 0.2)
    return float(np.sqrt(area) * (sharpness + 1.0) * aspect_weight * conf)


# =========================================================================
# 3. Deep Feature Deduplication (Merging Tracklets of the Same Vehicle)
# =========================================================================
def merge_duplicate_vehicles(
    candidates: List[Dict[str, Any]],
    cosine_thresh: float = 0.82,
    max_frame_gap: int = 350
) -> List[Dict[str, Any]]:
    """
    Merges broken tracks of the same vehicle using cosine similarity of deep visual
    embeddings and temporal proximity, keeping the single sharpest frame.
    """
    if not candidates:
        return []

    # Sort tracks chronologically
    candidates = sorted(candidates, key=lambda x: x["frame_idx"])
    merged = []
    suppressed = set()

    for i in range(len(candidates)):
        if i in suppressed:
            continue
        primary = candidates[i]

        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            compare = candidates[j]

            # Only compare if they appeared within a realistic temporal window
            if abs(compare["frame_idx"] - primary["frame_idx"]) < max_frame_gap:
                cosine_sim = float(np.dot(primary["embedding"], compare["embedding"]))

                if cosine_sim >= cosine_thresh:
                    suppressed.add(j)
                    # Keep whichever observation gave the higher-clarity plate
                    if compare["best_score"] > primary["best_score"]:
                        primary = compare

        merged.append(primary)

    return merged


# Default Model and Output Paths
DEFAULT_VEHICLE_MODEL = "models/yolov8n.pt" if os.path.exists("models/yolov8n.pt") else "yolov8n.pt"
DEFAULT_VIDEO_OUTPUT_DIR = "results/video_anpr"


# =========================================================================
# 4. Video Vehicle Extraction Phase (No OCR)
# =========================================================================
def extract_unique_vehicles_no_ocr(
    video_path: str,
    output_dir: str = "results/video_anpr/raw_video_candidates",
    vehicle_model_name: str = DEFAULT_VEHICLE_MODEL,
    cosine_sim_threshold: float = 0.82,
    max_frames: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Tracks vehicles in video using ByteTrack, extracts the best license plate observation
    per tracklet, and deduplicates vehicles via ResNet-18 visual embeddings.

    Returns:
        List of dicts containing vehicle crops, frame indices, scores, and save paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[*] Initializing video extraction on device: {device}")

    vehicle_model = YOLO(vehicle_model_name)
    plate_model = get_license_plate_model()
    feature_extractor = VisualFeatureExtractor(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if verbose:
        print(f"[*] Video loaded: {total_video_frames} frames @ {fps:.1f} FPS")

    car_tracks = defaultdict(lambda: {
        "best_score": -1.0,
        "best_car_img": None,
        "best_frame_idx": -1,
        "plate_detected": False,
        "total_hits": 0
    })

    frame_idx = 0
    if verbose:
        print("[*] Tracking vehicles and localizing candidate license plates...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        h_frame, w_frame = frame.shape[:2]

        # Track Vehicles with ByteTrack (2: Car, 5: Bus, 7: Truck)
        results = vehicle_model.track(
            frame,
            classes=[2, 5, 7],
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
            verbose=False
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                car_crop = frame[y1:y2, x1:x2]
                if car_crop.size == 0 or (x2 - x1) < 60 or (y2 - y1) < 60:
                    continue

                car_tracks[track_id]["total_hits"] += 1

                # Locate license plate inside vehicle crop (conf=0.20 captures low-res/skewed plates)
                plate_res = plate_model(car_crop, conf=0.20, device=device, verbose=False)

                if plate_res and plate_res[0].boxes is not None and len(plate_res[0].boxes) > 0:
                    car_tracks[track_id]["plate_detected"] = True

                    for pbox in plate_res[0].boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
                        p_conf = float(pbox.conf[0].cpu().numpy())

                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(car_crop.shape[1], px2), min(car_crop.shape[0], py2)
                        plate_crop = car_crop[py1:py2, px1:px2]

                        if plate_crop.size == 0:
                            continue

                        score = compute_plate_clarity(plate_crop, p_conf)

                        if score > car_tracks[track_id]["best_score"]:
                            car_tracks[track_id]["best_score"] = score
                            car_tracks[track_id]["best_car_img"] = car_crop.copy()
                            car_tracks[track_id]["best_frame_idx"] = frame_idx

        if frame_idx % 80 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cap.release()

    # Filter for verified plate presence during tracking & extract embeddings
    candidate_cars = []
    if verbose:
        print("[*] Extracting deep visual signatures for vehicle deduplication...")

    for track_id, data in car_tracks.items():
        if data["plate_detected"] and data["best_car_img"] is not None and data["best_car_img"].size > 0:
            embedding = feature_extractor.get_embedding(data["best_car_img"])
            candidate_cars.append({
                "track_id": track_id,
                "best_score": data["best_score"],
                "best_car_img": data["best_car_img"],
                "frame_idx": data["best_frame_idx"],
                "embedding": embedding
            })

    if verbose:
        print(f"[*] Raw vehicle tracks with candidate plates: {len(candidate_cars)}")

    # Merge ID switches / duplicates using ResNet feature similarity
    final_cars = merge_duplicate_vehicles(candidate_cars, cosine_thresh=cosine_sim_threshold)

    # Save images and prepare return structure
    saved_candidates = []
    for idx, car in enumerate(final_cars, 1):
        filename = f"car_{idx}_frame_{car['frame_idx']}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, car["best_car_img"])
        car["save_path"] = save_path
        car["candidate_id"] = idx
        saved_candidates.append(car)
        if verbose:
            print(f"[+] Candidate #{idx} (Frame #{car['frame_idx']}) saved -> {save_path}")

    if verbose:
        print(f"[✓] Candidate vehicles extracted from video: {len(saved_candidates)}")

    return saved_candidates


# =========================================================================
# 5. Forensic Plate Recognition, Filtering & Deduplication
# =========================================================================
def filter_and_deduplicate_vehicles(
    candidate_cars: List[Dict[str, Any]],
    pipeline: Any,
    min_plate_chars: int = 3,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Feeds candidate vehicle images into the Main Forensic Image Pipeline and filters:
    1. 'No-Plate' Images: Rejects false-positive vehicle crops where no plate was
       localized or where OCR produced no valid alphanumeric text.
    2. 'Duplicate Plate' Images: Groups vehicles by recognized plate text and retains
       only the single highest-clarity observation per physical license plate.
    Supports multi-GPU pipeline lists for parallel 2xT4 candidate enhancement.

    Returns:
        List of verified, unique vehicle records containing detected OCR text and visual cards.
    """
    if not candidate_cars:
        return []

    pipelines = pipeline if isinstance(pipeline, list) else [pipeline]
    num_workers = len(pipelines)

    if verbose:
        print("\n" + "="*70)
        print(f"[*] EXECUTING FORENSIC IMAGE PIPELINE ON VIDEO VEHICLE CROPS ({num_workers} Worker(s))")
        print("="*70)

    recognized_records = []

    def process_candidate(car_tuple):
        i, car = car_tuple
        pipe = pipelines[(i - 1) % num_workers]
        dev = getattr(pipe, "device", None)
        if dev is not None and dev.type == "cuda":
            torch.cuda.set_device(dev)

        car_img = car["best_car_img"]
        frame_idx = car["frame_idx"]
        track_id = car["track_id"]

        try:
            res = pipe.run(
                input_bgr=car_img,
                reference_clean_bgr=None,
                track_latency=True
            )
        except Exception:
            return None

        raw_text = res.get("final_text", "").strip()
        clean_plate = "".join([c for c in raw_text.upper() if c.isalnum()]).strip()
        has_bbox = res.get("detection_bbox") is not None
        is_failure_sentinel = (
            clean_plate in ("NOTDETECTED", "LOCALIZATIONFAILED", "FAILED", "NONE", "")
            or raw_text.upper().startswith("NOT_DETECTED")
            or raw_text.upper().startswith("LOCALIZATION_FAILED")
        )

        if not has_bbox or len(clean_plate) < min_plate_chars or is_failure_sentinel:
            return None

        return {
            "candidate_id": car.get("candidate_id", i),
            "track_id": track_id,
            "frame_idx": frame_idx,
            "car_img": car_img,
            "pipeline_result": res,
            "ocr_text": clean_plate,
            "raw_ocr_text": raw_text,
            "winning_candidate": res.get("winning_candidate", "Unknown"),
            "selection_strategy": res.get("selection_strategy", "Unknown"),
            "qa_metrics": res.get("qa_metrics", {}),
            "plate_bbox": res.get("detection_bbox"),
            "vis": res.get("vis", {}),
            "clarity_score": car.get("best_score", 0.0),
            "latency_ms": res.get("latency_breakdown", {}).get("total_pipeline_ms", 0.0)
        }

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_candidate, (i, car)): i for i, car in enumerate(candidate_cars, 1)}
            for f in as_completed(futures):
                rec = f.result()
                if rec is not None:
                    recognized_records.append(rec)
                    if verbose:
                        print(f"[✓] [CONFIRMED PLATE] Track #{rec['track_id']} -> OCR: '{rec['ocr_text']}' "
                              f"(Strategy: {rec['selection_strategy']}, Winner: {rec['winning_candidate']})")
    else:
        for i, car in enumerate(candidate_cars, 1):
            if verbose:
                print(f"\n[*] Processing Candidate #{i}/{len(candidate_cars)} (Track #{car['track_id']}, Frame #{car['frame_idx']})...")
            rec = process_candidate((i, car))
            if rec is not None:
                recognized_records.append(rec)
                if verbose:
                    print(f"[✓] [CONFIRMED PLATE] Candidate #{i} -> OCR: '{rec['ocr_text']}' "
                          f"(Strategy: {rec['selection_strategy']}, Winner: {rec['winning_candidate']})")
            elif verbose:
                print(f"[-] [FILTERED: NO PLATE] Candidate #{i} rejected (no verified plate).")

    if verbose:
        print(f"\n[*] Confirmed vehicles with valid plates before plate deduplication: {len(recognized_records)}")

    # -----------------------------------------------------------------
    # Filter 2: Deduplicate Across Recognized Plates
    # -----------------------------------------------------------------
    # Group observations by clean_plate text
    plate_groups = defaultdict(list)
    for rec in recognized_records:
        plate_groups[rec["ocr_text"]].append(rec)

    # Also handle near-duplicate plate numbers (Levenshtein distance <= 1 for plates >= 7 chars)
    canonical_plates = {}
    for plate in list(plate_groups.keys()):
        matched_canonical = None
        for canon in canonical_plates:
            if len(plate) >= 7 and len(canon) >= 7:
                # Compute edit distance
                dist = _compute_levenshtein(plate, canon)
                if dist <= 1:
                    matched_canonical = canon
                    break
        if matched_canonical is None:
            canonical_plates[plate] = plate
        else:
            canonical_plates[plate] = matched_canonical

    merged_groups = defaultdict(list)
    for plate, records in plate_groups.items():
        canon = canonical_plates.get(plate, plate)
        merged_groups[canon].extend(records)

    unique_vehicles = []
    for canon_plate, records in merged_groups.items():
        if len(records) > 1:
            # Sort by plate clarity and OCR confidence, pick the best observation
            records.sort(key=lambda r: (r["clarity_score"], r["qa_metrics"].get("sharpness", 0.0)), reverse=True)
            chosen = records[0]
            suppressed_frames = [str(r["frame_idx"]) for r in records[1:]]
            if verbose:
                print(f"[!] [FILTERED: DUPLICATE PLATE] License plate '{canon_plate}' observed multiple times. "
                      f"Retaining highest-clarity observation (Frame #{chosen['frame_idx']}); "
                      f"suppressing duplicates from Frame(s) #{', #'.join(suppressed_frames)}.")
        else:
            chosen = records[0]

        unique_vehicles.append(chosen)

    # Sort final unique vehicles chronologically by frame index
    unique_vehicles.sort(key=lambda x: x["frame_idx"])

    if verbose:
        print("\n" + "="*70)
        print(f"[✓] FINAL FORENSIC COUNT: {len(unique_vehicles)} UNIQUE CAR(S) WITH VERIFIED LICENSE PLATES")
        print("="*70)
        for idx, v in enumerate(unique_vehicles, 1):
            print(f"  [{idx}] Frame #{v['frame_idx']:4d} | Plate: {v['ocr_text']:<14} | "
                  f"Strategy: {v['selection_strategy']:<18} | Latency: {v['latency_ms']:.1f} ms")

    return unique_vehicles


def _compute_levenshtein(s1: str, s2: str) -> int:
    """Computes basic Levenshtein distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


# =========================================================================
# 6. Final Result Display: Cars with License Plates & Detected OCR
# =========================================================================
def display_final_vehicle_ocr_summary(
    verified_vehicles: List[Dict[str, Any]],
    output_dir: str = "forensic_video_outputs",
    save_fig: bool = True,
    show_fig: bool = True
) -> Optional[plt.Figure]:
    """
    Renders and displays the final forensic summary figure containing ONLY
    cars with verified license plates and their detected OCR text.

    For each verified vehicle:
      Column 1: Cropped Vehicle Image with Bounding Box overlaid.
      Column 2: Rectified & Enhanced License Plate ROI (Winning Ladder Variant).
      Column 3: Forensic OCR Diagnostic Metadata Banner.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not verified_vehicles:
        print("\n[-] [Summary] No cars with verified license plates to display.")
        return None

    num_vehicles = len(verified_vehicles)
    fig, axes = plt.subplots(num_vehicles, 3, figsize=(18, 4.2 * num_vehicles), squeeze=False)
    fig.suptitle("Forensic Video ANPR: Verified Unique Vehicles & Recognized License Plates", fontsize=16, y=0.995)

    for i, v in enumerate(verified_vehicles):
        car_bgr = v["car_img"].copy()
        bbox = v["plate_bbox"]
        frame_idx = v["frame_idx"]
        plate_text = v["ocr_text"]
        winning_variant = v["winning_candidate"]
        strategy = v["selection_strategy"]
        qa = v["qa_metrics"]
        vis = v.get("vis", {})

        # -------------------------------------------------------------
        # Column 1: Car Image with Bounding Box
        # -------------------------------------------------------------
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(car_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"PLATE: {plate_text}"
            cv2.putText(
                car_bgr, label, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
            )

        car_rgb = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(car_rgb)
        axes[i, 0].set_title(f"Vehicle #{i+1} (Captured at Frame #{frame_idx})", fontsize=12, fontweight='bold')
        axes[i, 0].axis("off")

        # -------------------------------------------------------------
        # Column 2: Extracted & Restored Plate Crop
        # -------------------------------------------------------------
        plate_crop = None
        cand_variants = vis.get("candidate_variants", {})
        if winning_variant in cand_variants and getattr(cand_variants[winning_variant], "size", 0) > 0:
            plate_crop = cand_variants[winning_variant]
        elif vis.get("display_crop") is not None and getattr(vis.get("display_crop"), "size", 0) > 0:
            plate_crop = vis["display_crop"]
        elif vis.get("rectified_crop") is not None and getattr(vis.get("rectified_crop"), "size", 0) > 0:
            plate_crop = vis["rectified_crop"]
        elif vis.get("raw_crop") is not None and getattr(vis.get("raw_crop"), "size", 0) > 0:
            plate_crop = vis["raw_crop"]
        elif bbox is not None and v.get("car_img") is not None:
            x1, y1, x2, y2 = bbox
            ch, cw = v["car_img"].shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(cw, int(x2)), min(ch, int(y2))
            if x2 > x1 and y2 > y1:
                plate_crop = v["car_img"][y1:y2, x1:x2]

        if plate_crop is not None and plate_crop.size > 0:
            plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(plate_rgb)
        else:
            axes[i, 1].text(0.5, 0.5, "Plate ROI", ha="center", va="center")

        axes[i, 1].set_title(f"Restored Plate (Winner: {winning_variant})", fontsize=11)
        axes[i, 1].axis("off")

        # -------------------------------------------------------------
        # Column 3: Forensic OCR Diagnostic Metadata Banner
        # -------------------------------------------------------------
        axes[i, 2].axis("off")
        card_text = (
            f"RECOGNIZED LICENSE PLATE:\n"
            f"  >>>  {plate_text}  <<<\n\n"
            f"FORENSIC METADATA:\n"
            f"• Video Frame       : #{frame_idx}\n"
            f"• Tracking ID       : Track #{v['track_id']}\n"
            f"• Selection Mode    : {strategy}\n"
            f"• Winning Variant   : {winning_variant}\n"
            f"• Plate Dimensions  : {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]} px\n"
            f"• Sharpness Score   : {qa.get('sharpness', 0.0):.1f}\n"
            f"• Noise Variance    : {qa.get('noise_var', 0.0):.1f}\n"
            f"• Contrast Index    : {qa.get('contrast', 0.0):.2f}\n"
            f"• Pipeline Latency  : {v['latency_ms']:.1f} ms"
        )
        axes[i, 2].text(
            0.05, 0.5, card_text,
            fontsize=11, family="monospace", va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#EBF5FB", edgecolor="#2980B9", linewidth=2)
        )

    plt.tight_layout()

    if save_fig:
        summary_path = os.path.join(output_dir, "verified_vehicles_anpr_summary.png")
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        print(f"\n[✓] [Visualization] Master forensic ANPR summary saved -> {summary_path}")

    # Also save individual annotated vehicle images
    for i, v in enumerate(verified_vehicles, 1):
        clean_name = f"verified_car_{i}_frame_{v['frame_idx']}_{v['ocr_text']}.jpg"
        save_path = os.path.join(output_dir, clean_name)
        out_bgr = v["car_img"].copy()
        if v["plate_bbox"] is not None:
            x1, y1, x2, y2 = v["plate_bbox"]
            cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out_bgr, v["ocr_text"], (x1, max(y1 - 6, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )
        cv2.imwrite(save_path, out_bgr)
        v["annotated_car_path"] = save_path
        v["annotated_car_img"] = out_bgr

    # Save master JSON metadata
    json_path = os.path.join(output_dir, "verified_vehicles_summary.json")
    summary_data = [
        {
            "rank": i + 1,
            "plate_number": v["ocr_text"],
            "frame_idx": v["frame_idx"],
            "track_id": int(v["track_id"]),
            "winning_variant": v["winning_candidate"],
            "selection_strategy": v["selection_strategy"],
            "qa_metrics": {k: float(val) if isinstance(val, (int, float, np.floating)) else val for k, val in v["qa_metrics"].items()},
            "latency_ms": float(v["latency_ms"])
        }
        for i, v in enumerate(verified_vehicles)
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[✓] [Export] Master JSON summary saved -> {json_path}")

    if show_fig:
        plt.show()
    plt.close(fig)

    return fig


# =========================================================================
# 7. Master Video Forensic Pipeline Facade
# =========================================================================
class ForensicVideoPipeline:
    """
    Unified end-to-end video pipeline:
    1. Extracts candidate vehicles and performs tracklet re-ID deduplication.
    2. Runs the Main Forensic Image Pipeline on all candidates.
    3. Filters out false-positive 'no plate' images.
    4. Filters out duplicate plate numbers.
    5. Displays and saves final clean visual summary of cars with plates and OCR.
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        pipeline: Optional[Any] = None,
        pipelines: Optional[List[Any]] = None
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if pipelines is not None and len(pipelines) > 0:
            self.pipelines = pipelines
            self.pipeline = pipelines
        elif isinstance(pipeline, list):
            self.pipelines = pipeline
            self.pipeline = pipeline
        else:
            self.pipeline = pipeline or MainImagePipeline(device=self.device)
            self.pipelines = [self.pipeline]

    def process_video(
        self,
        video_path: str,
        output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
        vehicle_model_name: str = DEFAULT_VEHICLE_MODEL,
        cosine_sim_threshold: float = 0.82,
        max_frames: Optional[int] = None,
        save_visual_summary: bool = True,
        show_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes complete video forensic pipeline from video file to verified plates.
        """
        raw_candidates_dir = os.path.join(output_dir, "raw_video_candidates")

        # Stage 1: Track vehicles, localize plates, and deduplicate visually
        candidate_cars = extract_unique_vehicles_no_ocr(
            video_path=video_path,
            output_dir=raw_candidates_dir,
            vehicle_model_name=vehicle_model_name,
            cosine_sim_threshold=cosine_sim_threshold,
            max_frames=max_frames,
            device=self.device,
            verbose=True
        )

        if not candidate_cars:
            print("[-] No candidate vehicles found in video.")
            return []

        # Stage 2: Run forensic image pipeline and filter no-plate & duplicate plates
        verified_vehicles = filter_and_deduplicate_vehicles(
            candidate_cars=candidate_cars,
            pipeline=self.pipeline,
            min_plate_chars=3,
            verbose=True
        )

        # Stage 3: Display clean visual summary
        display_final_vehicle_ocr_summary(
            verified_vehicles=verified_vehicles,
            output_dir=output_dir,
            save_fig=save_visual_summary,
            show_fig=show_summary
        )

        return verified_vehicles


def process_video_and_recognize_plates(
    video_path: str = "data/videos/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4",
    output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    vehicle_model_name: str = DEFAULT_VEHICLE_MODEL,
    cosine_sim_threshold: float = 0.82,
    max_frames: Optional[int] = None,
    device: Optional[torch.device] = None,
    show_summary: bool = True
) -> List[Dict[str, Any]]:
    """Functional API for processing video end-to-end with hardcoded defaults."""
    pipe = ForensicVideoPipeline(device=device)
    return pipe.process_video(
        video_path=video_path,
        output_dir=output_dir,
        vehicle_model_name=vehicle_model_name,
        cosine_sim_threshold=cosine_sim_threshold,
        max_frames=max_frames,
        save_visual_summary=True,
        show_summary=show_summary
    )


if __name__ == "__main__":
    # Hardcoded direct execution (no CLI arguments)
    DEFAULT_VIDEO = "data/videos/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
    DEFAULT_OUTPUT = DEFAULT_VIDEO_OUTPUT_DIR

    if not os.path.exists(DEFAULT_VIDEO):
        vids = list(Path("data/videos").glob("*.mp4"))
        if vids:
            DEFAULT_VIDEO = str(vids[0])

    process_video_and_recognize_plates(
        video_path=DEFAULT_VIDEO,
        output_dir=DEFAULT_OUTPUT,
        show_summary=True
    )
