"""
scripts/detection.py: Autonomous License Plate Detector Module
==============================================================
Encapsulates dedicated license plate detection using YOLOv8 trained specifically
for license plate localization (joker5914/yolov8n-license-plate).
"""

import os
from typing import Optional, List, Tuple, Any
import numpy as np
import torch
from huggingface_hub import hf_hub_download

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class AutonomousPlateDetector:
    """
    Detection Module:
    Localizes license plate regions in full-frame scene images using YOLOv8.
    Allows easy substitution with other object detectors (e.g. Faster R-CNN, RT-DETR).
    """
    def __init__(
        self,
        repo_id: str = "joker5914/yolov8n-license-plate",
        filename: str = "best.pt",
        conf_thresh: float = 0.25,
        model: Optional[YOLO] = None,
        device: Optional[Any] = None
    ):
        self.conf_thresh = conf_thresh
        self.device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model is not None:
            self.model = model
            print("[✓] [DetectionModule] Using preloaded YOLO model.")
        else:
            if YOLO is None:
                raise ImportError(
                    "[DetectionModule] 'ultralytics' is not installed in the current environment. "
                    "Install it via 'pip install ultralytics' or inject a preloaded detector model."
                )
            print(f"[*] [DetectionModule] Initializing YOLO Plate Detector on {self.device}...")
            try:
                local_cand = os.path.join("models", "plate_detector.pt")
                local_alt = os.path.join("models", filename)
                if os.path.isfile(local_cand):
                    local_weights = local_cand
                    print(f"[✓] [DetectionModule] Loaded local weights from: {local_weights}")
                elif os.path.isfile(local_alt):
                    local_weights = local_alt
                    print(f"[✓] [DetectionModule] Loaded local weights from: {local_weights}")
                else:
                    local_weights = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models")
                self.model = YOLO(local_weights)
                print("[✓] [DetectionModule] Autonomous YOLO Detector loaded.")
            except Exception as e:
                raise RuntimeError(
                    f"[DetectionModule] Failed to download or load dedicated license plate detector "
                    f"weights from '{repo_id}/{filename}': {e}. Explicit failure triggered: "
                    f"generic YOLO fallback is disabled because generic COCO YOLO does not detect license plates."
                ) from e

    def detect(self, img_bgr: np.ndarray) -> Tuple[Optional[List[int]], float]:
        """
        Executes plate detection on a BGR image.
        Uses FP16 half precision automatically when on CUDA (e.g. Tesla T4).
        Returns:
            Tuple ([x1, y1, x2, y2] clamped to image dimensions, confidence score).
            If no plate detected, returns (None, 0.0).
        """
        dev_str = str(self.device) if self.device is not None else None
        results = self.model(img_bgr, conf=self.conf_thresh, device=dev_str, verbose=False)[0]
        if len(results.boxes) == 0:
            return None, 0.0

        best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        conf = float(best_box.conf[0])

        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return [x1, y1, x2, y2], round(conf, 3)

