#!/usr/bin/env python3
"""
app.py: Streamlit Web Application for Forensic License Plate Recognition
========================================================================
Interactive web dashboard that accepts photos or traffic videos and displays:
  1. Extracted Plate (Raw ROI Bounding Box Crop)
  2. Enhanced Plate (Deep Restored & Super-Resolved Plate)
  3. Final OCR Value (Autonomous OCR Transcription)
  4. Step-by-Step Intermediate Visualizations across every processing stage:
     - Step 1: Input Scene (Original)
     - Step 2: NAFNet Motion Deblur (+ Localized YOLO Box)
     - Step 3: Raw YOLO Plate Crop (Localized ROI)
     - Step 4: Homography Rectification (Corner Deskewed)
     - Step 5: Real-ESRGAN Super-Resolution (Deep Neural Upscale)
     - Step 6: Adaptive Denoise & CLAHE (CIE LAB Space)
     - Step 7: Final Restored Plate (Stroke Sharpened)
     - Step 8: Final Forensic Result & Confidence Metrics

Includes dynamic module recovery and JPEG network optimization to eliminate
dropped chunks and missing images over remote tunnels (Localtunnel / Cloudflare).
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components

# Import pipeline components
from scripts.pipeline import MainImagePipeline
from scripts.video_pipeline import ForensicVideoPipeline, DEFAULT_VEHICLE_MODEL


# =========================================================================
# Streamlit App Configuration
# =========================================================================
st.set_page_config(
    page_title="Forensic License Plate OCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# Dynamic Module Chunk Shield & Remote Tunnel Error Recovery
# -------------------------------------------------------------------------
# When running on remote environments (Kaggle / Colab) through reverse proxies
# like Localtunnel (loca.lt), concurrent requests for dynamically imported
# Vite chunks (Selectbox, FileUploader, Radio) can be dropped or rate-limited.
# This client-side shield catches unhandled dynamic import rejections on
# window.parent and automatically reloads seamlessly, eliminating manual refreshes.
components.html(
    """
    <script>
    (function() {
        try {
            const p = window.parent;
            if (!p || p === window) return;

            p.addEventListener('unhandledrejection', function(event) {
                const reason = event ? (event.reason ? (event.reason.message || String(event.reason)) : '') : '';
                if (
                    reason.includes('Failed to fetch dynamically imported module') ||
                    reason.includes('Importing a module script failed') ||
                    reason.includes('error loading dynamically imported module')
                ) {
                    console.warn('[ForensicShield] Caught dynamic module failure, auto-recovering:', reason);
                    const lastReload = parseInt(p.sessionStorage.getItem('fs_last_reload') || '0', 10);
                    const now = Date.now();
                    if (now - lastReload > 1800) {
                        p.sessionStorage.setItem('fs_last_reload', String(now));
                        setTimeout(() => p.location.reload(), 300);
                    }
                }
            });
        } catch(e) {
            console.warn('[ForensicShield] Could not bind parent listener:', e);
        }
    })();
    </script>
    """,
    height=0,
    width=0
)

# Initialize Session State
if "photo_img_bgr" not in st.session_state:
    st.session_state["photo_img_bgr"] = None
if "photo_img_name" not in st.session_state:
    st.session_state["photo_img_name"] = None
if "photo_results" not in st.session_state:
    st.session_state["photo_results"] = None
if "photo_stages" not in st.session_state:
    st.session_state["photo_stages"] = None
if "video_source_path" not in st.session_state:
    st.session_state["video_source_path"] = None
if "video_results" not in st.session_state:
    st.session_state["video_results"] = None


# =========================================================================
# Resource Caching
# =========================================================================
@st.cache_resource
def load_pipeline():
    """Initializes and caches the forensic image pipeline."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return MainImagePipeline(device=device)


@st.cache_resource
def load_video_pipeline():
    """Initializes and caches the forensic video pipeline."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline()
    return ForensicVideoPipeline(device=device, pipeline=pipe)


# =========================================================================
# Image Display & Quality Helpers
# =========================================================================
def safe_display_image(img: Any, caption: Optional[str] = None, use_container_width: bool = True):
    """
    Safely sanitizes, converts, and renders image buffers in Streamlit:
      1. Resolves paths or numpy arrays.
      2. Validates dimensions (rejects 0-sized arrays).
      3. Clamps float arrays to uint8 [0, 255].
      4. Converts Grayscale, BGRA, or BGR to standard RGB.
      5. Enforces C-contiguous layout.
      6. Downscales oversized scenes (max 960px) and outputs as JPEG to prevent
         tunnel packet congestion and dropped image tiles.
    """
    if img is None:
        st.warning("Image unavailable")
        return

    # Path fallback
    if isinstance(img, (str, Path)):
        if os.path.isfile(str(img)):
            loaded = cv2.imread(str(img))
            if loaded is None:
                st.warning("Could not read image from path")
                return
            img = loaded
        else:
            st.warning("Image path not found on disk")
            return

    if not isinstance(img, np.ndarray) or getattr(img, "size", 0) == 0:
        st.warning("Empty or invalid image data")
        return

    # Check dimensions
    if len(img.shape) < 2 or img.shape[0] < 2 or img.shape[1] < 2:
        st.warning("Image dimensions too small")
        return

    # Normalize dtype to uint8
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.01:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert color space
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb = img

    # Ensure C-contiguous
    rgb = np.ascontiguousarray(rgb)

    # Downscale overly huge preview images to conserve tunnel throughput
    h, w = rgb.shape[:2]
    if max(h, w) > 960:
        scale = 960.0 / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    st.image(rgb, caption=caption, use_container_width=use_container_width, output_format="JPEG")


def compute_plate_sharpness_metrics(raw_crop: np.ndarray, enhanced_crop: np.ndarray) -> Dict[str, float]:
    """Computes Laplacian sharpness and variance gain between raw and enhanced plates."""
    if raw_crop is None or enhanced_crop is None or getattr(raw_crop, "size", 0) == 0 or getattr(enhanced_crop, "size", 0) == 0:
        return {"sharpness_before": 0.0, "sharpness_after": 0.0, "sharpness_gain_pct": 0.0}

    try:
        th, tw = enhanced_crop.shape[:2]
        raw_aligned = cv2.resize(raw_crop, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        g_before = cv2.cvtColor(raw_aligned, cv2.COLOR_BGR2GRAY) if len(raw_aligned.shape) == 3 else raw_aligned
        g_after = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2GRAY) if len(enhanced_crop.shape) == 3 else enhanced_crop
        var_before = float(cv2.Laplacian(g_before, cv2.CV_64F).var())
        var_after = float(cv2.Laplacian(g_after, cv2.CV_64F).var())
        gain_pct = ((var_after - var_before) / max(1.0, var_before)) * 100.0 if var_before > 0 else 0.0
        return {
            "sharpness_before": var_before,
            "sharpness_after": var_after,
            "sharpness_gain_pct": gain_pct
        }
    except Exception:
        return {"sharpness_before": 0.0, "sharpness_after": 0.0, "sharpness_gain_pct": 0.0}


def build_step_progression(pipeline: MainImagePipeline, res: Dict[str, Any], input_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Extracts intermediate visual outputs for each of the 8 forensic pipeline steps:
      - Step 1: Input Scene (Original)
      - Step 2: NAFNet Motion Deblur (+ Localized YOLO Box)
      - Step 3: Raw YOLO Plate Crop (Localized ROI)
      - Step 4: Homography Rectification (Corner Deskewed)
      - Step 5: Real-ESRGAN Super-Resolution (Deep Neural Upscale)
      - Step 6: Adaptive Denoise & CLAHE (CIE LAB Space)
      - Step 7: Final Restored Plate (Stroke Sharpened)
      - Step 8: Final Forensic Result & Transcription Metrics
    """
    vis = res.get("vis", {}) if res else {}

    # Step 1: Input Scene (Original)
    step1_orig = input_bgr.copy()
    h1, w1 = step1_orig.shape[:2]
    step1_caption = f"Raw Source Scene ({w1}x{h1} px)"

    # Step 2: NAFNet Deblur + YOLO Plate Box
    scene_restored = vis.get("scene_restored")
    if scene_restored is None or getattr(scene_restored, "size", 0) == 0:
        scene_restored = input_bgr.copy()
    step2_nafnet = scene_restored.copy()
    bbox = res.get("detection_bbox")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(step2_nafnet.shape[1], int(x2)), min(step2_nafnet.shape[0], int(y2))
        cv2.rectangle(step2_nafnet, (x1, y1), (x2, y2), (0, 255, 0), 2)
        conf = res.get("final_conf", 0.0)
        cv2.putText(
            step2_nafnet, f"PLATE {conf:.2f}", (x1, max(y1 - 6, 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
        )
        step2_caption = f"NAFNet Deblurred Scene (Localized Box: {x2-x1}x{y2-y1} px)"
    else:
        step2_caption = "NAFNet Deblurred Scene (Full Frame)"

    # Step 3: Raw Plate Crop (Localized ROI)
    raw_crop = vis.get("raw_crop")
    if raw_crop is None or getattr(raw_crop, "size", 0) == 0:
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(input_bgr.shape[1], int(x2)), min(input_bgr.shape[0], int(y2))
            if x2 > x1 and y2 > y1:
                raw_crop = input_bgr[y1:y2, x1:x2].copy()
    if raw_crop is None or getattr(raw_crop, "size", 0) == 0:
        raw_crop = input_bgr.copy()
    h3, w3 = raw_crop.shape[:2]
    step3_caption = f"Raw Bounding Box ROI ({w3}x{h3} px)"

    # Step 4: Homography Rectification
    rect_crop = vis.get("rectified_crop")
    if rect_crop is None or getattr(rect_crop, "size", 0) == 0:
        rect_crop = raw_crop.copy()
    h4, w4 = rect_crop.shape[:2]
    ar4 = float(w4) / max(1e-4, float(h4))
    step4_caption = f"Corner Homography ({w4}x{h4} px, AR={ar4:.2f})"

    # Step 5: Real-ESRGAN Super-Resolution
    variants = vis.get("candidate_variants", {})
    sr_crop = variants.get("SR_2X_Downsampled")
    if sr_crop is None:
        sr_crop = variants.get("SR_4X_Native")
    if sr_crop is None or getattr(sr_crop, "size", 0) == 0:
        try:
            sr_crop, _ = pipeline.sr_engine.upscale(rect_crop, target_scale=2.0)
        except Exception:
            sr_crop = cv2.resize(rect_crop, (w4 * 2, h4 * 2), interpolation=cv2.INTER_LANCZOS4)
    h5, w5 = sr_crop.shape[:2]
    step5_caption = f"Real-ESRGAN Upscaled ({w5}x{h5} px)"

    # Step 6: Adaptive Denoise & CLAHE (CIE LAB)
    qa = res.get("qa_metrics", {})
    try:
        lab = cv2.cvtColor(sr_crop, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_nlm = pipeline.router.restoration.nlm.apply(l_ch) if qa.get("needs_denoise", False) else l_ch
        l_clahe = pipeline.router.restoration.clahe.apply(l_nlm) if qa.get("needs_clahe", False) else l_nlm
        l_norm = pipeline.router.restoration.normalize_percentile(l_clahe, 1.0, 99.0)
        clahe_bgr = cv2.cvtColor(l_norm, cv2.COLOR_GRAY2BGR)
    except Exception:
        clahe_bgr = sr_crop.copy()
    step6_caption = f"LAB CLAHE (Noise: {qa.get('noise_var', 0):.1f})"

    # Step 7: Final Restored Plate (Stroke Sharpened)
    enhanced_crop = vis.get("display_crop")
    if enhanced_crop is None or getattr(enhanced_crop, "size", 0) == 0:
        try:
            l_sharp = pipeline.router.restoration.sharpen.apply(l_norm)
            enhanced_crop = cv2.cvtColor(l_sharp, cv2.COLOR_GRAY2BGR)
        except Exception:
            enhanced_crop = clahe_bgr.copy()
    h7, w7 = enhanced_crop.shape[:2]
    step7_caption = f"Stroke Sharpened ({w7}x{h7} px)"

    # Step 8: Metrics & Result
    metrics = compute_plate_sharpness_metrics(raw_crop, enhanced_crop)
    ocr_text = res.get("final_text", "UNKNOWN")
    ocr_conf = res.get("final_conf", 0.0)
    winner = res.get("winning_candidate", "Adaptive_Targeted")
    strategy = res.get("selection_strategy", "Standard")

    return {
        "step1_original": step1_orig,
        "step1_caption": step1_caption,
        "step2_nafnet": step2_nafnet,
        "step2_caption": step2_caption,
        "step3_raw_crop": raw_crop,
        "step3_caption": step3_caption,
        "step4_rectified": rect_crop,
        "step4_caption": step4_caption,
        "step5_super_resolved": sr_crop,
        "step5_caption": step5_caption,
        "step6_clahe_denoise": clahe_bgr,
        "step6_caption": step6_caption,
        "step7_enhanced": enhanced_crop,
        "step7_caption": step7_caption,
        "ocr_text": ocr_text,
        "ocr_conf": ocr_conf,
        "winner": winner,
        "strategy": strategy,
        "metrics": metrics
    }


def render_step_progression_ui(stages: Dict[str, Any], title_prefix: str = ""):
    """Renders an 8-panel grid showing the image after each individual processing step."""
    header = f"{title_prefix} 🔬 End-to-End Pipeline Progression (Step-by-Step Visualizations)" if title_prefix else "🔬 End-to-End Pipeline Progression (Step-by-Step Visualizations)"
    st.markdown(f"### {header}")
    st.caption("Visual progression of the image after each individual stage of the forensic restoration and recognition pipeline.")

    # Row 1: Stages 1 to 4
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    with r1_c1:
        st.markdown("**Step 1: Input Scene**")
        safe_display_image(stages["step1_original"], caption=stages.get("step1_caption"))
    with r1_c2:
        st.markdown("**Step 2: NAFNet Deblur + YOLO**")
        safe_display_image(stages["step2_nafnet"], caption=stages.get("step2_caption"))
    with r1_c3:
        st.markdown("**Step 3: YOLO Plate Crop**")
        safe_display_image(stages["step3_raw_crop"], caption=stages.get("step3_caption"))
    with r1_c4:
        st.markdown("**Step 4: Homography Rectification**")
        safe_display_image(stages["step4_rectified"], caption=stages.get("step4_caption"))

    st.write("")

    # Row 2: Stages 5 to 8
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    with r2_c1:
        st.markdown("**Step 5: Real-ESRGAN Super-Res**")
        safe_display_image(stages["step5_super_resolved"], caption=stages.get("step5_caption"))
    with r2_c2:
        st.markdown("**Step 6: Adaptive Denoise & CLAHE**")
        safe_display_image(stages["step6_clahe_denoise"], caption=stages.get("step6_caption"))
    with r2_c3:
        st.markdown("**Step 7: Final Restored Plate**")
        safe_display_image(stages["step7_enhanced"], caption=stages.get("step7_caption"))
    with r2_c4:
        st.markdown("**Step 8: Final Forensic Result**")
        st.success(f"### `{stages['ocr_text']}`")
        st.metric("Recognition Confidence", f"{stages['ocr_conf'] * 100:.1f}%")
        st.caption(f"**Strategy**: `{stages.get('strategy', 'N/A')}`")
        st.caption(f"**Winning Variant**: `{stages.get('winner', 'N/A')}`")
        if "metrics" in stages:
            m = stages["metrics"]
            st.caption(f"**Sharpness Gain**: {m.get('sharpness_before', 0):.1f} ➔ {m.get('sharpness_after', 0):.1f} ({m.get('sharpness_gain_pct', 0):+.1f}%)")


# =========================================================================
# File Discovery Helpers
# =========================================================================
def find_available_sample_images() -> List[str]:
    """Finds all available sample vehicle images on disk."""
    candidates = []
    search_dirs = [
        "data/dataset_prepared/images",
        "data/images",
        "../data/dataset_prepared/images",
        "../data/images"
    ]
    for sdir in search_dirs:
        p = Path(sdir)
        if p.is_dir():
            for f in sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")):
                candidates.append(str(f))
    return candidates


def find_available_sample_videos() -> List[str]:
    """Finds all available traffic video clips on disk."""
    candidates = []
    search_dirs = [
        "data/videos",
        "../data/videos"
    ]
    for sdir in search_dirs:
        p = Path(sdir)
        if p.is_dir():
            for f in sorted(p.glob("*.mp4")) + sorted(p.glob("*.avi")):
                candidates.append(str(f))
    return candidates


# =========================================================================
# Sidebar
# =========================================================================
with st.sidebar:
    st.title("🚗 Forensic ANPR")

    # Hardware Info
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"⚡ Hardware: {gpu_name} ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
    else:
        st.info("💻 Hardware: CPU Mode")

    st.markdown("---")
    st.markdown("""
    ### 🔬 Pipeline Modules
    - **Deblurring**: Full-scene learned motion deblurring (NAFNet FP32)
    - **Detection**: Autonomous YOLOv8 plate localization
    - **Geometry**: Multi-factor ranked homography deskewing
    - **Super-Resolution**: Standalone Real-ESRGAN x4
    - **Enhancement**: Quality-adaptive CIE LAB NLM + CLAHE
    - **OCR**: Dual-path character recognition & consensus
    """)

    st.markdown("---")
    st.markdown("""
    ### 🌐 Remote Tunnel Notice
    If running remotely on **Kaggle** or **Colab**:
    - **Recommended (Cloudflare Tunnel)**: Zero passwords, zero dropped chunks:
      `python scripts/launch_tunnel.py`
    - **Localtunnel (`loca.lt`)**: Requires entering your public IP once. The app's built-in shield automatically recovers any dropped chunks.
    """)


# =========================================================================
# Main Content Tabs
# =========================================================================
st.title("🚗 Autonomous Forensic License Plate Recognition")
st.caption("Deep restoration, super-resolution, and automated alphanumeric transcription for vehicles and traffic surveillance.")

tab_photo, tab_video = st.tabs(["📷 Photo License Plate", "🎥 Traffic Video (ANPR)"])


# =========================================================================
# TAB 1: PHOTO / IMAGE MODE
# =========================================================================
with tab_photo:
    st.header("📷 Single-Image Plate Recognition")
    st.markdown("Select a sample image from the benchmark dataset or upload your own vehicle photograph.")

    sample_images = find_available_sample_images()
    sample_options = ["-- Upload Custom Image --"] + [Path(s).name for s in sample_images]

    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        selected_sample = st.selectbox(
            "Select Sample Image",
            options=sample_options,
            index=1 if len(sample_options) > 1 else 0,
            help="Choose an existing benchmark sample or select 'Upload Custom Image'"
        )

        uploaded_image = None
        if selected_sample == "-- Upload Custom Image --":
            uploaded_image = st.file_uploader(
                "Upload a vehicle photograph",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Supported formats: JPEG, PNG, BMP"
            )

    with col_input2:
        st.write("&nbsp;")
        custom_path_input = st.text_input(
            "Or enter local image path:",
            placeholder="data/dataset_prepared/images/AN1.jpg"
        )
        if custom_path_input and os.path.isfile(custom_path_input):
            loaded = cv2.imread(custom_path_input)
            if loaded is not None:
                st.session_state["photo_img_bgr"] = loaded
                st.session_state["photo_img_name"] = Path(custom_path_input).name
                st.session_state["photo_results"] = None
                st.session_state["photo_stages"] = None

    # Resolve active image
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        loaded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if loaded is not None:
            st.session_state["photo_img_bgr"] = loaded
            st.session_state["photo_img_name"] = uploaded_image.name
            st.session_state["photo_results"] = None
            st.session_state["photo_stages"] = None
    elif selected_sample != "-- Upload Custom Image --":
        matching_path = next((s for s in sample_images if Path(s).name == selected_sample), None)
        if matching_path and os.path.isfile(matching_path):
            if st.session_state["photo_img_name"] != selected_sample:
                st.session_state["photo_img_bgr"] = cv2.imread(matching_path)
                st.session_state["photo_img_name"] = selected_sample
                st.session_state["photo_results"] = None
                st.session_state["photo_stages"] = None

    # Display image preview and execution button
    active_bgr = st.session_state.get("photo_img_bgr")
    active_name = st.session_state.get("photo_img_name", "None")

    if active_bgr is not None and getattr(active_bgr, "size", 0) > 0:
        st.markdown("---")
        col_preview, col_action = st.columns([2, 1])

        with col_preview:
            safe_display_image(
                active_bgr,
                caption=f"Input Scene: {active_name} ({active_bgr.shape[1]}x{active_bgr.shape[0]} px)",
                use_container_width=True
            )

        with col_action:
            st.info("Execute full forensic deblurring, perspective rectification, Real-ESRGAN super-resolution, and OCR.")
            run_photo_btn = st.button("🚀 Process License Plate", type="primary", use_container_width=True)

        if run_photo_btn:
            with st.spinner("Executing forensic pipeline stages..."):
                pipeline = load_pipeline()
                res = pipeline.run(input_bgr=active_bgr, track_latency=True)
                st.session_state["photo_results"] = res
                st.session_state["photo_stages"] = build_step_progression(pipeline, res, active_bgr)

        # Display results from session state
        res = st.session_state.get("photo_results")
        stages = st.session_state.get("photo_stages")

        if res is not None:
            st.markdown("---")
            st.subheader("🔍 Recognition Results")

            vis = res.get("vis")
            final_text = res.get("final_text", "NOT_DETECTED")
            final_conf = res.get("final_conf", 0.0)

            if vis is not None and res.get("detection_bbox") is not None:
                raw_crop = vis.get("raw_crop")
                enhanced_crop = None
                if vis.get("display_crop") is not None and getattr(vis.get("display_crop"), "size", 0) > 0:
                    enhanced_crop = vis.get("display_crop")
                elif vis.get("rectified_crop") is not None and getattr(vis.get("rectified_crop"), "size", 0) > 0:
                    enhanced_crop = vis.get("rectified_crop")

                c1, c2, c3 = st.columns([1, 1, 1.4])

                with c1:
                    st.caption("🔍 **1. Extracted Plate (Raw ROI)**")
                    safe_display_image(raw_crop, use_container_width=True)

                with c2:
                    st.caption("✨ **2. Enhanced Plate (Restored + SR)**")
                    safe_display_image(enhanced_crop, use_container_width=True)

                with c3:
                    st.caption("🔤 **3. Final OCR Transcription**")
                    st.success(f"### `{final_text}`")
                    st.metric(label="Recognition Confidence", value=f"{final_conf * 100:.1f}%")

                # Step-by-Step Pipeline Progression Section
                if stages is not None:
                    st.markdown("---")
                    render_step_progression_ui(stages)

            else:
                st.error("No vehicle license plate could be detected in this image.")


# =========================================================================
# TAB 2: VIDEO MODE (TRAFFIC FOOTAGE ANPR)
# =========================================================================
with tab_video:
    st.header("🎥 Traffic Video Recognition (ANPR)")
    st.markdown("Track multiple moving vehicles, filter false-positives, extract clearest observations, and recognize license plates.")

    sample_videos = find_available_sample_videos()
    sample_vid_options = ["-- Upload Custom Video --"] + [Path(s).name for s in sample_videos]

    col_vid1, col_vid2 = st.columns([2, 1])

    with col_vid1:
        selected_vid = st.selectbox(
            "Select Sample Video",
            options=sample_vid_options,
            index=1 if len(sample_vid_options) > 1 else 0,
            help="Choose an existing traffic sample or select 'Upload Custom Video'"
        )

        uploaded_vid = None
        if selected_vid == "-- Upload Custom Video --":
            uploaded_vid = st.file_uploader(
                "Upload traffic footage",
                type=["mp4", "avi", "mov", "mkv"],
                help="Supported formats: MP4, AVI, MOV"
            )

    with col_vid2:
        st.write("&nbsp;")
        custom_vid_input = st.text_input(
            "Or enter local video path:",
            placeholder="data/videos/traffic_sample.mp4"
        )
        if custom_vid_input and os.path.isfile(custom_vid_input):
            st.session_state["video_source_path"] = custom_vid_input
            st.session_state["video_results"] = None

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        process_all = st.checkbox(
            "Analyze Full Video (No Limit)",
            value=False,
            help="Process all video frames from start to finish."
        )
    with col_opt2:
        if not process_all:
            max_frames_slider = st.slider(
                "Max Frames to Analyze",
                min_value=50, max_value=1200, value=250, step=50,
                help="Limits analysis to the first N frames for rapid processing (e.g. 250 frames ≈ 8-10s)."
            )
        else:
            max_frames_slider = None
            st.caption("⚡ **Full Video Mode:** The pipeline will process every frame until the video terminates.")

    # Resolve active video
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()
        tfile.close()
        st.session_state["video_source_path"] = tfile.name
        st.session_state["video_results"] = None
    elif selected_vid != "-- Upload Custom Video --":
        matching_vid = next((s for s in sample_videos if Path(s).name == selected_vid), None)
        if matching_vid and os.path.isfile(matching_vid):
            if st.session_state["video_source_path"] != matching_vid:
                st.session_state["video_source_path"] = matching_vid
                st.session_state["video_results"] = None

    active_vid = st.session_state.get("video_source_path")
    if active_vid and os.path.isfile(active_vid):
        st.markdown("---")
        frame_limit_desc = "Entire Video" if max_frames_slider is None else f"First {max_frames_slider} frames"
        st.info(f"Target Video: `{Path(active_vid).name}` | Scope: `{frame_limit_desc}`")

        run_vid_btn = st.button("🚀 Process Traffic Video", type="primary", use_container_width=True)

        if run_vid_btn:
            with st.spinner("Tracking vehicles, extracting plates, and running forensic OCR..."):
                video_pipe = load_video_pipeline()
                verified_cars = video_pipe.process_video(
                    video_path=active_vid,
                    output_dir="results/video_anpr",
                    vehicle_model_name=DEFAULT_VEHICLE_MODEL,
                    max_frames=max_frames_slider,
                    save_visual_summary=True,
                    show_summary=False
                )
                st.session_state["video_results"] = verified_cars

        # Display video results from session state
        verified_cars = st.session_state.get("video_results")
        if verified_cars is not None:
            st.markdown("---")
            st.subheader(f"🚗 Verified Vehicles ({len(verified_cars)} Confirmed)")

            if verified_cars:
                pipe_instance = load_pipeline()

                for idx, car in enumerate(verified_cars, 1):
                    ocr_val = car.get("ocr_text", "UNKNOWN")
                    pipeline_res = car.get("pipeline_result", {})
                    vis = pipeline_res.get("vis") if pipeline_res else None

                    # 1. Vehicle image resolution (checks annotated buffer, in-memory crop, or disk paths)
                    car_crop_disp = car.get("annotated_car_img")
                    if car_crop_disp is None or getattr(car_crop_disp, "size", 0) == 0:
                        car_crop_disp = car.get("car_img")
                    if (car_crop_disp is None or getattr(car_crop_disp, "size", 0) == 0) and car.get("annotated_car_path") and os.path.isfile(car["annotated_car_path"]):
                        car_crop_disp = cv2.imread(car["annotated_car_path"])
                    if (car_crop_disp is None or getattr(car_crop_disp, "size", 0) == 0) and car.get("save_path") and os.path.isfile(car["save_path"]):
                        car_crop_disp = cv2.imread(car["save_path"])

                    car_disp = None
                    if car_crop_disp is not None and getattr(car_crop_disp, "size", 0) > 0:
                        car_disp = car_crop_disp.copy()
                        # If bbox overlay wasn't already drawn, draw it now with boundary clamping
                        if "annotated_car_img" not in car:
                            bbox = car.get("plate_bbox")
                            if bbox is not None and len(bbox) == 4:
                                bx1, by1, bx2, by2 = bbox
                                h, w = car_disp.shape[:2]
                                bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                                bx2, by2 = min(w, int(bx2)), min(h, int(by2))
                                if bx2 > bx1 and by2 > by1:
                                    cv2.rectangle(car_disp, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                    # 2. Plate crops resolution
                    raw_crop = vis.get("raw_crop") if vis else None
                    if raw_crop is None or getattr(raw_crop, "size", 0) == 0:
                        # Extract from vehicle crop using plate_bbox
                        bbox = car.get("plate_bbox")
                        c_img = car.get("car_img")
                        if bbox is not None and len(bbox) == 4 and c_img is not None and getattr(c_img, "size", 0) > 0:
                            bx1, by1, bx2, by2 = bbox
                            ch, cw = c_img.shape[:2]
                            bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                            bx2, by2 = min(cw, int(bx2)), min(ch, int(by2))
                            if bx2 > bx1 and by2 > by1:
                                raw_crop = c_img[by1:by2, bx1:bx2]
                    if raw_crop is None or getattr(raw_crop, "size", 0) == 0:
                        raw_crop = car_disp

                    enhanced_crop = None
                    if vis:
                        if vis.get("display_crop") is not None and getattr(vis.get("display_crop"), "size", 0) > 0:
                            enhanced_crop = vis.get("display_crop")
                        elif vis.get("rectified_crop") is not None and getattr(vis.get("rectified_crop"), "size", 0) > 0:
                            enhanced_crop = vis.get("rectified_crop")
                        elif "candidate_variants" in vis:
                            winner = car.get("winning_candidate")
                            variants = vis["candidate_variants"]
                            if winner in variants and getattr(variants[winner], "size", 0) > 0:
                                enhanced_crop = variants[winner]
                            elif "Adaptive_Targeted" in variants:
                                enhanced_crop = variants["Adaptive_Targeted"]

                    if enhanced_crop is None or getattr(enhanced_crop, "size", 0) == 0:
                        enhanced_crop = raw_crop

                    with st.container():
                        st.markdown(f"**Vehicle #{idx} (Track ID: {car['track_id']}, Frame #{car['frame_idx']})**")
                        c_car, c_plate, c_enh, c_ocr = st.columns([1.4, 1.0, 1.0, 1.4])

                        with c_car:
                            st.caption("🚗 **1. Extracted Vehicle**")
                            safe_display_image(car_disp, use_container_width=True)

                        with c_plate:
                            st.caption("🔍 **2. Extracted Plate**")
                            safe_display_image(raw_crop, use_container_width=True)

                        with c_enh:
                            st.caption("✨ **3. Enhanced Plate**")
                            safe_display_image(enhanced_crop, use_container_width=True)

                        with c_ocr:
                            st.caption("🔤 **4. Final OCR Value**")
                            st.success(f"### `{ocr_val}`")
                            st.caption(f"Strategy: `{car.get('selection_strategy', 'N/A')}`")

                        # Expandable Step-by-Step Visualization for this vehicle
                        car_img_input = car.get("car_img")
                        if pipeline_res and car_img_input is not None and getattr(car_img_input, "size", 0) > 0:
                            with st.expander(f"🔬 View Step-by-Step Pipeline Progression (Vehicle #{idx}: {ocr_val})"):
                                v_stages = build_step_progression(pipe_instance, pipeline_res, car_img_input)
                                render_step_progression_ui(v_stages, title_prefix=f"Vehicle #{idx}")

                        st.markdown("---")
            else:
                st.warning("No readable license plates were detected in the processed video frames.")
