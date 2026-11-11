#!/usr/bin/env python3
"""
app.py: Streamlit Web Application for Forensic License Plate Recognition
========================================================================
Interactive web dashboard that accepts photos or traffic videos and displays:
  1. Extracted Plate (Raw ROI Bounding Box Crop)
  2. Enhanced Plate (Deep Restored & Super-Resolved Plate)
  3. Final OCR Value (Autonomous OCR Transcription)

Uses native Streamlit components and session state management to guarantee
seamless execution across local environments, Google Colab, and Kaggle.
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

# Initialize Session State
if "photo_img_bgr" not in st.session_state:
    st.session_state["photo_img_bgr"] = None
if "photo_img_name" not in st.session_state:
    st.session_state["photo_img_name"] = None
if "photo_results" not in st.session_state:
    st.session_state["photo_results"] = None
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
# Helper Functions for File Discovery
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
    - **Detection**: Autonomous YOLOv8 plate localization
    - **Restoration**: Learned scene motion deblurring (NAFNet FP32)
    - **Geometry**: Multi-factor ranked homography deskewing
    - **Super-Resolution**: Standalone Real-ESRGAN x4
    - **OCR**: Dual-path character recognition
    """)

    st.markdown("---")
    st.markdown("""
    ### 🌐 Running Remotely?
    If running on **Kaggle** or **Colab**:
    - **Cloudflare Tunnel (Recommended)**: Zero setup, no passwords or dynamic module chunk errors:
      `./cloudflared tunnel --url http://localhost:8501`
    - **Localtunnel (`loca.lt`)**: Requires submitting your container public IP (`curl -s ipv4.icanhazip.com`) once on the reminder page, then hard-refreshing (`Ctrl+Shift+R`).
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

    # Resolve active image
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        loaded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if loaded is not None:
            st.session_state["photo_img_bgr"] = loaded
            st.session_state["photo_img_name"] = uploaded_image.name
            st.session_state["photo_results"] = None
    elif selected_sample != "-- Upload Custom Image --":
        matching_path = next((s for s in sample_images if Path(s).name == selected_sample), None)
        if matching_path and os.path.isfile(matching_path):
            if st.session_state["photo_img_name"] != selected_sample:
                st.session_state["photo_img_bgr"] = cv2.imread(matching_path)
                st.session_state["photo_img_name"] = selected_sample
                st.session_state["photo_results"] = None

    # Display image preview and execution button
    active_bgr = st.session_state.get("photo_img_bgr")
    active_name = st.session_state.get("photo_img_name", "None")

    if active_bgr is not None and getattr(active_bgr, "size", 0) > 0:
        st.markdown("---")
        col_preview, col_action = st.columns([2, 1])

        with col_preview:
            st.image(
                cv2.cvtColor(active_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Input Scene: {active_name} ({active_bgr.shape[1]}x{active_bgr.shape[0]} px)",
                use_container_width=True
            )

        with col_action:
            st.info("Execute full forensic deblurring, perspective rectification, Real-ESRGAN super-resolution, and OCR.")
            run_photo_btn = st.button("🚀 Process License Plate", type="primary", use_container_width=True)

        if run_photo_btn:
            with st.spinner("Executing forensic pipeline stages..."):
                pipeline = load_pipeline()
                res = pipeline.run(input_bgr=active_bgr, track_latency=False)
                st.session_state["photo_results"] = res

        # Display results from session state
        res = st.session_state.get("photo_results")
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
                    if raw_crop is not None and getattr(raw_crop, "size", 0) > 0:
                        st.image(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.warning("Extracted crop unavailable")

                with c2:
                    st.caption("✨ **2. Enhanced Plate (Restored + SR)**")
                    if enhanced_crop is not None and getattr(enhanced_crop, "size", 0) > 0:
                        st.image(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.warning("Enhanced crop unavailable")

                with c3:
                    st.caption("🔤 **3. Final OCR Transcription**")
                    st.success(f"### `{final_text}`")
                    st.metric(label="Recognition Confidence", value=f"{final_conf * 100:.1f}%")
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
                for idx, car in enumerate(verified_cars, 1):
                    ocr_val = car.get("ocr_text", "UNKNOWN")
                    pipeline_res = car.get("pipeline_result", {})
                    vis = pipeline_res.get("vis") if pipeline_res else None

                    raw_crop = None
                    enhanced_crop = None
                    if vis:
                        raw_crop = vis.get("raw_crop")
                        if vis.get("display_crop") is not None and getattr(vis.get("display_crop"), "size", 0) > 0:
                            enhanced_crop = vis.get("display_crop")
                        elif vis.get("rectified_crop") is not None and getattr(vis.get("rectified_crop"), "size", 0) > 0:
                            enhanced_crop = vis.get("rectified_crop")

                    if raw_crop is None or getattr(raw_crop, "size", 0) == 0:
                        raw_crop = car.get("car_img")
                    if enhanced_crop is None or getattr(enhanced_crop, "size", 0) == 0:
                        enhanced_crop = raw_crop

                    # Prepare vehicle image with localized plate bounding box overlay
                    car_crop_disp = car.get("car_img")
                    car_disp = None
                    if car_crop_disp is not None and getattr(car_crop_disp, "size", 0) > 0:
                        car_disp = car_crop_disp.copy()
                        bbox = car.get("plate_bbox")
                        if bbox is not None and len(bbox) == 4:
                            bx1, by1, bx2, by2 = bbox
                            cv2.rectangle(car_disp, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                    with st.container():
                        st.markdown(f"**Vehicle #{idx} (Track ID: {car['track_id']}, Frame #{car['frame_idx']})**")
                        c_car, c_plate, c_enh, c_ocr = st.columns([1.4, 1.0, 1.0, 1.4])

                        with c_car:
                            st.caption("🚗 **1. Extracted Vehicle**")
                            if car_disp is not None and getattr(car_disp, "size", 0) > 0:
                                st.image(cv2.cvtColor(car_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.warning("Vehicle image unavailable")

                        with c_plate:
                            st.caption("🔍 **2. Extracted Plate**")
                            if raw_crop is not None and getattr(raw_crop, "size", 0) > 0:
                                st.image(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.warning("Extracted crop unavailable")

                        with c_enh:
                            st.caption("✨ **3. Enhanced Plate**")
                            if enhanced_crop is not None and getattr(enhanced_crop, "size", 0) > 0:
                                st.image(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.warning("Enhanced crop unavailable")

                        with c_ocr:
                            st.caption("🔤 **4. Final OCR Value**")
                            st.success(f"### `{ocr_val}`")
                            st.caption("Verified Vehicle License Plate")

                        st.markdown("---")
            else:
                st.warning("No readable license plates were detected in the processed video frames.")
