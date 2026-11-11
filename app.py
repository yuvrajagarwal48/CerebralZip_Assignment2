#!/usr/bin/env python3
"""
app.py: Streamlit Web Application for Forensic License Plate Recognition
========================================================================
Interactive web dashboard that accepts photos or traffic videos and displays:
  1. Extracted Plate (Raw ROI Bounding Box Crop)
  2. Enhanced Plate (Deep Restored & Super-Resolved Plate)
  3. Final OCR Value (Autonomous OCR Transcription)

Strictly displays only the core visual components and final recognized plate
values without extraneous technical evaluation metrics.
"""

import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

# Import pipeline components
from scripts.pipeline import MainImagePipeline
from scripts.video_pipeline import ForensicVideoPipeline, DEFAULT_VEHICLE_MODEL


# =========================================================================
# Streamlit App Configuration & Styling
# =========================================================================
st.set_page_config(
    page_title="Forensic License Plate OCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .plate-badge {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #3b82f6;
        border-radius: 8px;
        padding: 18px 24px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .plate-text {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 5px;
        color: #f8fafc;
        margin: 0;
    }
    .plate-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .card-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


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
# Sidebar
# =========================================================================
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.selectbox(
    "Select Processing Mode",
    ["📷 Photo / Image", "🎥 Video (Traffic Footage)"],
    index=0
)

# Hardware info
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
if num_gpus > 0:
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"⚡ Hardware Acceleration: {gpu_name} ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
else:
    st.sidebar.info("💻 Hardware Acceleration: CPU Mode")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 About This App
This application performs autonomous vehicle license plate recognition:
- **Detection**: Dedicated YOLOv8 plate localization
- **Restoration**: Learned scene motion deblurring & geometric rectification
- **Super-Resolution**: Standalone Real-ESRGAN x4
- **OCR**: Dual-path character recognition
""")


# =========================================================================
# PHOTO / IMAGE MODE
# =========================================================================
if mode == "📷 Photo / Image":
    st.markdown("<div class='main-header'>📷 Photo License Plate Recognition</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload a vehicle image to extract, restore, and read the license plate.</div>", unsafe_allow_html=True)

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPEG, PNG, BMP"
        )
    with col_demo:
        st.write("&nbsp;")
        use_demo_image = st.button("🧪 Use Sample Demo Image", use_container_width=True)

    # Determine input image
    input_bgr = None
    input_filename = "Sample"

    demo_path = None
    default_candidates = [
        "data/dataset_prepared/images/AN1.jpg",
        "data/dataset_prepared/images/AR13.jpg",
        "data/dataset_prepared/images/AN2.jpg"
    ]
    for cand in default_candidates:
        if os.path.exists(cand):
            demo_path = cand
            break

    if demo_path is None:
        search_dirs = ["data/dataset_prepared/images", "data/images"]
        for sdir in search_dirs:
            if os.path.isdir(sdir):
                cands = list(Path(sdir).glob("*.jpg")) + list(Path(sdir).glob("*.png"))
                if cands:
                    demo_path = str(cands[0])
                    break

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        input_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_filename = uploaded_image.name
    elif use_demo_image:
        if demo_path and os.path.exists(demo_path):
            input_bgr = cv2.imread(demo_path)
            input_filename = Path(demo_path).name
        else:
            st.warning("⚠️ No sample demo image found on disk. Please upload an image file.")

    if input_bgr is not None:
        st.markdown("---")
        # Display full uploaded image
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB), caption=f"Uploaded Scene: {input_filename}", use_container_width=True)
        with col_info:
            st.info("Click the button below to execute the full forensic super-resolution and OCR pipeline on this image.")
            run_btn = st.button("🚀 Process License Plate", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Processing image through forensic pipeline..."):
                pipeline = load_pipeline()
                res = pipeline.run(input_bgr=input_bgr, track_latency=False)

            vis = res.get("vis")
            final_text = res.get("final_text", "NOT_DETECTED")

            st.markdown("### 🔍 Forensic Recognition Results")

            if vis is not None and res.get("detection_bbox") is not None:
                raw_crop = vis.get("raw_crop")
                enhanced_crop = None
                if vis.get("display_crop") is not None and getattr(vis.get("display_crop"), "size", 0) > 0:
                    enhanced_crop = vis.get("display_crop")
                elif vis.get("rectified_crop") is not None and getattr(vis.get("rectified_crop"), "size", 0) > 0:
                    enhanced_crop = vis.get("rectified_crop")

                # 3-Column Display: Extracted Plate | Enhanced Plate | Final OCR Value
                c1, c2, c3 = st.columns([1, 1, 1.4])

                with c1:
                    st.markdown("#### 1. Extracted Plate")
                    if raw_crop is not None and getattr(raw_crop, "size", 0) > 0:
                        st.image(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.warning("Extracted crop unavailable")

                with c2:
                    st.markdown("#### 2. Enhanced Plate")
                    if enhanced_crop is not None and getattr(enhanced_crop, "size", 0) > 0:
                        st.image(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.warning("Enhanced crop unavailable")

                with c3:
                    st.markdown("#### 3. Final OCR Value")
                    st.markdown(f"""
                    <div class='plate-badge'>
                        <p class='plate-text'>{final_text}</p>
                        <p class='plate-sub'>Verified License Plate</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error("No license plate could be localized in this image.")


# =========================================================================
# VIDEO MODE
# =========================================================================
elif mode == "🎥 Video (Traffic Footage)":
    st.markdown("<div class='main-header'>🎥 Video License Plate Recognition (ANPR)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload traffic footage to track vehicles, filter no-plate false positives, and recognize license plates.</div>", unsafe_allow_html=True)

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV"
        )
    with col_demo:
        st.write("&nbsp;")
        use_demo_video = st.button("🧪 Use Sample Traffic Video", use_container_width=True)

    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        process_all = st.checkbox("Analyze Full Video (No Limit)", value=False,
                                  help="Process every frame from start to finish. Useful for complete ANPR on shorter clips.")
    with col_opt2:
        if not process_all:
            max_frames_slider = st.slider(
                "Max Frames to Analyze", min_value=50, max_value=1200, value=250, step=50,
                help="Limits analysis to the first N frames for quick feedback (e.g. 250 frames ≈ 8-10s of traffic footage)."
            )
        else:
            max_frames_slider = None
            st.caption("⚡ **Full Video Mode:** The pipeline will process all frames until the video terminates.")

    video_source_path = None
    demo_video_path = None
    default_vid = "data/videos/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4"
    if os.path.exists(default_vid):
        demo_video_path = default_vid
    elif os.path.isdir("data/videos"):
        cands = list(Path("data/videos").glob("*.mp4")) + list(Path("data/videos").glob("*.avi"))
        if cands:
            demo_video_path = str(cands[0])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close()
        video_source_path = tfile.name
    elif use_demo_video:
        if demo_video_path and os.path.exists(demo_video_path):
            video_source_path = demo_video_path
        else:
            st.warning("⚠️ No sample demo video found on disk. Please upload a video file.")

    if video_source_path is not None:
        st.markdown("---")
        frame_limit_desc = "Entire Video" if max_frames_slider is None else f"First {max_frames_slider} frames"
        st.info(f"Target Video: `{os.path.basename(video_source_path)}` | Scope: `{frame_limit_desc}`")

        run_vid_btn = st.button("🚀 Process Traffic Video", type="primary", use_container_width=True)

        if run_vid_btn:
            with st.spinner("Tracking vehicles, extracting plates, and running forensic OCR..."):
                video_pipe = load_video_pipeline()
                verified_cars = video_pipe.process_video(
                    video_path=video_source_path,
                    output_dir="results/video_anpr",
                    vehicle_model_name=DEFAULT_VEHICLE_MODEL,
                    max_frames=max_frames_slider,
                    save_visual_summary=True,
                    show_summary=False
                )

            st.markdown(f"### 🚗 Verified Vehicles ({len(verified_cars)} Confirmed)")

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

                    # Fallback to vehicle crop if plate crop is missing
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
                            st.caption("🚗 1. Extracted Vehicle")
                            if car_disp is not None and getattr(car_disp, "size", 0) > 0:
                                st.image(cv2.cvtColor(car_disp, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.warning("Vehicle image unavailable")

                        with c_plate:
                            st.caption("🔍 2. Extracted Plate")
                            if raw_crop is not None and getattr(raw_crop, "size", 0) > 0:
                                st.image(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.warning("Extracted crop unavailable")

                        with c_enh:
                            st.caption("✨ 3. Enhanced Plate")
                            if enhanced_crop is not None and getattr(enhanced_crop, "size", 0) > 0:
                                st.image(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                            else:
                                st.warning("Enhanced crop unavailable")

                        with c_ocr:
                            st.caption("🔤 4. Final OCR Value")
                            st.markdown(f"""
                            <div class='plate-badge'>
                                <p class='plate-text'>{ocr_val}</p>
                                <p class='plate-sub'>Verified Vehicle License Plate</p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)
            else:
                st.warning("No readable license plates were detected in the processed video frames.")

