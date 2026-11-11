# Forensic Step-by-Step Pipeline Progression Report

A granular 8-stage visualization report detailing intermediate visual transformations for all 15 comparative samples.

### Pipeline Stages Captured in Each Figure:
1. **Original Image**: Raw source capture from vehicle photo or traffic video.
2. **NAFNet Deblurred Scene**: Full-scene learned motion deblurring with YOLO plate detection box.
3. **YOLO Plate Crop**: Isolated bounding box ROI before enhancement.
4. **Homography Rectification**: Perspective deskewing using ranked candidate contours with corner-derived dimensions.
5. **Real-ESRGAN x4 (2x Scaled)**: Deep RRDBNet super-resolution in FP32.
6. **Adaptive Denoise & Contrast**: CIE LAB space NLM denoising and CLAHE dynamic range expansion.
7. **Final Enhanced Plate**: Unsharp mask stroke sharpening.
8. **Final Forensic Result**: RapidOCR consensus recognition, ground truth, PSNR, SSIM, and confidence.

## 🔬 Sample Progression Table

| # | Filename | Recognized OCR | Ground Truth | PSNR (dB) | SSIM | Sharpness Gain | Step Progression Figure |
|---|---|---|---|---|---|---|---|
| 01 | `AR13.jpg` | **AR200692** | `AR200692` | 8.70 | 0.2327 | +11166.8% | [`step_progression_01_AR13.png`](./step_progression_01_AR13.png) |
| 02 | `AN1.jpg` | **AN01P9687** | `AN01P9687` | 7.20 | 0.1354 | +15986.4% | [`step_progression_02_AN1.png`](./step_progression_02_AN1.png) |
| 03 | `AN2.jpg` | **AND1D4063** | `AN01D4163` | 6.92 | 0.1075 | +21505.2% | [`step_progression_03_AN2.png`](./step_progression_03_AN2.png) |
| 04 | `AN4.jpg` | **AN01M4M** | `AN01M4M` | 10.20 | 0.1708 | +10637.2% | [`step_progression_04_AN4.png`](./step_progression_04_AN4.png) |
| 05 | `AN5.jpg` | **ANO1L6155** | `AN01L6155` | 7.85 | 0.2179 | +19247.6% | [`step_progression_05_AN5.png`](./step_progression_05_AN5.png) |
| 06 | `AN6.jpg` | **PANOTHI689** | `AN01H1689` | 8.58 | 0.2803 | +38115.3% | [`step_progression_06_AN6.png`](./step_progression_06_AN6.png) |
| 07 | `AN7.jpg` | **AN01H0908** | `AN01H0908` | 12.08 | 0.0877 | +5009.1% | [`step_progression_07_AN7.png`](./step_progression_07_AN7.png) |
| 08 | `AN10.jpg` | **PANO1X9412** | `AN01K9412` | 8.51 | 0.1210 | +63920.9% | [`step_progression_08_AN10.png`](./step_progression_08_AN10.png) |
| 09 | `car_1_frame_88.jpg` | **657648** | `N/A` | 6.33 | -0.0018 | +1403.6% | [`step_progression_09_car_1_frame_88.png`](./step_progression_09_car_1_frame_88.png) |
| 10 | `car_3_frame_221.jpg` | **KA02HN182G** | `N/A` | 9.45 | 0.2853 | +5338.8% | [`step_progression_10_car_3_frame_221.png`](./step_progression_10_car_3_frame_221.png) |
| 11 | `car_4_frame_260.jpg` | **1073** | `N/A` | 8.51 | 0.2513 | +1356.4% | [`step_progression_11_car_4_frame_260.png`](./step_progression_11_car_4_frame_260.png) |
| 12 | `car_6_frame_265.jpg` | **KA02HN1826** | `N/A` | 7.74 | 0.0431 | +287.3% | [`step_progression_12_car_6_frame_265.png`](./step_progression_12_car_6_frame_265.png) |
| 13 | `car_7_frame_306.jpg` | **3789** | `N/A` | 8.44 | 0.1777 | +1521.2% | [`step_progression_13_car_7_frame_306.png`](./step_progression_13_car_7_frame_306.png) |
| 14 | `car_8_frame_315.jpg` | **NOT_DETECTED** | `N/A` | 7.92 | 0.2155 | +2269.7% | [`step_progression_14_car_8_frame_315.png`](./step_progression_14_car_8_frame_315.png) |
| 15 | `car_10_frame_335.jpg` | **EKH9211** | `N/A` | 12.99 | 0.4055 | +1289.3% | [`step_progression_15_car_10_frame_335.png`](./step_progression_15_car_10_frame_335.png) |
