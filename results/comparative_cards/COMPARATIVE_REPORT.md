# Forensic 15-Sample Comparative Evaluation Report

A quantitative and visual before/after analysis across 15 real-world vehicle images and video ANPR frames.

## 📊 Summary Metrics

- **Total Samples Processed**: 15
- **Mean PSNR (dB)**: 12.32 dB
- **Mean SSIM**: 0.5054
- **Mean Laplacian Sharpness Gain**: +1147.4%

## 🔬 Sample-by-Sample Breakdown

| # | Sample Filename | Recognized OCR | Ground Truth | PSNR (dB) | SSIM | Sharpness Gain | Composite Card |
|---|---|---|---|---|---|---|---|
| 01 | `AR13.jpg` | **AR200692** | `AR200692` | 13.43 | 0.6684 | +1199.2% | [`comparative_card_01_AR13.png`](./comparative_card_01_AR13.png) |
| 02 | `AN1.jpg` | **AN01P9687** | `AN01P9687` | 6.23 | 0.1954 | +223.2% | [`comparative_card_02_AN1.png`](./comparative_card_02_AN1.png) |
| 03 | `AN2.jpg` | **ANO1D4163** | `AN01D4163` | 10.64 | 0.5002 | +1733.8% | [`comparative_card_03_AN2.png`](./comparative_card_03_AN2.png) |
| 04 | `AN4.jpg` | **AN01H4H** | `AN01M4M` | 16.73 | 0.7287 | +860.8% | [`comparative_card_04_AN4.png`](./comparative_card_04_AN4.png) |
| 05 | `AN5.jpg` | **ANOIL6155** | `AN01L6155` | 7.37 | 0.2207 | +6323.5% | [`comparative_card_05_AN5.png`](./comparative_card_05_AN5.png) |
| 06 | `AN6.jpg` | **AHOIHI689** | `AN01H1689` | 15.21 | 0.6549 | +1622.3% | [`comparative_card_06_AN6.png`](./comparative_card_06_AN6.png) |
| 07 | `AN7.jpg` | **AN01H0906** | `AN01H0908` | 13.15 | 0.3441 | +2691.5% | [`comparative_card_07_AN7.png`](./comparative_card_07_AN7.png) |
| 08 | `AN10.jpg` | **BANO1K9412** | `AN01K9412` | 15.07 | 0.6502 | +1348.3% | [`comparative_card_08_AN10.png`](./comparative_card_08_AN10.png) |
| 09 | `car_1_frame_88.jpg` | **657648** | `N/A` | 13.98 | 0.7543 | +39.3% | [`comparative_card_09_car_1_frame_88.png`](./comparative_card_09_car_1_frame_88.png) |
| 10 | `car_3_frame_221.jpg` | **KA02HN182G** | `N/A` | 16.58 | 0.8009 | +467.1% | [`comparative_card_10_car_3_frame_221.png`](./comparative_card_10_car_3_frame_221.png) |
| 11 | `car_4_frame_260.jpg` | **1073** | `N/A` | 7.06 | 0.0907 | -13.3% | [`comparative_card_11_car_4_frame_260.png`](./comparative_card_11_car_4_frame_260.png) |
| 12 | `car_6_frame_265.jpg` | **KA02MN1826** | `N/A` | 7.79 | 0.1752 | +676.3% | [`comparative_card_12_car_6_frame_265.png`](./comparative_card_12_car_6_frame_265.png) |
| 13 | `car_7_frame_306.jpg` | **3789** | `N/A` | 14.63 | 0.7407 | +72.0% | [`comparative_card_13_car_7_frame_306.png`](./comparative_card_13_car_7_frame_306.png) |
| 14 | `car_8_frame_315.jpg` | **C64457T** | `N/A` | 6.23 | 0.0948 | -47.0% | [`comparative_card_14_car_8_frame_315.png`](./comparative_card_14_car_8_frame_315.png) |
| 15 | `car_10_frame_335.jpg` | **FKH9211** | `N/A` | 20.73 | 0.9610 | +13.2% | [`comparative_card_15_car_10_frame_335.png`](./comparative_card_15_car_10_frame_335.png) |
