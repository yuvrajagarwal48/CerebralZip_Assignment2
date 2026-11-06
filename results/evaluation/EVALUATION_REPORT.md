# Forensic License Plate Super-Resolution, Restoration & OCR
## Comprehensive Standardized Pipeline Evaluation Report

---

### Executive Performance Summary
- **Total Test Samples Evaluated**: **300 samples** (Denominator N = 300)
- **End-to-End Exact Match Accuracy**: **50.0%**
- **Mean (Macro) Character Error Rate (CER)**: **0.181**
- **Normalized Macro Character Accuracy**: **81.9%**
- **Corpus (Micro) CER**: **0.181**
- **Corpus (Micro) Character Accuracy**: **81.9%**
- **YOLO Detection Recall @ 0.50 IoU**: **99.0%** (True AP@0.50 = 0.987)
- **Mean Localization IoU**: **0.791** (All plates)
- **Average Pipeline Latency**: **3713.0 ms / frame** (~**0.3 FPS** on cuda:0)

---

### 1. License Plate Detection & Localization (YOLOv8)
| Metric | Value | Description |
| :--- | :---: | :--- |
| **Precision @ 0.50 IoU** | 99.7% | TP / (TP + FP) at IoU >= 0.50 |
| **Recall @ 0.50 IoU** | 99.0% | TP / Total Manifest at IoU >= 0.50 |
| **F1-Score @ 0.50 IoU** | 0.993 | Harmonic mean of detection precision and recall |
| **True AP @ 0.50** | 0.987 | Area under Precision-Recall curve at IoU 0.50 (101-point COCO interpolation) |
| **True mAP @ 0.50:0.95** | 0.534 | COCO mAP averaged across IoU thresholds 0.50 to 0.95 |
| **Mean IoU (All Plates)** | 0.791 | Average box overlap across all GT plates (misses counted as 0.0) |
| **Mean IoU (Detected Only)** | 0.796 | Average box overlap conditioned on localization |
| **Detection Failure Rate** | 0.7% | Percentage of images where plate was unlocalized |

**Detection Stratification by Plate Size:**
| Scale Bucket | Area Criteria | Test Count | Recall @ 0.50 | Mean IoU (All) | Mean IoU (Det) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Medium** | 32² - 96² px | 167 | 100.0% | 0.781 | 0.781 |
| **Large** | >= 96² px | 99 | 97.0% | 0.836 | 0.854 |
| **Small** | < 32² px | 34 | 100.0% | 0.706 | 0.706 |

---

### 2. Geometric Rectification & Deskewing
| Metric | Measured Value | Target Standard |
| :--- | :---: | :---: |
| **Mean Residual Skew Angle** | **0.23°** | < 1.0° |
| **Mean Aspect Ratio Error** | **19.4%** | < 15.0% |

---

### 3. Three-Way Baseline Comparison: Progressive Enhancement Impact
| Evaluation Mode | Description | Exact Match | Mean CER | Char Accuracy | Corpus CER |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original GT Plate Crop** | Original GT Plate Crop -> OCR | **51.7%** | **0.171** | **82.9%** | 0.170 |
| **Enhanced GT Plate Crop** | Enhanced GT Plate Crop -> OCR | **50.3%** | **0.171** | **82.9%** | 0.171 |
| **Full Autonomous Pipeline** | Full Autonomous Pipeline -> OCR | **49.3%** | **0.184** | **81.5%** | 0.184 |

---

### 4. OCR Error & Character Confusion Analysis
#### Top Character Substitution Pairs
| Substitution (Actual -> Predicted) | Occurrences | Typical Root Cause |
| :---: | :---: | :--- |
| **'M' -> 'H'** | 11 | Morphological character geometry similarity |
| **'0' -> 'O'** | 11 | Morphological character geometry similarity |
| **'B' -> '8'** | 10 | Morphological character geometry similarity |
| **'4' -> '2'** | 9 | Morphological character geometry similarity |
| **'1' -> 'I'** | 6 | Morphological character geometry similarity |
| **'M' -> 'N'** | 6 | Morphological character geometry similarity |

#### Alignment Error Breakdown
- **Total Edit Distance (Errors)**: 522 character errors across 2882 ground-truth characters
- **Top Deletions**: 'M' (32x), 'H' (28x), '0' (27x), '1' (26x), '7' (21x)
- **Top Insertions**: '1' (8x), 'E' (4x), 'N' (3x), 'K' (2x), 'F' (2x)

---

### 5. Runtime & Throughput Profile
- **Average Frame Latency**: **3713.0 ms**
- **System Throughput**: **0.3 frames / second**

| Pipeline Stage | Mean Execution Time (ms) |
| :--- | :---: |
| **Deblurring** | 576.21 ms |
| **Detection** | 66.24 ms |
| **Geometry** | 4.80 ms |
| **Qa** | 10.11 ms |
| **Routing And Restoration** | 204.21 ms |
| **Ocr And Selection** | 2851.09 ms |
| **Evaluation** | 0.00 ms |

---
*Report generated autonomously by ForensicPipelineEvaluator.*