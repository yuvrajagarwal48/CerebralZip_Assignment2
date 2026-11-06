"""
data_prep.py: Unified Real-World Dataset Preparation Pipeline
==============================================================
Consolidates the complete Indian Vehicle License Plate Dataset across all
subdirectories into a single unified dataset directory.

Key Architecture:
- Discovers and pairs all real-world images (.jpg, .jpeg, .png, .bmp) with their
  corresponding Pascal VOC XML annotations.
- Zero synthetic degradation / artificial noise generation.
- Robust filename collision handling across source subfolders.
- Annotation filenames (.xml and .txt) match image filenames exactly.
- Preserves multi-plate annotations for images containing multiple vehicles/plates.
- Validates that every image file physically exists before registration.
- Exports ground truth in clean, standard formats:
    1. manifest.json (clean metadata without redundant fields)
    2. ground_truth.csv (tabular metadata)
    3. labels/ (YOLO format: multi-line normalized bounding boxes)
    4. annotations/ (Pascal VOC XML synchronized with new filenames)
    5. dataset_summary.json (dataset statistics)
"""

import os
import sys
import re
import cv2
import json
import shutil
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}


def find_image_for_xml(xml_path: Path, parent_dir: Path) -> Optional[Path]:
    """
    Finds the corresponding image file for a given Pascal VOC XML file.
    Tries multiple resolution strategies:
    1. Direct <filename> tag inside XML.
    2. <filename> basename inside XML parent directory.
    3. Direct stem matching with known image extensions.
    4. Compound extension matching (e.g. .jpg.jpeg).
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        fn = root.findtext("filename")
        if fn:
            fn_clean = fn.strip()
            # Direct path check
            cand1 = parent_dir / fn_clean
            if cand1.is_file():
                return cand1
            # Basename check
            cand2 = parent_dir / Path(fn_clean).name
            if cand2.is_file():
                return cand2
    except Exception:
        pass

    # Stem matching with image extensions
    for ext in IMAGE_EXTENSIONS:
        cand = xml_path.with_suffix(ext)
        if cand.is_file():
            return cand
        cand_compound = parent_dir / (xml_path.stem + ext)
        if cand_compound.is_file():
            return cand_compound

    return None


def parse_voc_annotation(xml_path: Path, image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parses Pascal VOC XML for license plate annotations.
    Extracts ALL valid license plate objects in the image, preserving multi-plate
    annotations, validated bounding boxes, and normalized YOLO coordinates.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Parse image dimensions from XML
        size_elem = root.find("size")
        xml_w = int(float(size_elem.findtext("width", "0"))) if size_elem is not None else 0
        xml_h = int(float(size_elem.findtext("height", "0"))) if size_elem is not None else 0

        # Read actual image dimensions from image header
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            return None

        actual_h, actual_w = img.shape[:2]
        img_w = actual_w if actual_w > 0 else xml_w
        img_h = actual_h if actual_h > 0 else xml_h

        if img_w <= 0 or img_h <= 0:
            return None

        objects = root.findall("object")
        if not objects:
            return None

        plates = []
        for obj in objects:
            raw_name = obj.findtext("name")
            if not raw_name:
                continue

            clean_text = "".join([c for c in raw_name.strip().upper() if c.isalnum()])
            if len(clean_text) < 2:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            raw_xmin = float(bndbox.findtext("xmin", "0"))
            raw_ymin = float(bndbox.findtext("ymin", "0"))
            raw_xmax = float(bndbox.findtext("xmax", str(img_w)))
            raw_ymax = float(bndbox.findtext("ymax", str(img_h)))

            # Clamp bounding box coordinates to image boundaries
            xmin = max(0, min(int(round(raw_xmin)), img_w - 1))
            ymin = max(0, min(int(round(raw_ymin)), img_h - 1))
            xmax = max(xmin + 1, min(int(round(raw_xmax)), img_w))
            ymax = max(ymin + 1, min(int(round(raw_ymax)), img_h))

            # YOLO format: normalized x_center, y_center, width, height in [0.0, 1.0]
            box_w = xmax - xmin
            box_h = ymax - ymin
            x_center = (xmin + box_w / 2.0) / img_w
            y_center = (ymin + box_h / 2.0) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h

            plates.append({
                "ground_truth_text": clean_text,
                "raw_text": raw_name.strip(),
                "gt_bbox": [xmin, ymin, xmax, ymax],
                "yolo_box": [round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6)]
            })

        if not plates:
            return None

        return {
            "plates": plates,
            "image_width": img_w,
            "image_height": img_h
        }
    except Exception as e:
        print(f"[!] Warning: Failed to parse XML '{xml_path}': {e}")
        return None


def discover_all_dataset_pairs(input_root: Path) -> List[Tuple[Path, Path, str]]:
    """
    Recursively scans all subdirectories of input_root, identifying all valid
    (image_path, xml_path, source_subfolder) pairs across all folders.
    """
    print(f"[*] Scanning input directory: {input_root.resolve()}")
    xml_set = set(input_root.rglob("*.xml")) | set(input_root.rglob("*.XML"))
    all_xmls = sorted(list(xml_set))
    print(f"[*] Discovered {len(all_xmls)} XML annotation files across all folders.")

    valid_pairs = []
    missing_images = []

    for xml_path in all_xmls:
        rel_parent = xml_path.parent.relative_to(input_root)
        source_folder = str(rel_parent).replace("\\", "/")

        image_path = find_image_for_xml(xml_path, xml_path.parent)
        if image_path is not None and image_path.is_file():
            valid_pairs.append((image_path, xml_path, source_folder))
        else:
            missing_images.append(xml_path)

    print(f"[✓] Successfully matched {len(valid_pairs)} image-annotation pairs.")
    if missing_images:
        print(f"[!] Note: {len(missing_images)} XML annotations had no matching image file on disk.")

    return valid_pairs


def prepare_unified_dataset(
    input_dir: str = "./data/images",
    output_dir: str = "./data/dataset_prepared",
    copy_mode: str = "copy",
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Builds the unified dataset from raw image folders with zero synthetic noise:
    1. Consolidates all images into a single '<output_dir>/images/' folder without collisions.
    2. Writes '<output_dir>/manifest.json' (without redundant clean_file).
    3. Writes '<output_dir>/ground_truth.csv'.
    4. Writes '<output_dir>/labels/' (YOLO format, all plates preserved).
    5. Writes '<output_dir>/annotations/' (Pascal VOC format matching image filenames).
    6. Writes '<output_dir>/dataset_summary.json'.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"[FATAL] Input directory does not exist: {in_path.resolve()}")

    images_dest_dir = out_path / "images"
    labels_dest_dir = out_path / "labels"
    annotations_dest_dir = out_path / "annotations"

    images_dest_dir.mkdir(parents=True, exist_ok=True)
    labels_dest_dir.mkdir(parents=True, exist_ok=True)
    annotations_dest_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_all_dataset_pairs(in_path)
    if limit is not None and limit > 0:
        print(f"[*] Applying sample limit: {limit} of {len(pairs)} pairs.")
        pairs = pairs[:limit]

    manifest = []
    csv_rows = []
    char_lengths = Counter()
    resolution_records = []
    skipped_count = 0
    multi_plate_count = 0
    collision_count = 0

    seen_target_stems = set()
    total_pairs = len(pairs)
    print(f"\n[*] Processing and copying {total_pairs} images to unified repository '{images_dest_dir}'...")

    for idx, (img_path, xml_path, source_folder) in enumerate(pairs):
        parsed = parse_voc_annotation(xml_path, img_path)
        if parsed is None:
            skipped_count += 1
            continue

        plates = parsed["plates"]
        img_w = parsed["image_width"]
        img_h = parsed["image_height"]
        primary_plate = plates[0]

        if len(plates) > 1:
            multi_plate_count += 1

        # -------------------------------------------------------------
        # Collision-Free Filename Resolution
        # -------------------------------------------------------------
        base_stem = img_path.stem
        ext = img_path.suffix.lower()

        target_stem = base_stem
        if target_stem in seen_target_stems:
            collision_count += 1
            clean_prefix = re.sub(r'[^a-zA-Z0-9_]', '_', source_folder).strip('_')
            target_stem = f"{clean_prefix}_{base_stem}"
            counter = 1
            while target_stem in seen_target_stems:
                target_stem = f"{clean_prefix}_{base_stem}_{counter}"
                counter += 1

        seen_target_stems.add(target_stem)
        target_filename = f"{target_stem}{ext}"

        dest_img_path = images_dest_dir / target_filename
        dest_label_path = labels_dest_dir / f"{target_stem}.txt"
        dest_xml_path = annotations_dest_dir / f"{target_stem}.xml"

        # Copy or symlink image into unified images/ directory
        if not dest_img_path.exists():
            if copy_mode == "symlink":
                try:
                    os.symlink(img_path.resolve(), dest_img_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dest_img_path)
            else:
                shutil.copy2(img_path, dest_img_path)

        # Validate that the destination image exists on disk
        if not dest_img_path.is_file():
            print(f"[!] Warning: Failed to copy or create file: {dest_img_path}. Skipping.")
            skipped_count += 1
            continue

        # Write YOLO multi-plate label: one line per plate
        with open(dest_label_path, "w", encoding="utf-8") as f_lbl:
            for p in plates:
                yb = p["yolo_box"]
                f_lbl.write(f"0 {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}\n")

        # Copy and synchronize VOC XML annotation with exact matching filename
        try:
            tree = ET.parse(xml_path)
            root_elem = tree.getroot()
            fn_elem = root_elem.find("filename")
            if fn_elem is not None:
                fn_elem.text = target_filename
            else:
                fn_elem = ET.SubElement(root_elem, "filename")
                fn_elem.text = target_filename
            tree.write(dest_xml_path, encoding="utf-8", xml_declaration=True)
        except Exception:
            shutil.copy2(xml_path, dest_xml_path)

        rel_img_path = str(dest_img_path).replace("\\", "/")

        manifest_item = {
            "sample_id": idx,
            "filename": target_filename,
            "image_path": rel_img_path,
            "ground_truth_text": primary_plate["ground_truth_text"],
            "gt_bbox": primary_plate["gt_bbox"],
            "all_plates": plates,
            "num_plates": len(plates),
            "image_width": img_w,
            "image_height": img_h
        }
        manifest.append(manifest_item)

        csv_rows.append({
            "sample_id": idx,
            "filename": target_filename,
            "image_path": rel_img_path,
            "ground_truth_text": primary_plate["ground_truth_text"],
            "xmin": primary_plate["gt_bbox"][0],
            "ymin": primary_plate["gt_bbox"][1],
            "xmax": primary_plate["gt_bbox"][2],
            "ymax": primary_plate["gt_bbox"][3],
            "num_plates": len(plates),
            "width": img_w,
            "height": img_h
        })

        char_lengths[len(primary_plate["ground_truth_text"])] += 1
        resolution_records.append((img_w, img_h))

        if (idx + 1) % 250 == 0 or (idx + 1) == total_pairs:
            print(f"  -> Processed {idx + 1}/{total_pairs} images ({(idx + 1) / total_pairs * 100:.1f}%)")

    # 1. Export manifest.json
    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. Export ground_truth.csv
    csv_file = out_path / "ground_truth.csv"
    csv_headers = [
        "sample_id", "filename", "image_path", "ground_truth_text",
        "xmin", "ymin", "xmax", "ymax", "num_plates", "width", "height"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    # 3. Export dataset_summary.json
    widths = [w for w, h in resolution_records]
    heights = [h for w, h in resolution_records]
    summary_data = {
        "dataset_name": "Indian License Plate Real-World Forensic Dataset",
        "total_images": len(manifest),
        "skipped_images": skipped_count,
        "multi_plate_images": multi_plate_count,
        "filename_collisions_resolved": collision_count,
        "unified_images_folder": str(images_dest_dir.resolve()),
        "manifest_file": str(manifest_file.resolve()),
        "ground_truth_csv": str(csv_file.resolve()),
        "labels_yolo_dir": str(labels_dest_dir.resolve()),
        "annotations_xml_dir": str(annotations_dest_dir.resolve()),
        "character_length_distribution": {f"{k}_chars": v for k, v in sorted(char_lengths.items())},
        "resolution_statistics": {
            "min_width": int(min(widths)) if widths else 0,
            "max_width": int(max(widths)) if widths else 0,
            "mean_width": round(sum(widths) / len(widths), 1) if widths else 0,
            "min_height": int(min(heights)) if heights else 0,
            "max_height": int(max(heights)) if heights else 0,
            "mean_height": round(sum(heights) / len(heights), 1) if heights else 0
        }
    }

    summary_file = out_path / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f" Total Unified Real Images : {len(manifest)}")
    print(f" Multi-Plate Images Saved  : {multi_plate_count}")
    print(f" Collisions Disambiguated  : {collision_count}")
    print(f" Consolidated Images Dir   : {images_dest_dir.resolve()}")
    print(f" Manifest JSON Path        : {manifest_file.resolve()}")
    print(f" Tabular CSV Path          : {csv_file.resolve()}")
    print(f" YOLO Labels Directory     : {labels_dest_dir.resolve()}")
    print(f" VOC Annotations Directory : {annotations_dest_dir.resolve()}")
    print(f" Dataset Summary JSON      : {summary_file.resolve()}")
    print("=" * 70 + "\n")

    return summary_data


# =========================================================================
# HARDCODED CONFIGURATION & PATHS (NO CLI ARGUMENTS)
# =========================================================================
DATA_INPUT_DIR = "./data/images"
DATA_OUTPUT_DIR = "./data/dataset_prepared"
DATA_COPY_MODE = "copy"
DATA_LIMIT = None


def main():
    print("=" * 70)
    print("UNIFIED DATASET PREPARATION PIPELINE")
    print("=" * 70)
    print(f"[*] Input Images Directory : {os.path.abspath(DATA_INPUT_DIR)}")
    print(f"[*] Output Dataset Directory: {os.path.abspath(DATA_OUTPUT_DIR)}")
    print(f"[*] Copy Mode              : {DATA_COPY_MODE}")
    print(f"[*] Sample Limit           : {'All' if DATA_LIMIT is None else DATA_LIMIT}")
    print("=" * 70 + "\n")

    prepare_unified_dataset(
        input_dir=DATA_INPUT_DIR,
        output_dir=DATA_OUTPUT_DIR,
        copy_mode=DATA_COPY_MODE,
        limit=DATA_LIMIT
    )


if __name__ == "__main__":
    main()
