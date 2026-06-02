"""
LabelMe → COCO conversion for Hyperreflective Foci (HF) detection.

Traverses all numbered subdirectories inside RVO_Lesion_Labelme/, extracts
point annotations labelled "HF", converts each point to a 10×10 bounding box
(clipped to image bounds), and writes a single COCO annotations.json together
with a flat images/ directory.

Usage:
    python convert_labelme_to_coco.py \
        --labelme_root /path/to/RVO_Lesion_Labelme \
        --output_dir  /path/to/coco_output

The script is idempotent: re-running with the same output_dir will overwrite
the existing annotations.json and re-copy images.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2


# ─── Constants ────────────────────────────────────────────────────────────────

HF_BOX_HALF: int = 5          # half of the 10×10 box → ±5 px from centre point
HF_LABEL: str   = "HF"        # LabelMe label name (case-insensitive comparison)
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    """Return parsed JSON dict, or None if the file is missing / corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Could not read {path}: {exc}")
        return None


def _resolve_image_path(json_path: Path, labelme_data: dict) -> Path | None:
    """
    Find the image file that corresponds to a LabelMe JSON.

    LabelMe stores the relative path in data['imagePath'].  We also try
    every supported extension with the same stem as the JSON, in case the
    path is missing or wrong.
    """
    # 1. Try the path recorded inside the JSON
    recorded = labelme_data.get("imagePath", "")
    if recorded:
        candidate = (json_path.parent / Path(recorded).name).resolve()
        if candidate.exists():
            return candidate

    # 2. Fall back to same stem as JSON
    for ext in SUPPORTED_IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


def _point_to_coco_bbox(
    px: float, py: float,
    img_w: int, img_h: int,
    half: int = HF_BOX_HALF,
) -> list[float] | None:
    """
    Convert a point annotation to a COCO bounding box [x, y, w, h].

    Returns None when the box is degenerate (zero area after clipping).
    """
    x1 = max(0.0, px - half)
    y1 = max(0.0, py - half)
    x2 = min(float(img_w), px + half)
    y2 = min(float(img_h), py + half)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return [x1, y1, w, h]


def _polygon_to_coco_bbox(
    points: list[list[float]],
    img_w: int, img_h: int,
) -> list[float] | None:
    """Compute axis-aligned bounding box from a list of polygon vertices."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(img_w), max(xs))
    y2 = min(float(img_h), max(ys))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return [x1, y1, w, h]


# ─── Core conversion ──────────────────────────────────────────────────────────

def convert_labelme_to_coco(
    labelme_root: str | Path,
    output_dir:   str | Path,
    hf_label:     str = HF_LABEL,
    box_half:     int = HF_BOX_HALF,
    copy_images:  bool = True,
) -> dict:
    """
    Convert LabelMe annotations to COCO format.

    Args:
        labelme_root: Root directory containing numbered subfolders with
                      image + JSON pairs (e.g. RVO_Lesion_Labelme/).
        output_dir:   Destination directory.  Will contain:
                        annotations.json
                        images/           (flat, prefixed with image_id)
        hf_label:     LabelMe label to treat as HF (default "HF").
        box_half:     Half-size for point→bbox conversion (default 5 → 10×10).
        copy_images:  If False, skip copying images (useful for Kaggle where
                      images are already on a separate mount).

    Returns:
        The COCO dict (also written to output_dir/annotations.json).
    """
    labelme_root = Path(labelme_root)
    output_dir   = Path(output_dir)
    images_dir   = output_dir / "images"
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    coco: dict = {
        "info": {
            "description": "RVO HF Detection Dataset – converted from LabelMe",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "HF", "supercategory": "retinal_lesion"}],
        "images": [],
        "annotations": [],
    }

    # Collect every JSON under the root (depth-first, sorted for reproducibility)
    json_paths = sorted(labelme_root.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found under {labelme_root}")
    print(f"Found {len(json_paths)} JSON file(s) — starting conversion…")

    image_id = 0
    ann_id   = 0
    skipped  = 0

    for json_path in json_paths:
        data = _load_json(json_path)
        if data is None:
            skipped += 1
            continue

        image_path = _resolve_image_path(json_path, data)
        if image_path is None:
            print(f"  [WARN] No image found for {json_path.name} — skipping")
            skipped += 1
            continue

        # Read image dimensions (avoid loading full pixels into RAM for large sets)
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [WARN] OpenCV could not read {image_path} — skipping")
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]
        del img  # release memory

        # Destination filename: zero-padded id + original filename
        dest_name = f"{image_id:06d}_{image_path.name}"
        if copy_images:
            shutil.copy2(str(image_path), str(images_dir / dest_name))

        coco["images"].append({
            "id":        image_id,
            "file_name": dest_name,
            "width":     img_w,
            "height":    img_h,
        })

        # ── Extract HF annotations ──────────────────────────────────────────
        hf_count = 0
        for shape in data.get("shapes", []):
            if shape.get("label", "").upper() != hf_label.upper():
                continue

            shape_type = shape.get("shape_type", "").lower()
            pts        = shape.get("points", [])

            if shape_type == "point" and len(pts) >= 1:
                px, py = float(pts[0][0]), float(pts[0][1])
                bbox   = _point_to_coco_bbox(px, py, img_w, img_h, half=box_half)

            elif shape_type in ("rectangle", "polygon", "linestrip") and len(pts) >= 2:
                bbox = _polygon_to_coco_bbox(pts, img_w, img_h)

            else:
                print(f"  [WARN] Unknown shape_type '{shape_type}' in {json_path.name}")
                continue

            if bbox is None:
                print(f"  [WARN] Degenerate bbox for HF in {json_path.name} — skipped")
                continue

            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        bbox,
                "area":        bbox[2] * bbox[3],
                "iscrowd":     0,
            })
            ann_id   += 1
            hf_count += 1

        image_id += 1

    # ── Write annotations.json ──────────────────────────────────────────────
    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)

    print(
        f"\nConversion complete:\n"
        f"  Total JSON files : {len(json_paths)}\n"
        f"  Skipped          : {skipped}\n"
        f"  Images written   : {len(coco['images'])}\n"
        f"  HF annotations   : {len(coco['annotations'])}\n"
        f"  Saved to         : {ann_path}"
    )
    return coco


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert LabelMe point annotations to COCO format for HF detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--labelme_root", required=True,
                   help="Path to RVO_Lesion_Labelme directory")
    p.add_argument("--output_dir",   required=True,
                   help="Destination directory for annotations.json + images/")
    p.add_argument("--hf_label",     default=HF_LABEL,
                   help="LabelMe label name for HF annotations")
    p.add_argument("--box_half",     type=int, default=HF_BOX_HALF,
                   help="Half-width of the point-to-bbox box (default 5 → 10×10 px)")
    p.add_argument("--no_copy",      action="store_true",
                   help="Skip copying images to output_dir/images/")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_labelme_to_coco(
        labelme_root=args.labelme_root,
        output_dir=args.output_dir,
        hf_label=args.hf_label,
        box_half=args.box_half,
        copy_images=not args.no_copy,
    )
