"""
Generate ground truth visualization images for segmentation and HF detection.

For each image stem in IMAGE_STEMS, produces three separate PNG files
with NO borders, titles, or padding — raw pixel output only:
  {stem}_original.png  — original OCT image
  {stem}_seg_gt.png    — pure coloured mask on black background
  {stem}_det_gt.png    — original OCT + alpha-blend mask overlay + cyan HF boxes

Colour scheme:
  Background : black  (#000000)
  SRF        : #000066 dark navy blue
  IRF        : #660000 dark red
  ELM        : #666600 olive
  EZ         : #006600 dark green
  HF boxes   : cyan

Output directory: backend/evaluation/outputs/ground_truth/
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]
IMAGES_DIR  = REPO_ROOT / "dataset" / "RVO-Lesion" / "Image_Seg" / "images"
MASKS_DIR   = REPO_ROOT / "dataset" / "RVO-Lesion" / "Image_Seg" / "masks"
LABELME_DIR = REPO_ROOT / "dataset" / "RVO-Lesion" / "RVO_Lesion_Labelme"
OUTPUT_DIR  = REPO_ROOT / "backend" / "evaluation" / "outputs" / "ground_truth"

# ---------------------------------------------------------------------------
IMAGE_STEMS = ["1_1", "2_15", "5_12", "6_1"]

# Segmentation colours (RGB)
SEG_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (0,   0,   0),
    1: (0,   0,   102),   # SRF
    2: (102, 0,   0),     # IRF
    3: (102, 102, 0),     # ELM
    4: (0,   102, 0),     # EZ
}

OVERLAY_ALPHA = 0.45
HF_BOX_HALF   = 5
HF_COLOR_BGR  = (255, 255, 0)   # cyan in BGR
HF_THICKNESS  = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_png(rgb: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {path.name}")


def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask


def load_hf_points(stem: str) -> list[tuple[float, float]]:
    patient_id = stem.split("_")[0]
    json_path  = LABELME_DIR / patient_id / f"{stem}.json"
    if not json_path.exists():
        print(f"  Warning: LabelMe JSON not found: {json_path}")
        return []
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    points: list[tuple[float, float]] = []
    for shape in data.get("shapes", []):
        if shape.get("label", "").upper() != "HF":
            continue
        pts   = shape.get("points", [])
        stype = shape.get("shape_type", "")
        if stype == "point" and pts:
            points.append((float(pts[0][0]), float(pts[0][1])))
        elif stype in ("rectangle", "polygon", "linestrip") and len(pts) >= 2:
            points.append((float(np.mean([p[0] for p in pts])),
                           float(np.mean([p[1] for p in pts]))))
    return points


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in SEG_COLORS_RGB.items():
        rgb[mask == cls_id] = color
    return rgb


def alpha_blend(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colored = mask_to_rgb(mask).astype(float)
    blended = (1.0 - OVERLAY_ALPHA) * image_rgb.astype(float) + OVERLAY_ALPHA * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_hf_boxes(image_rgb: np.ndarray, hf_points: list[tuple[float, float]]) -> np.ndarray:
    h, w    = image_rgb.shape[:2]
    out_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for px, py in hf_points:
        x1 = max(0, int(px) - HF_BOX_HALF)
        y1 = max(0, int(py) - HF_BOX_HALF)
        x2 = min(w - 1, int(px) + HF_BOX_HALF)
        y2 = min(h - 1, int(py) + HF_BOX_HALF)
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), HF_COLOR_BGR, HF_THICKNESS)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    for stem in IMAGE_STEMS:
        print(f"Processing {stem} ...")
        img_path  = IMAGES_DIR / f"{stem}.jpg"
        mask_path = MASKS_DIR  / f"{stem}.png"

        try:
            image = load_image_rgb(img_path)
            mask  = load_mask(mask_path)
        except FileNotFoundError as exc:
            print(f"  Error: {exc}\n")
            continue

        hf_points = load_hf_points(stem)
        print(f"  {image.shape[1]}×{image.shape[0]}  |  "
              f"classes: {np.unique(mask).tolist()}  |  HF: {len(hf_points)}")

        # Original
        save_png(image, OUTPUT_DIR / f"{stem}_original.png")

        # Segmentation GT — pure coloured mask on black
        save_png(mask_to_rgb(mask), OUTPUT_DIR / f"{stem}_seg_gt.png")

        # Detection GT — original + seg overlay + HF boxes
        blended = alpha_blend(image, mask)
        save_png(draw_hf_boxes(blended, hf_points), OUTPUT_DIR / f"{stem}_det_gt.png")

        print()

    print(f"Done. {len(IMAGE_STEMS) * 3} images saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
