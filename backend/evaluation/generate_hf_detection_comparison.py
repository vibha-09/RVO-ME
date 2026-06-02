"""
Generate HF detection comparison images: ground truth vs model predictions.

Uses the EXACT same model architecture, preprocessing, and inference path
as the website (backend/app/services/detection.py) so predictions match.

Saves raw pixel images with NO borders, titles, or padding.

For each image stem in IMAGE_STEMS, produces two PNG files saved to
backend/evaluation/outputs/ground_truth/:

  {stem}_hf_gt.png   — original OCT + ground-truth HF boxes  (cyan, bold)
  {stem}_hf_pred.png — original OCT + model-predicted HF boxes (cyan, bold)

Run from the repo root:
  python backend/evaluation/generate_hf_detection_comparison.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
IMAGES_DIR  = REPO_ROOT / "dataset" / "RVO-Lesion" / "Image_Seg" / "images"
LABELME_DIR = REPO_ROOT / "dataset" / "RVO-Lesion" / "RVO_Lesion_Labelme"
WEIGHTS_DIR = BACKEND_DIR / "weights"
OUTPUT_DIR  = REPO_ROOT / "backend" / "evaluation" / "outputs" / "ground_truth"

WEIGHTS_PATH = WEIGHTS_DIR / "hf_frcnn.pt"

# Add backend to sys.path so we can import from the app package
sys.path.insert(0, str(BACKEND_DIR))

from app.models.detector import get_detection_model
from app.services.preprocessor import apply_clahe

# ---------------------------------------------------------------------------
IMAGE_STEMS     = ["1_1", "2_15", "5_12", "6_1"]
HF_BOX_HALF     = 5
HF_COLOR_BGR    = (255, 255, 0)   # cyan in BGR
HF_THICKNESS    = 2
SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Detection model — identical to backend/app/services/detection.py
# ---------------------------------------------------------------------------

def _preprocess(enhanced_image: np.ndarray) -> torch.Tensor:
    """Exact copy of _preprocess() from app/services/detection.py."""
    if enhanced_image.ndim == 2 or (enhanced_image.ndim == 3 and enhanced_image.shape[2] == 1):
        gray = enhanced_image if enhanced_image.ndim == 2 else enhanced_image[:, :, 0]
        img_3c = np.stack([gray] * 3, axis=-1)
    else:
        gray   = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        img_3c = np.stack([gray] * 3, axis=-1)
    img_f32 = img_3c.astype(np.float32) / 255.0
    return torch.from_numpy(img_f32).permute(2, 0, 1)


def load_detector(device: torch.device) -> torch.nn.Module:
    model = get_detection_model(num_classes=2).to(device)
    for path in [str(WEIGHTS_PATH), str(WEIGHTS_DIR / "faster_rcnn.pth")]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"  Weights loaded: {Path(path).name}")
            break
    else:
        print("  Warning: no weights found — stub mode")
    model.eval()
    return model


@torch.no_grad()
def predict_boxes(
    model: torch.nn.Module,
    enhanced_gray: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Returns (N, 4) float32 array of [x1, y1, x2, y2] boxes."""
    tensor      = _preprocess(enhanced_gray).unsqueeze(0).to(device)
    predictions = model(tensor)
    pred        = predictions[0]
    boxes       = pred["boxes"].cpu().numpy()
    scores      = pred["scores"].cpu().numpy()
    keep        = scores >= SCORE_THRESHOLD
    return boxes[keep].astype(np.float32)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


# ---------------------------------------------------------------------------
# Visualisation — raw pixel output, no matplotlib
# ---------------------------------------------------------------------------

def draw_gt_boxes(image_rgb: np.ndarray, hf_points: list[tuple[float, float]]) -> np.ndarray:
    h, w    = image_rgb.shape[:2]
    out_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for px, py in hf_points:
        x1 = max(0, int(px) - HF_BOX_HALF)
        y1 = max(0, int(py) - HF_BOX_HALF)
        x2 = min(w - 1, int(px) + HF_BOX_HALF)
        y2 = min(h - 1, int(py) + HF_BOX_HALF)
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), HF_COLOR_BGR, HF_THICKNESS)
    return out_bgr


def draw_pred_boxes(image_rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), HF_COLOR_BGR, HF_THICKNESS)
    return out_bgr


def save_bgr(bgr: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), bgr)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading detection model (device={device}) ...")
    model = load_detector(device)
    print()

    for stem in IMAGE_STEMS:
        print(f"Processing {stem} ...")
        try:
            image = load_image_rgb(IMAGES_DIR / f"{stem}.jpg")
        except FileNotFoundError as exc:
            print(f"  Error: {exc}\n")
            continue

        enhanced = apply_clahe(image)

        # Ground truth
        hf_points = load_hf_points(stem)
        save_bgr(draw_gt_boxes(image, hf_points),
                 OUTPUT_DIR / f"{stem}_hf_gt.png")

        # Model prediction
        pred_boxes = predict_boxes(model, enhanced, device)
        save_bgr(draw_pred_boxes(image, pred_boxes),
                 OUTPUT_DIR / f"{stem}_hf_pred.png")

        print(f"  GT: {len(hf_points)}  |  Predicted: {len(pred_boxes)}")
        print()

    print(f"Done. {len(IMAGE_STEMS) * 2} images saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
