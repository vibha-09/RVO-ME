"""
HF detection service – wraps the trained Faster R-CNN model.

PREPROCESSING PIPELINE (must stay in sync with backend/detection/dataset.py):
  enhanced_image (grayscale uint8 from apply_clahe)
      → stack × 3  →  / 255.0  →  permute CHW  →  float32 tensor [0, 1]
  The model's internal GeneralizedRCNNTransform then handles ImageNet
  normalisation and resizing to [min_size, max_size].

Weight loading priority:
  1. weights/hf_frcnn.pt   (trained model from backend/detection/train.py)
  2. weights/faster_rcnn.pth  (legacy fallback)
  When neither exists the model runs in stub / demo mode.
"""

import os

import cv2
import numpy as np
import torch

from ..models.detector import get_detection_model

# ─── Globals ─────────────────────────────────────────────────────────────────

_model: torch.nn.Module | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_WEIGHTS_PRIMARY  = "weights/hf_frcnn.pt"
_WEIGHTS_FALLBACK = "weights/faster_rcnn.pth"
_STUB_MAX_BOXES   = 5   # cap random-init predictions to keep the UI usable


# ─── Model loading ────────────────────────────────────────────────────────────

def load_detection_model(weights_path: str | None = None) -> None:
    """
    Build the Faster R-CNN and load trained weights.

    Searches for weights in this order:
      1. weights_path argument (if given)
      2. weights/hf_frcnn.pt  (trained model)
      3. weights/faster_rcnn.pth (legacy)
    Falls back to randomly initialised weights (stub mode) when none found.
    """
    global _model
    _model = get_detection_model(num_classes=2).to(_device)

    candidates = [p for p in [weights_path, _WEIGHTS_PRIMARY, _WEIGHTS_FALLBACK] if p]
    for path in candidates:
        if os.path.exists(path):
            _model.load_state_dict(torch.load(path, map_location=_device))
            print(f"[detection] Loaded weights from {path}")
            _model.eval()
            return

    print(
        f"[detection] Warning: no weights found "
        f"({_WEIGHTS_PRIMARY}, {_WEIGHTS_FALLBACK}). "
        "Running in stub mode with random initialisation."
    )
    _model.eval()


# ─── Preprocessing (canonical – must match detection/dataset.preprocess_image) ─

def _preprocess(enhanced_image: np.ndarray) -> torch.Tensor:
    """
    Convert a grayscale (or 3-channel) uint8 image to a CHW float tensor [0,1].

    This replicates the exact pipeline in dataset.preprocess_image() called
    during training.  Do NOT alter this function without retraining.
    """
    if enhanced_image.ndim == 2 or (enhanced_image.ndim == 3 and enhanced_image.shape[2] == 1):
        gray = enhanced_image if enhanced_image.ndim == 2 else enhanced_image[:, :, 0]
        img_3c = np.stack([gray] * 3, axis=-1)          # (H, W, 3) uint8
    else:
        # Colour input: convert BGR → grayscale → stack (rare, but handled)
        gray   = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        img_3c = np.stack([gray] * 3, axis=-1)

    img_f32 = img_3c.astype(np.float32) / 255.0         # [0, 1] float
    tensor  = torch.from_numpy(img_f32).permute(2, 0, 1) # CHW
    return tensor


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_detections(
    enhanced_image: np.ndarray,
    threshold:      float = 0.5,
) -> dict:
    """
    Detect Hyperreflective Foci in an OCT image.

    Args:
        enhanced_image: Grayscale uint8 output of apply_clahe().
        threshold:      Minimum confidence score to keep a detection.

    Returns:
        dict with keys:
            boxes  – (N, 4) float32 array  [x1, y1, x2, y2]
            scores – (N,)   float32 array
            labels – (N,)   int64 array (all 1 = HF)
    """
    global _model
    if _model is None:
        load_detection_model()

    tensor = _preprocess(enhanced_image).unsqueeze(0).to(_device)

    with torch.no_grad():
        predictions = _model(tensor)

    pred   = predictions[0]
    boxes  = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    keep           = scores >= threshold
    filtered_boxes  = boxes[keep]
    filtered_scores = scores[keep]
    filtered_labels = labels[keep]

    # Cap stub-mode spam: random-init models produce many low-quality boxes
    is_stub = not any(os.path.exists(p) for p in [_WEIGHTS_PRIMARY, _WEIGHTS_FALLBACK])
    if is_stub and len(filtered_boxes) > _STUB_MAX_BOXES:
        filtered_boxes  = filtered_boxes[:_STUB_MAX_BOXES]
        filtered_scores = filtered_scores[:_STUB_MAX_BOXES]
        filtered_labels = filtered_labels[:_STUB_MAX_BOXES]

    return {
        "boxes":  filtered_boxes,
        "scores": filtered_scores,
        "labels": filtered_labels,
    }
