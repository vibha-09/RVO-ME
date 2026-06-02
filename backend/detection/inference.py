"""
Plug-and-play HF (Hyperreflective Foci) inference module.

Loads a Faster R-CNN model trained by train.py and exposes a clean API for
the rest of the backend.  This module is intentionally self-contained: it
imports only from dataset.py (for preprocessing) and standard libraries.

KAGGLE → LOCAL CONSISTENCY GUARANTEE
──────────────────────────────────────
1. Architecture is rebuilt from hf_frcnn_config.json (same config used during
   training). If the JSON is absent the same DEFAULT_CONFIG from train.py is
   used as a fallback.
2. Preprocessing calls preprocess_image() from dataset.py – the SAME function
   called inside HFDetectionDataset.__getitem__ on Kaggle.
3. The model's internal GeneralizedRCNNTransform applies ImageNet normalisation
   and resizes images using the same min_size / max_size parameters stored in
   the config.  No external resize or normalise is needed here.

Usage – standalone:
    detector = HFDetector("weights/hf_frcnn.pt")
    boxes, scores = detector.predict(image_bgr)
    vis = detector.visualize(image_bgr, boxes, scores)

Usage – as a drop-in for backend/app/services/detection.py:
    from detection.inference import HFDetector
    _detector = HFDetector("weights/hf_frcnn.pt")
    result = _detector.predict(enhanced_grayscale_image)
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from dataset import preprocess_image


# ─── Fallback config (must mirror DEFAULT_CONFIG in train.py) ─────────────────
_FALLBACK_CONFIG: dict = {
    "num_classes":          2,
    "image_min_size":       800,
    "image_max_size":       1333,
    "anchor_sizes":         [[8], [16], [32], [64], [128]],
    "anchor_aspect_ratios": [[0.5, 1.0, 2.0]] * 5,
    "score_threshold":      0.5,
}


# ─── Architecture builder (must be identical to train.build_model) ───────────

def _build_model(config: dict) -> torch.nn.Module:
    anchor_gen = AnchorGenerator(
        sizes=tuple(tuple(s) for s in config["anchor_sizes"]),
        aspect_ratios=tuple(tuple(r) for r in config["anchor_aspect_ratios"]),
    )
    try:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            min_size=config["image_min_size"],
            max_size=config["image_max_size"],
            rpn_anchor_generator=anchor_gen,
        )
    except TypeError:
        model = fasterrcnn_resnet50_fpn(  # type: ignore[call-arg]
            pretrained=False,
            min_size=config["image_min_size"],
            max_size=config["image_max_size"],
            rpn_anchor_generator=anchor_gen,
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])
    return model


# ─── Detector ────────────────────────────────────────────────────────────────

class HFDetector:
    """
    Ready-to-use wrapper around the trained Faster R-CNN.

    The class accepts either a BGR colour image (the normal OpenCV format) or
    a grayscale image (the output of apply_clahe, as used in the live backend
    pipeline).  Both inputs are handled transparently via preprocess_image().

    Attributes:
        score_threshold: Default confidence cut-off; can be overridden per call.
    """

    def __init__(
        self,
        weights_path:    str | Path,
        config_path:     Optional[str | Path] = None,
        device:          Optional[str | torch.device] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            weights_path:    Path to hf_frcnn.pt (output of train.py).
            config_path:     Path to hf_frcnn_config.json.  Auto-detected in
                             the same directory as weights_path when omitted.
            device:          'cuda', 'cpu', or torch.device.  Auto-selected.
            score_threshold: Override the threshold stored in config.
        """
        weights_path = Path(weights_path)

        # Auto-detect config
        if config_path is None:
            auto = weights_path.parent / "hf_frcnn_config.json"
            config_path = auto if auto.exists() else None

        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r") as fh:
                self.config: dict = json.load(fh)
            print(f"[HFDetector] Config loaded from {config_path}")
        else:
            self.config = dict(_FALLBACK_CONFIG)
            print("[HFDetector] Warning: config not found – using fallback defaults")

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model and load weights
        self.model = _build_model(self.config).to(self.device)
        state = torch.load(str(weights_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.score_threshold: float = (
            score_threshold
            if score_threshold is not None
            else self.config.get("score_threshold", 0.5)
        )
        print(f"[HFDetector] Ready on {self.device} | threshold={self.score_threshold}")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect HF in a single image.

        Args:
            image:           BGR uint8 array (OpenCV) OR grayscale uint8 array.
                             The preprocessing pipeline handles both.
            score_threshold: Per-call override; uses self.score_threshold when None.

        Returns:
            boxes:  (N, 4) float32 array of [x1, y1, x2, y2] bounding boxes.
            scores: (N,)   float32 array of confidence scores.
            Both arrays have the same first dimension N (possibly 0).
        """
        threshold = score_threshold if score_threshold is not None \
                    else self.score_threshold

        # If grayscale, convert to synthetic BGR so preprocess_image receives BGR
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            gray  = image if image.ndim == 2 else image[:, :, 0]
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        tensor  = preprocess_image(image).to(self.device)
        outputs = self.model([tensor])[0]

        boxes  = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        keep = scores >= threshold
        return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualize(
        self,
        image_bgr: np.ndarray,
        boxes:     np.ndarray,
        scores:    np.ndarray,
        color:     tuple[int, int, int] = (0, 255, 255),  # cyan
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detection boxes and scores onto a BGR image.

        Args:
            image_bgr: Original BGR uint8 image.
            boxes:     (N, 4) float32 [x1, y1, x2, y2].
            scores:    (N,)   float32.
            color:     Box / label background colour (BGR).
            thickness: Rectangle line thickness.

        Returns:
            Annotated BGR uint8 image (copy of input).
        """
        vis = image_bgr.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            label       = f"HF {score:.2f}"
            font        = cv2.FONT_HERSHEY_SIMPLEX
            font_scale  = 0.4
            font_thick  = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            # Filled label background
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(vis, label, (x1 + 1, y1 - 2), font, font_scale,
                        (0, 0, 0), font_thick, cv2.LINE_AA)
        return vis


# ─── Standalone CLI ───────────────────────────────────────────────────────────

def run_inference(
    image_path:      str | Path,
    weights_path:    str | Path,
    config_path:     Optional[str | Path] = None,
    output_path:     Optional[str | Path] = None,
    score_threshold: Optional[float]      = None,
    show:            bool                 = False,
) -> dict:
    """
    Run inference on a single image and optionally save the visualisation.

    Returns a dict with keys: boxes, scores, num_detections, image_path.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    detector       = HFDetector(weights_path, config_path=config_path,
                                score_threshold=score_threshold)
    boxes, scores  = detector.predict(image)

    print(f"Detections: {len(boxes)}")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  [{i+1}] box={b.astype(int).tolist()}  score={s:.4f}")

    vis = detector.visualize(image, boxes, scores)

    if output_path:
        cv2.imwrite(str(output_path), vis)
        print(f"Saved: {output_path}")

    if show:
        cv2.imshow("HF Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "boxes":          boxes.tolist(),
        "scores":         scores.tolist(),
        "num_detections": int(len(boxes)),
        "image_path":     str(image_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HF detection on a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",     required=True, help="Path to input OCT image")
    parser.add_argument("--weights",   required=True, help="Path to hf_frcnn.pt")
    parser.add_argument("--config",    default=None,  help="Path to hf_frcnn_config.json")
    parser.add_argument("--output",    default=None,  help="Path to save annotated image")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold")
    parser.add_argument("--show",      action="store_true")
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        weights_path=args.weights,
        config_path=args.config,
        output_path=args.output,
        score_threshold=args.threshold,
        show=args.show,
    )
