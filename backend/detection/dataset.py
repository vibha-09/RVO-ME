"""
PyTorch Dataset and preprocessing utilities for HF detection.

PREPROCESSING CONTRACT
──────────────────────
This module defines the canonical preprocessing pipeline that MUST be
identical between training and inference.  The same `preprocess_image()`
function is imported by inference.py, and the same logic is reproduced
inside backend/app/services/detection.py.

Pipeline (mirrors the live backend exactly):
  1. BGR uint8  →  RGB uint8           (cv2.cvtColor)
  2. RGB uint8  →  grayscale uint8     (apply_clahe – identical to preprocessor.py)
  3. grayscale  →  3-channel uint8     (np.stack × 3)
  4. uint8      →  float32 [0, 1]      (/ 255.0)
  5. (H, W, 3)  →  (3, H, W) tensor   (torch.from_numpy + permute)

The model's internal GeneralizedRCNNTransform then handles ImageNet
normalisation and image resizing – no manual resize or normalise needed here.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ─── CLAHE (copied from backend/app/services/preprocessor.py) ───────────────
# Kept inline so this module works standalone on Kaggle without the app package.

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation → grayscale uint8."""
    if image.ndim == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


# ─── Canonical preprocessing ─────────────────────────────────────────────────

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR uint8 numpy image into a float32 CHW tensor in [0, 1].

    This function is the single source of truth for preprocessing.
    It is called in HFDetectionDataset.__getitem__ and in inference.py.

    Args:
        image_bgr: (H, W, 3) uint8 array, BGR channel order (OpenCV default).

    Returns:
        (3, H, W) float32 tensor with values in [0, 1].
    """
    # Step 1-2: BGR → RGB → CLAHE grayscale
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    enhanced  = apply_clahe(image_rgb)                         # (H, W) uint8

    # Step 3-5: stack → float → CHW tensor
    img_3c   = np.stack([enhanced] * 3, axis=-1)               # (H, W, 3) uint8
    img_f32  = img_3c.astype(np.float32) / 255.0               # (H, W, 3) float [0,1]
    tensor   = torch.from_numpy(img_f32).permute(2, 0, 1)      # (3, H, W) float [0,1]
    return tensor


# ─── Augmentation ────────────────────────────────────────────────────────────

def _hflip(tensor: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip of image tensor and corresponding boxes."""
    W = tensor.shape[-1]
    tensor = tensor.flip(-1)
    if boxes.shape[0] > 0:
        flipped = boxes.clone()
        flipped[:, 0] = W - boxes[:, 2]
        flipped[:, 2] = W - boxes[:, 0]
        boxes = flipped
    return tensor, boxes


def get_transform(train: bool) -> Callable:
    """
    Returns a callable  (image_bgr_ndarray, target_dict) → (tensor, target_dict).

    During training a random horizontal flip is applied with p=0.5.
    During inference / validation no augmentation is applied.
    Both paths call preprocess_image(), guaranteeing identical normalisation.
    """
    def transform(image_bgr: np.ndarray, target: dict) -> tuple[torch.Tensor, dict]:
        tensor = preprocess_image(image_bgr)

        if train and torch.rand(1).item() < 0.5:
            tensor, target["boxes"] = _hflip(tensor, target["boxes"])

        return tensor, target

    return transform


# ─── Dataset ─────────────────────────────────────────────────────────────────

class HFDetectionDataset(Dataset):
    """
    PyTorch Dataset for Hyperreflective Foci (HF) object detection.

    Loads images and COCO-format annotations produced by
    convert_labelme_to_coco.py.  Returns image tensors and target dicts
    compatible with torchvision's Faster R-CNN API.

    Args:
        coco_json_path: Path to annotations.json.
        images_dir:     Directory containing the images referenced in the JSON.
        transforms:     Callable(image_bgr, target) → (tensor, target).
                        Use get_transform(train=True/False) from this module.
    """

    def __init__(
        self,
        coco_json_path: str | Path,
        images_dir:     str | Path,
        transforms:     Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(coco_json_path, "r", encoding="utf-8") as fh:
            coco = json.load(fh)

        self.images: list[dict] = coco["images"]

        # image_id → list of annotation dicts
        self._ann_index: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            self._ann_index[ann["image_id"]].append(ann)

    # ── dunder methods ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_info  = self.images[idx]
        image_id  = img_info["id"]

        # ── Load image ──────────────────────────────────────────────────────
        img_path = self.images_dir / img_info["file_name"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        # ── Build target ────────────────────────────────────────────────────
        anns = self._ann_index[image_id]

        if anns:
            # COCO bbox [x, y, w, h] → xyxy for torchvision
            boxes = torch.tensor(
                [[a["bbox"][0],
                  a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2],
                  a["bbox"][1] + a["bbox"][3]] for a in anns],
                dtype=torch.float32,
            )
            labels = torch.ones(len(anns), dtype=torch.int64)  # 1 = HF
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0,      dtype=torch.int64)

        target: dict = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }

        # ── Apply transforms ────────────────────────────────────────────────
        if self.transforms is not None:
            image_bgr, target = self.transforms(image_bgr, target)  # type: ignore[assignment]

        return image_bgr, target  # type: ignore[return-value]


# ─── DataLoader helper ────────────────────────────────────────────────────────

def collate_fn(batch: list) -> tuple:
    """
    Custom collate for variable-size object detection batches.
    Returns a tuple of (images_list, targets_list) – required by Faster R-CNN.
    """
    return tuple(zip(*batch))
