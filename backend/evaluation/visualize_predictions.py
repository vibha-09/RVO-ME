"""
visualize_predictions.py — Qualitative prediction grids for paper figures.

Generates a publication-quality figure grid showing, for each sampled image:
  Column 1 : Original OCT image
  Column 2 : CLAHE-enhanced image
  Column 3 : Ground-truth mask (colour-coded)
  Column 4 : Predicted mask  (colour-coded)
  Column 5 : Alpha-blended overlay (pred on enhanced)

Sampling modes
--------------
  random  — random sample (seed-fixed, reproducible)
  best    — images with highest mean-fg Dice
  worst   — images with lowest mean-fg Dice (non-zero GT)
  diverse — one image per foreground class with highest class-specific Dice

Usage
-----
  cd backend
  python -m evaluation.visualize_predictions \\
      --images-dir  ../dataset/RVO-Lesion/Image_Seg/images \\
      --masks-dir   ../dataset/RVO-Lesion/Image_Seg/masks  \\
      --test-txt    ../dataset/RVO-Lesion/Image_Seg/test.txt \\
      --weights     weights/attention_unet.pth \\
      --output      evaluation/results/segmentation \\
      --n-samples   8 --mode random
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["Background", "SRF", "IRF", "ELM", "EZ"]
N_CLASSES    = 5
FOREGROUND   = slice(1, None)

# BGR colours for cv2 mask rendering
_CLASS_BGR = [
    (30,  30,  30),   # 0 Background
    (243, 150,  33),  # 1 SRF   — blue
    (54,  67, 244),   # 2 IRF   — red
    (59, 235, 255),   # 3 ELM   — amber
    (80, 175,  76),   # 4 EZ    — green
]
# Matplotlib-compatible hex for legend patches
_CLASS_HEX = ["#1a1a1a", "#2196F3", "#F44336", "#FFC107", "#4CAF50"]

OVERLAY_ALPHA = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) int mask to (H, W, 3) uint8 RGB image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, bgr in enumerate(_CLASS_BGR):
        region = mask == c
        out[region] = bgr[::-1]   # BGR → RGB
    return out


def _overlay(gray_img: np.ndarray, mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Alpha-blend colour mask onto grayscale image."""
    base  = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB).astype(float)
    color = _mask_to_rgb(mask).astype(float)
    blend = (1.0 - alpha) * base + alpha * color
    return np.clip(blend, 0, 255).astype(np.uint8)


def _dice_fg(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean foreground Dice for a single image (NaN classes excluded)."""
    dices = []
    for c in range(1, N_CLASSES):
        p_c, g_c = pred == c, gt == c
        denom = p_c.sum() + g_c.sum()
        if denom > 0:
            dices.append(2.0 * float((p_c & g_c).sum()) / float(denom))
    return float(np.mean(dices)) if dices else np.nan


def _class_dice(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p_c, g_c = pred == c, gt == c
    denom = p_c.sum() + g_c.sum()
    return 2.0 * float((p_c & g_c).sum()) / float(denom) if denom > 0 else np.nan


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(
    samples:    list[dict],
    output_path: Path,
    title_prefix: str = "",
) -> None:
    """
    samples: list of dicts with keys:
      stem, original, enhanced, gt_mask, pred_mask, dice_fg
    """
    n_rows   = len(samples)
    n_cols   = 5
    col_titles = ["Original", "Enhanced\n(CLAHE)", "Ground Truth", "Prediction", "Overlay"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        gt_rgb   = _mask_to_rgb(s["gt_mask"])
        pred_rgb = _mask_to_rgb(s["pred_mask"])
        overlay  = _overlay(s["enhanced"], s["pred_mask"])

        images = [
            cv2.cvtColor(s["original"], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.cvtColor(s["enhanced"], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB),
            gt_rgb,
            pred_rgb,
            overlay,
        ]

        for col, (img, ax) in enumerate(zip(images, axes[row])):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(
                    f"{s['stem']}\nDice={s['dice_fg']:.3f}",
                    fontsize=7, rotation=0, labelpad=55, va="center", ha="right",
                )

    # Class-colour legend
    patches = [
        mpatches.Patch(color=_CLASS_HEX[c], label=CLASS_NAMES[c])
        for c in range(N_CLASSES)
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=N_CLASSES,
        fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0),
    )

    suptitle = (f"{title_prefix}Segmentation Predictions — Attention U-Net"
                if title_prefix else "Segmentation Predictions — Attention U-Net")
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(
    images_dir:  Path,
    masks_dir:   Path,
    test_txt:    Path,
    weights_path: Optional[Path],
    output_dir:  Path,
    n_samples:   int  = 8,
    mode:        str  = "random",
    seed:        int  = 42,
) -> None:
    from app.services.preprocessor import apply_clahe
    from app.services.segmentation import load_segmentation_model, predict_segmentation

    wp = str(weights_path) if weights_path and weights_path.exists() else "weights/attention_unet.pth"
    load_segmentation_model(wp)

    stems = []
    with open(test_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(Path(line.replace("\\", "/")).stem)

    print(f"\nBuilding qualitative grid ({mode} mode, n={n_samples}) …")

    # ── Full pass: load, predict, record dice ─────────────────────────────────
    records: list[dict] = []
    for stem in stems:
        img_path  = _find_file(images_dir, stem)
        mask_path = _find_file(masks_dir,  stem)
        if img_path is None or mask_path is None:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enhanced = apply_clahe(img_rgb)

        gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = gt.astype(np.uint8)

        pred = predict_segmentation(enhanced)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        dice_fg = _dice_fg(pred, gt)
        records.append({
            "stem":      stem,
            "original":  img_bgr,
            "enhanced":  enhanced,
            "gt_mask":   gt,
            "pred_mask": pred,
            "dice_fg":   dice_fg,
            # per-class dice for diverse selection
            "class_dice": [_class_dice(pred, gt, c) for c in range(N_CLASSES)],
        })

    if not records:
        print("  WARNING: No valid images found.")
        return

    # ── Select samples based on mode ──────────────────────────────────────────
    rng = np.random.default_rng(seed)

    if mode == "random":
        idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
        selected = [records[i] for i in idx]
        tag = "random"

    elif mode == "best":
        sorted_recs = sorted(
            [r for r in records if not np.isnan(r["dice_fg"])],
            key=lambda r: r["dice_fg"], reverse=True,
        )
        selected = sorted_recs[:n_samples]
        tag = "best"

    elif mode == "worst":
        valid = [r for r in records if not np.isnan(r["dice_fg"]) and r["dice_fg"] > 0]
        sorted_recs = sorted(valid, key=lambda r: r["dice_fg"])
        selected = sorted_recs[:n_samples]
        tag = "worst"

    elif mode == "diverse":
        # Best image per foreground class + fill remaining with random
        selected = []
        seen_stems = set()
        for c in range(1, N_CLASSES):
            valid = [r for r in records if not np.isnan(r["class_dice"][c])]
            if valid:
                best = max(valid, key=lambda r: r["class_dice"][c])
                if best["stem"] not in seen_stems:
                    selected.append(best)
                    seen_stems.add(best["stem"])
        # Fill remaining
        remaining = [r for r in records if r["stem"] not in seen_stems]
        fill_n = max(0, n_samples - len(selected))
        if fill_n > 0 and remaining:
            extra_idx = rng.choice(len(remaining), size=min(fill_n, len(remaining)), replace=False)
            selected += [remaining[i] for i in extra_idx]
        tag = "diverse"

    else:
        raise ValueError(f"Unknown mode: {mode!r}.  Choose from: random, best, worst, diverse")

    # ── Build figure ──────────────────────────────────────────────────────────
    if selected:
        output_dir.mkdir(parents=True, exist_ok=True)
        build_grid(
            selected[:n_samples],
            output_path   = output_dir / f"qualitative_{tag}.png",
            title_prefix  = f"({tag.capitalize()} samples) — ",
        )
        print(f"  Grid written with {len(selected[:n_samples])} rows.")

    # ── Always write all four modes if called without args ────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate qualitative segmentation prediction grids for paper figures"
    )
    p.add_argument("--images-dir",  required=True, type=Path)
    p.add_argument("--masks-dir",   required=True, type=Path)
    p.add_argument("--test-txt",    required=True, type=Path)
    p.add_argument("--weights",     default=None,  type=Path)
    p.add_argument("--output",      default=Path("evaluation/results/segmentation"), type=Path)
    p.add_argument("--n-samples",   default=8,     type=int,
                   help="Number of image rows in the grid (default 8)")
    p.add_argument("--mode",        default="random",
                   choices=["random", "best", "worst", "diverse"],
                   help="Sampling strategy (default: random)")
    p.add_argument("--seed",        default=42,    type=int)
    p.add_argument("--all-modes",   action="store_true",
                   help="Generate grids for all four sampling modes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    modes = ["random", "best", "worst", "diverse"] if args.all_modes else [args.mode]
    for m in modes:
        run_visualization(
            images_dir   = args.images_dir,
            masks_dir    = args.masks_dir,
            test_txt     = args.test_txt,
            weights_path = args.weights,
            output_dir   = args.output,
            n_samples    = args.n_samples,
            mode         = m,
            seed         = args.seed,
        )
