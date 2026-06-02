"""
visualize_detections.py — Qualitative detection prediction grids for paper figures.

For each sampled image shows a two-panel figure:
  Left  : Image with GT boxes (cyan dashed) + predicted boxes coloured by outcome
            TP = green (IoU ≥ 0.50 with a GT box)
            FP = red   (no matching GT box)
  Right : Zoomed confidence-score annotation on the same image

Box legend
----------
  Green solid  — True Positive (TP)
  Red   solid  — False Positive (FP)
  Cyan  dashed — False Negative / missed GT box

Sampling modes
--------------
  random  — reproducible random sample
  best    — images with highest per-image recall (most TP, fewest FN)
  worst   — images with most false negatives (missed GT boxes)
  fp_heavy— images with most false positives

Usage
-----
  cd backend
  python -m evaluation.visualize_detections \\
      --images-dir   ../dataset/RVO-Lesion/Image_Seg/images \\
      --labelme-dir  ../dataset/RVO-Lesion/RVO_Lesion_Labelme \\
      --test-txt     ../dataset/RVO-Lesion/Image_Seg/test.txt \\
      --weights      weights/hf_frcnn.pt \\
      --output       evaluation/results/detection \\
      --n-samples    8 --mode all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

IOU_THRESHOLD = 0.50
_HF_HALF      = 5   # half-width for point annotations → 10×10 proxy box

# Colours used when drawing boxes
_COLOR_TP  = (0,   200,  0)    # green  (BGR)
_COLOR_FP  = (0,    0,  220)   # red    (BGR — OpenCV)
_COLOR_FN  = (220, 165,  0)    # cyan-ish dashed  (BGR)
_COLOR_GT  = (200, 180,  0)    # all GT thin line  (BGR)

# Matplotlib hex (for legend patches)
_HEX_TP = "#00C800"
_HEX_FP = "#DC0000"
_HEX_FN = "#00A0DC"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _classify_boxes(
    pred_boxes:  np.ndarray,   # (N, 4)
    pred_scores: np.ndarray,   # (N,)
    gt_boxes:    np.ndarray,   # (M, 4)
    conf_thr:    float,
    iou_thr:     float = IOU_THRESHOLD,
) -> tuple[list[int], list[int], list[int]]:
    """
    Returns (tp_indices, fp_indices, fn_gt_indices)
    all relative to the filtered prediction list (score >= conf_thr).
    fn_gt_indices are indices into gt_boxes.
    """
    mask  = pred_scores >= conf_thr
    pred  = pred_boxes[mask]
    scr   = pred_scores[mask]
    orig_idx = np.where(mask)[0]    # back-map to original pred indices

    tp_idx: list[int] = []
    fp_idx: list[int] = []
    fn_gt:  list[int] = list(range(len(gt_boxes)))

    if len(pred) > 0 and len(gt_boxes) > 0:
        iou        = _box_iou_matrix(pred, gt_boxes)
        matched_gt = set()
        for i in np.argsort(-scr):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= iou_thr and j not in matched_gt:
                tp_idx.append(int(orig_idx[i]))
                matched_gt.add(j)
            else:
                fp_idx.append(int(orig_idx[i]))
        fn_gt = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    elif len(pred) > 0:
        fp_idx = list(map(int, orig_idx))
    # else: all gt are FN

    return tp_idx, fp_idx, fn_gt


def _draw_boxes(
    image:       np.ndarray,   # BGR uint8
    pred_boxes:  np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes:    np.ndarray,
    conf_thr:    float,
    enlarge:     int = 8,
) -> np.ndarray:
    """
    Returns annotated BGR image with TP/FP/FN boxes.
    Boxes are enlarged by `enlarge` pixels for visibility on small HF.
    """
    img = image.copy()
    h, w = img.shape[:2]

    tp_idx, fp_idx, fn_gt = _classify_boxes(pred_boxes, pred_scores, gt_boxes, conf_thr)
    tp_set = set(tp_idx)
    fp_set = set(fp_idx)

    # Draw GT boxes (all, thin cyan dashed)
    for gt in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in gt]
        # Slightly enlarge for visibility
        x1 = max(0, x1 - enlarge); y1 = max(0, y1 - enlarge)
        x2 = min(w, x2 + enlarge); y2 = min(h, y2 + enlarge)
        cv2.rectangle(img, (x1, y1), (x2, y2), _COLOR_GT, 1)

    # Draw predicted boxes coloured by TP/FP
    for i, (box, scr) in enumerate(zip(pred_boxes, pred_scores)):
        if scr < conf_thr:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1 - enlarge); y1 = max(0, y1 - enlarge)
        x2 = min(w, x2 + enlarge); y2 = min(h, y2 + enlarge)
        color     = _COLOR_TP if i in tp_set else _COLOR_FP
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # Score label
        label = f"{scr:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        ly = max(y1 - 2, lh + 2)
        cv2.rectangle(img, (x1, ly - lh - 1), (x1 + lw + 1, ly + 1), color, -1)
        cv2.putText(img, label, (x1 + 1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".png", ".jpeg", ".bmp"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt_labelme(labelme_dir: Path) -> dict[str, np.ndarray]:
    gt: dict[str, list] = defaultdict(list)
    for jp in labelme_dir.rglob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        img_h = data.get("imageHeight", 10000)
        img_w = data.get("imageWidth",  10000)
        for shape in data.get("shapes", []):
            if shape.get("label", "").upper() != "HF":
                continue
            stype = shape.get("shape_type", "").lower()
            pts   = shape.get("points", [])
            if stype == "point" and pts:
                cx, cy = float(pts[0][0]), float(pts[0][1])
                box = [max(0., cx - _HF_HALF), max(0., cy - _HF_HALF),
                       min(float(img_w), cx + _HF_HALF), min(float(img_h), cy + _HF_HALF)]
            elif len(pts) >= 2:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                box = [max(0, min(xs)), max(0, min(ys)),
                       min(img_w, max(xs)), min(img_h, max(ys))]
            else:
                continue
            if box[2] > box[0] and box[3] > box[1]:
                gt[jp.stem].append(box)
    return {k: np.array(v, dtype=np.float32).reshape(-1, 4) for k, v in gt.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(
    samples:     list[dict],
    output_path: Path,
    conf_thr:    float,
    title_tag:   str = "",
) -> None:
    """
    samples: list of dicts with keys:
      stem, image_bgr, pred_boxes, pred_scores, gt_boxes
    """
    n_rows = len(samples)
    if n_rows == 0:
        return

    col_titles = ["Image + GT (cyan)", "Predictions\n(TP=green, FP=red, FN=yellow missed)"]

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(samples):
        img_rgb = cv2.cvtColor(s["image_bgr"], cv2.COLOR_BGR2RGB)
        ann_bgr = _draw_boxes(
            s["image_bgr"],
            s["pred_boxes"], s["pred_scores"],
            s["gt_boxes"], conf_thr,
        )
        ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

        tp_idx, fp_idx, fn_gt = _classify_boxes(
            s["pred_boxes"], s["pred_scores"], s["gt_boxes"], conf_thr
        )

        for col, (img, ax) in enumerate(zip([img_rgb, ann_rgb], axes[row])):
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight="bold", pad=4)

        # Row label: image stem + TP/FP/FN counts
        axes[row, 0].set_ylabel(
            f"{s['stem']}\nGT={len(s['gt_boxes'])}  "
            f"TP={len(tp_idx)}  FP={len(fp_idx)}  FN={len(fn_gt)}",
            fontsize=7, rotation=0, labelpad=80, va="center", ha="right",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=_HEX_TP, label="TP (IoU ≥ 0.50)"),
        mpatches.Patch(color=_HEX_FP, label="FP (no GT match)"),
        mpatches.Patch(facecolor="none", edgecolor="#D4A800",
                       linewidth=1.5, label="GT box (cyan=all GT)"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=3,
        fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0),
    )

    prefix = f"({title_tag}) " if title_tag else ""
    fig.suptitle(
        f"{prefix}HF Detection Predictions — Faster R-CNN  "
        f"(conf ≥ {conf_thr:.3f})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization(
    images_dir:       Path,
    test_txt:         Path,
    weights_path:     Optional[Path],
    output_dir:       Path,
    annotations_json: Optional[Path] = None,
    labelme_dir:      Optional[Path] = None,
    conf_threshold:   float = 0.50,
    n_samples:        int   = 8,
    mode:             str   = "random",
    seed:             int   = 42,
) -> None:
    from app.services.detection    import load_detection_model, predict_detections
    from app.services.preprocessor import apply_clahe

    wp = str(weights_path) if weights_path and weights_path.exists() else "weights/hf_frcnn.pt"
    load_detection_model(wp)

    # Load GT
    if annotations_json and annotations_json.exists():
        from evaluate_detection import _load_gt_from_coco
        gt_map = _load_gt_from_coco(annotations_json)
    elif labelme_dir and labelme_dir.exists():
        gt_map = _load_gt_labelme(labelme_dir)
    else:
        raise FileNotFoundError("Provide --annotations or --labelme-dir for ground truth.")

    stems = []
    with open(test_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(Path(line.replace("\\", "/")).stem)

    print(f"\nBuilding detection grid ({mode} mode, n={n_samples}) …")

    # ── Full pass: predict and collect stats ──────────────────────────────────
    records: list[dict] = []

    for stem in stems:
        img_path = _find_file(images_dir, stem)
        if img_path is None:
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enhanced = apply_clahe(img_rgb)

        det         = predict_detections(enhanced, threshold=0.05)
        pred_boxes  = np.array(det["boxes"],  dtype=np.float32).reshape(-1, 4) \
                      if len(det["boxes"])  > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.array(det["scores"], dtype=np.float32).ravel() \
                      if len(det["scores"]) > 0 else np.zeros(0, dtype=np.float32)
        gt_boxes    = gt_map.get(stem, np.zeros((0, 4), dtype=np.float32))

        tp_idx, fp_idx, fn_gt = _classify_boxes(
            pred_boxes, pred_scores, gt_boxes, conf_threshold
        )
        recall = len(tp_idx) / max(len(gt_boxes), 1)

        records.append({
            "stem":        stem,
            "image_bgr":   img_bgr,
            "pred_boxes":  pred_boxes,
            "pred_scores": pred_scores,
            "gt_boxes":    gt_boxes,
            "n_gt":        len(gt_boxes),
            "n_tp":        len(tp_idx),
            "n_fp":        len(fp_idx),
            "n_fn":        len(fn_gt),
            "recall":      recall,
        })

    if not records:
        print("  WARNING: No valid images found.")
        return

    # ── Select samples ────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    if mode == "random":
        idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
        selected = [records[i] for i in idx]
        tag = "random"

    elif mode == "best":
        pos  = [r for r in records if r["n_gt"] > 0]
        sel  = sorted(pos, key=lambda r: r["recall"], reverse=True)[:n_samples]
        selected = sel if sel else records[:n_samples]
        tag = "best_recall"

    elif mode == "worst":
        pos  = [r for r in records if r["n_gt"] > 0]
        sel  = sorted(pos, key=lambda r: r["n_fn"], reverse=True)[:n_samples]
        selected = sel if sel else records[:n_samples]
        tag = "most_fn"

    elif mode == "fp_heavy":
        sel  = sorted(records, key=lambda r: r["n_fp"], reverse=True)[:n_samples]
        selected = sel if sel else records[:n_samples]
        tag = "most_fp"

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose: random, best, worst, fp_heavy")

    output_dir.mkdir(parents=True, exist_ok=True)
    build_grid(
        selected[:n_samples],
        output_path = output_dir / f"qualitative_{tag}.png",
        conf_thr    = conf_threshold,
        title_tag   = tag.replace("_", " ").title(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detection qualitative grids — TP/FP/FN bounding box figures"
    )
    p.add_argument("--images-dir",     required=True,  type=Path)
    p.add_argument("--test-txt",       required=True,  type=Path)
    p.add_argument("--weights",        default=None,   type=Path)
    p.add_argument("--annotations",    default=None,   type=Path)
    p.add_argument("--labelme-dir",    default=None,   type=Path)
    p.add_argument("--output",         default=Path("evaluation/results/detection"), type=Path)
    p.add_argument("--conf-threshold", default=0.50,   type=float,
                   help="Confidence threshold for drawing boxes (default 0.50)")
    p.add_argument("--n-samples",      default=8,      type=int)
    p.add_argument("--mode",           default="random",
                   choices=["random", "best", "worst", "fp_heavy", "all"])
    p.add_argument("--seed",           default=42,     type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    modes = ["random", "best", "worst", "fp_heavy"] if args.mode == "all" else [args.mode]
    for m in modes:
        run_visualization(
            images_dir       = args.images_dir,
            test_txt         = args.test_txt,
            weights_path     = args.weights,
            output_dir       = args.output,
            annotations_json = args.annotations,
            labelme_dir      = args.labelme_dir,
            conf_threshold   = args.conf_threshold,
            n_samples        = args.n_samples,
            mode             = m,
            seed             = args.seed,
        )
