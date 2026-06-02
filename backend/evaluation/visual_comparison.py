"""
Visual comparison evaluation: 5 images each for segmentation and HF detection.
Produces side-by-side figures: Original OCT | Ground Truth | Model Prediction.

Outputs:
  seg_comparison.png          — 5 rows × 3 cols (original / GT overlay / pred overlay)
  det_comparison.png          — 5 rows × 3 cols (original / GT boxes / pred boxes TP/FP/FN)
  seg_comparison_metrics.csv  — per-image Dice & IoU for each of the 5 images
  det_comparison_metrics.csv  — per-image TP/FP/FN/Precision/Recall

Usage:
  python backend/evaluation/visual_comparison.py \
    --images-dir dataset/RVO-Lesion/Image_Seg/test_images \
    --masks-dir  dataset/RVO-Lesion/Image_Seg/test_masks \
    --test-txt   dataset/RVO-Lesion/Image_Seg/test.txt \
    --seg-weights backend/weights/attention_unet.pth \
    --det-weights backend/weights/hf_frcnn.pt \
    --labelme-dir RVO_Lesion_LabelMe \
    --output      backend/evaluation/outputs/visual
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Background", "SRF", "IRF", "ELM", "EZ"]
N_CLASSES = len(CLASS_NAMES)

# RGBA overlays for each class (r, g, b) — 0-255 uint8
MASK_RGB = {
    1: (33,  150, 243),   # SRF — blue
    2: (244,  67,  54),   # IRF — red
    3: (255, 193,   7),   # ELM — amber
    4:  (76, 175,  80),   # EZ  — green
}

BOX_GT  = (50,  255,  50)   # lime green  — ground-truth boxes
BOX_TP  = (50,  220,  50)   # lime        — true-positive pred
BOX_FP  = (240,  40,  40)   # red         — false-positive pred
BOX_FN  = (30,  144, 255)   # dodger blue — false-negative (missed GT)

OVERLAY_ALPHA = 0.55
FIG_DPI = 120
PANEL_SIZE = (4.0, 3.0)   # inches per panel


# ── Overlay helpers ────────────────────────────────────────────────────────────

def mask_to_overlay(mask: np.ndarray, base_img: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Alpha-blend coloured class mask onto base_img (HWC uint8 RGB). BG pixels untouched."""
    overlay = base_img.copy().astype(np.float32)
    for cls_id, colour in MASK_RGB.items():
        region = mask == cls_id
        if region.any():
            for c, v in enumerate(colour):
                overlay[region, c] = (1 - alpha) * overlay[region, c] + alpha * v
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── Segmentation metrics ───────────────────────────────────────────────────────

def seg_metrics(pred: np.ndarray, gt: np.ndarray):
    """Returns per-class Dice and IoU arrays (indices 0..4, 0=background)."""
    dice = np.zeros(N_CLASSES, dtype=np.float32)
    iou  = np.zeros(N_CLASSES, dtype=np.float32)
    for c in range(N_CLASSES):
        p = pred == c
        g = gt   == c
        inter = (p & g).sum()
        union = (p | g).sum()
        denom_d = p.sum() + g.sum()
        dice[c] = (2 * inter / denom_d) if denom_d > 0 else 1.0
        iou[c]  = (inter / union)        if union  > 0 else 1.0
    return {"dice": dice, "iou": iou}


# ── Detection metrics ──────────────────────────────────────────────────────────

def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute NxM IoU matrix between two arrays of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h
    area_a  = (ax2 - ax1) * (ay2 - ay1)
    area_b  = (bx2 - bx1) * (by2 - by1)
    union   = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_boxes(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                gt_boxes: np.ndarray, iou_thr: float = 0.5):
    """
    Greedy descending-confidence matching.
    Returns:
        tp_mask    : bool array [len(pred_boxes)] — True if TP
        fn_indices : list of int indices into gt_boxes that were unmatched (FN)
    """
    tp_mask    = np.zeros(len(pred_boxes), dtype=bool)
    gt_matched = np.zeros(len(gt_boxes),  dtype=bool)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp_mask, list(range(len(gt_boxes)))

    order = np.argsort(-pred_scores)
    iou   = box_iou_matrix(pred_boxes, gt_boxes)  # N_pred × N_gt

    for pi in order:
        best_gt = -1
        best_iou = iou_thr
        for gi in range(len(gt_boxes)):
            if gt_matched[gi]:
                continue
            if iou[pi, gi] >= best_iou:
                best_iou = iou[pi, gi]
                best_gt  = gi
        if best_gt >= 0:
            tp_mask[pi]       = True
            gt_matched[best_gt] = True

    fn_indices = [gi for gi in range(len(gt_boxes)) if not gt_matched[gi]]
    return tp_mask, fn_indices


# ── Data loading ───────────────────────────────────────────────────────────────

def load_stems(test_txt: Optional[str]) -> list:
    if test_txt is None or not Path(test_txt).exists():
        return []
    stems = []
    with open(test_txt) as f:
        for line in f:
            s = line.strip()
            if s:
                stems.append(Path(s).stem)
    return stems


def find_file(directory: str, stem: str) -> Optional[Path]:
    """Find a file with the given stem (any extension) in directory."""
    d = Path(directory)
    if not d.exists():
        return None
    for p in d.iterdir():
        if p.stem == stem:
            return p
    return None


def load_gt_labelme(labelme_dir: str) -> dict:
    """
    Returns {stem: np.ndarray(N,4) xyxy} from LabelMe point annotations.
    HF points are converted to 10×10 proxy boxes (centred on the point).
    """
    gt_map = {}
    ld = Path(labelme_dir)
    if not ld.exists():
        return gt_map
    for json_path in sorted(ld.rglob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
            boxes = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") == "point":
                    x, y = shape["points"][0]
                    boxes.append([x - 5, y - 5, x + 5, y + 5])
            stem = json_path.stem
            if stem not in gt_map:
                gt_map[stem] = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        except Exception:
            continue
    return gt_map


def load_gt_coco(annotations_json: str) -> dict:
    """Returns {stem: np.ndarray(N,4) xyxy} from COCO-format annotations JSON."""
    gt_map = {}
    if not Path(annotations_json).exists():
        return gt_map
    with open(annotations_json) as f:
        coco = json.load(f)
    id_to_stem = {}
    for img in coco.get("images", []):
        id_to_stem[img["id"]] = Path(img["file_name"]).stem
    boxes_by_img = {}
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        x, y, w, h = ann["bbox"]
        boxes_by_img.setdefault(iid, []).append([x, y, x + w, y + h])
    for iid, boxes in boxes_by_img.items():
        stem = id_to_stem.get(iid, str(iid))
        gt_map[stem] = np.array(boxes, dtype=np.float32)
    return gt_map


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_seg_model(weights_path: Optional[str], device: torch.device):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.models.attention_unet import AttentionUNet
    model = AttentionUNet(img_ch=1, output_ch=5)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[seg] Loaded weights: {weights_path}")
    else:
        print(f"[seg] WARNING — weights not found ({weights_path}); using random init")
    model.to(device).eval()
    return model


def _load_det_model(weights_path: Optional[str], device: torch.device):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.models.detector import get_detection_model
    model = get_detection_model(num_classes=2)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[det] Loaded weights: {weights_path}")
    else:
        print(f"[det] WARNING — weights not found ({weights_path}); using random init")
    model.to(device).eval()
    return model


def predict_seg(model, img_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Returns (H,W) uint8 prediction mask."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def _clahe_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def predict_det(model, img_bgr: np.ndarray, device: torch.device,
                threshold: float = 0.5) -> tuple:
    """Returns (boxes np.ndarray(N,4), scores np.ndarray(N,))."""
    gray    = _clahe_preprocess(img_bgr)          # CLAHE-enhanced grayscale uint8
    stacked = np.stack([gray, gray, gray], axis=-1)
    tensor  = torch.from_numpy(stacked.astype(np.float32) / 255.0) \
                   .permute(2, 0, 1).unsqueeze(0).to(device)        # (1, 3, H, W)
    with torch.no_grad():
        outputs = model(tensor)
    out    = outputs[0]
    boxes  = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep   = scores >= threshold
    return boxes[keep], scores[keep]


# ── Image selection strategies ─────────────────────────────────────────────────

def _select_random(stems: list, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    pool = stems[:]
    rng.shuffle(pool)
    return pool[:n]


def _select_diverse_seg(stems: list, images_dir: str, masks_dir: str,
                        n: int = 5, seed: int = 42) -> list:
    """
    Try to pick visually diverse cases:
      one high-SRF, one high-IRF, one low-EZ-integrity, one high-ELM, one mixed.
    Falls back to random if masks cannot be read.
    """
    if not stems:
        return []
    scored = []
    for stem in stems:
        mf = find_file(masks_dir, stem)
        if mf is None:
            continue
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        total = mask.size
        srf = (mask == 1).sum() / total
        irf = (mask == 2).sum() / total
        elm = (mask == 3).sum() / total
        ez  = (mask == 4).sum() / total
        scored.append((stem, srf, irf, elm, ez))

    if len(scored) < n:
        return _select_random(stems, n, seed)

    scored_arr = np.array([[s[1], s[2], s[3], s[4]] for s in scored])

    def best(col_idx, descending=True):
        order = np.argsort(-scored_arr[:, col_idx] if descending else scored_arr[:, col_idx])
        for i in order:
            if scored[i][0] not in chosen:
                return scored[i][0]
        return None

    chosen = []
    for criteria in [(0, True), (1, True), (3, False), (2, True)]:
        c = best(*criteria)
        if c:
            chosen.append(c)
    # fill remainder randomly
    rng = random.Random(seed)
    remainder = [s[0] for s in scored if s[0] not in chosen]
    rng.shuffle(remainder)
    chosen.extend(remainder)
    return chosen[:n]


def _select_diverse_det(stems: list, gt_map: dict, n: int = 5, seed: int = 42) -> list:
    """Pick: most-HF, fewest-HF, mid-HF, 2 random."""
    annotated = [(s, len(gt_map.get(s, []))) for s in stems if s in gt_map]
    if len(annotated) < n:
        return _select_random(list(gt_map.keys()), n, seed)

    annotated.sort(key=lambda x: x[1])
    chosen = [annotated[-1][0], annotated[0][0], annotated[len(annotated)//2][0]]
    chosen = list(dict.fromkeys(chosen))  # deduplicate, preserve order

    rng = random.Random(seed)
    remainder = [s for s, _ in annotated if s not in chosen]
    rng.shuffle(remainder)
    chosen.extend(remainder)
    return chosen[:n]


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_boxes(ax, boxes: np.ndarray, colour_rgb: tuple, linestyle: str = "-",
                linewidth: float = 1.5, label: Optional[str] = None):
    c_norm = tuple(v / 255 for v in colour_rgb)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=linewidth, linestyle=linestyle,
            edgecolor=c_norm, facecolor="none",
            label=label if i == 0 else None
        )
        ax.add_patch(rect)


def _legend_patch(colour_rgb: tuple, label: str):
    return mpatches.Patch(color=tuple(v / 255 for v in colour_rgb), label=label)


def draw_seg_row(axes_row, img_rgb: np.ndarray, gt_mask: np.ndarray,
                 pred_mask: np.ndarray, stem: str, metrics: dict, row_idx: int):
    """Fill a 3-axis row with: original | GT overlay | pred overlay + metrics."""
    gt_overlay   = mask_to_overlay(gt_mask,   img_rgb)
    pred_overlay = mask_to_overlay(pred_mask, img_rgb)

    titles = ["Original OCT", "Ground Truth", "Model Prediction"]
    imgs   = [img_rgb, gt_overlay, pred_overlay]

    for col, (ax, title, img) in enumerate(zip(axes_row, titles, imgs)):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        if col == 0:
            ax.set_ylabel(f"{stem[:20]}", fontsize=7, rotation=0, labelpad=55, va="center")

    # Metrics inset on prediction panel
    ax_pred = axes_row[2]
    classes_fg = [1, 2, 3, 4]
    lines = []
    for c in classes_fg:
        d = metrics["dice"][c]
        i = metrics["iou"][c]
        lines.append(f"{CLASS_NAMES[c]}: D={d:.2f} IoU={i:.2f}")
    text = "\n".join(lines)
    ax_pred.text(0.02, 0.02, text,
                 transform=ax_pred.transAxes,
                 fontsize=6, color="white",
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55))

    # Legend on GT panel
    legend_patches = [
        _legend_patch(MASK_RGB[1], "SRF"),
        _legend_patch(MASK_RGB[2], "IRF"),
        _legend_patch(MASK_RGB[3], "ELM"),
        _legend_patch(MASK_RGB[4], "EZ"),
    ]
    axes_row[1].legend(handles=legend_patches, loc="lower right",
                       fontsize=5, framealpha=0.6, ncol=2)


def draw_det_row(axes_row, img_rgb: np.ndarray,
                 gt_boxes: np.ndarray, pred_boxes: np.ndarray,
                 pred_scores: np.ndarray, tp_mask: np.ndarray,
                 fn_indices: list, stem: str, row_idx: int):
    """Fill a 3-axis row with: original | GT boxes | pred TP/FP + FN boxes."""
    titles = ["Original OCT", "Ground Truth HF", "Model Prediction"]
    for col, (ax, title) in enumerate(zip(axes_row, titles)):
        ax.imshow(img_rgb, cmap="gray" if img_rgb.ndim == 2 else None)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        if col == 0:
            ax.set_ylabel(f"{stem[:20]}", fontsize=7, rotation=0, labelpad=55, va="center")

    # GT panel
    _draw_boxes(axes_row[1], gt_boxes, BOX_GT, linewidth=1.5)
    axes_row[1].text(0.02, 0.98, f"GT HF: {len(gt_boxes)}",
                     transform=axes_row[1].transAxes,
                     fontsize=7, color="white", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55))

    # Pred panel: colour by TP/FP
    if len(pred_boxes) > 0:
        tp_boxes = pred_boxes[tp_mask]
        fp_boxes = pred_boxes[~tp_mask]
        _draw_boxes(axes_row[2], tp_boxes, BOX_TP, linewidth=1.5)
        _draw_boxes(axes_row[2], fp_boxes, BOX_FP, linewidth=1.5)

    # FN boxes (dashed) from GT
    fn_boxes = gt_boxes[fn_indices] if len(fn_indices) > 0 and len(gt_boxes) > 0 else np.zeros((0, 4))
    _draw_boxes(axes_row[2], fn_boxes, BOX_FN, linestyle="--", linewidth=1.5)

    tp_n = int(tp_mask.sum())
    fp_n = int((~tp_mask).sum())
    fn_n = len(fn_indices)
    prec = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
    rec  = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
    text = f"TP={tp_n} FP={fp_n} FN={fn_n}\nPrec={prec:.2f} Rec={rec:.2f}"
    axes_row[2].text(0.02, 0.02, text,
                     transform=axes_row[2].transAxes,
                     fontsize=6.5, color="white", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55))

    # Legend
    patches = [
        _legend_patch(BOX_GT, "GT"),
        _legend_patch(BOX_TP, "TP"),
        _legend_patch(BOX_FP, "FP"),
        _legend_patch(BOX_FN, "FN"),
    ]
    axes_row[2].legend(handles=patches, loc="upper right",
                       fontsize=5, framealpha=0.6, ncol=2)


# ── Figure assembly ────────────────────────────────────────────────────────────

def plot_seg_comparison(records: list, output_path: str):
    n = len(records)
    fig_w = PANEL_SIZE[0] * 3
    fig_h = PANEL_SIZE[1] * n
    fig, axes = plt.subplots(n, 3, figsize=(fig_w, fig_h), dpi=FIG_DPI)
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Segmentation: Original | Ground Truth | Prediction",
                 fontsize=13, fontweight="bold", y=1.005)
    for row_idx, rec in enumerate(records):
        draw_seg_row(axes[row_idx], rec["img_rgb"], rec["gt_mask"],
                     rec["pred_mask"], rec["stem"], rec["metrics"], row_idx)
    plt.tight_layout(rect=[0.07, 0, 1, 1])
    plt.savefig(output_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"[seg] Saved figure → {output_path}")


def plot_det_comparison(records: list, output_path: str):
    n = len(records)
    fig_w = PANEL_SIZE[0] * 3
    fig_h = PANEL_SIZE[1] * n
    fig, axes = plt.subplots(n, 3, figsize=(fig_w, fig_h), dpi=FIG_DPI)
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("HF Detection: Original | Ground Truth | Prediction (TP/FP/FN)",
                 fontsize=13, fontweight="bold", y=1.005)
    for row_idx, rec in enumerate(records):
        draw_det_row(axes[row_idx],
                     rec["img_rgb"], rec["gt_boxes"],
                     rec["pred_boxes"], rec["pred_scores"],
                     rec["tp_mask"], rec["fn_indices"],
                     rec["stem"], row_idx)
    plt.tight_layout(rect=[0.07, 0, 1, 1])
    plt.savefig(output_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"[det] Saved figure → {output_path}")


# ── CSV export ─────────────────────────────────────────────────────────────────

def save_seg_csv(records: list, output_path: str):
    rows = []
    for rec in records:
        m = rec["metrics"]
        row = {"image": rec["stem"]}
        for c in range(1, N_CLASSES):
            row[f"dice_{CLASS_NAMES[c]}"] = f"{m['dice'][c]:.4f}"
            row[f"iou_{CLASS_NAMES[c]}"]  = f"{m['iou'][c]:.4f}"
        mean_dice = m["dice"][1:].mean()
        mean_iou  = m["iou"][1:].mean()
        row["mean_dice"] = f"{mean_dice:.4f}"
        row["mean_iou"]  = f"{mean_iou:.4f}"
        rows.append(row)
    fieldnames = ["image"] + [f for c in range(1, N_CLASSES)
                               for f in (f"dice_{CLASS_NAMES[c]}", f"iou_{CLASS_NAMES[c]}")] + \
                 ["mean_dice", "mean_iou"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[seg] Saved CSV   → {output_path}")


def save_det_csv(records: list, output_path: str):
    rows = []
    for rec in records:
        tp = int(rec["tp_mask"].sum())
        fp = int((~rec["tp_mask"]).sum())
        fn = len(rec["fn_indices"])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        rows.append({
            "image": rec["stem"],
            "gt_count":  len(rec["gt_boxes"]),
            "pred_count": len(rec["pred_boxes"]),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": f"{prec:.4f}",
            "recall":    f"{rec_:.4f}",
            "F1":        f"{f1:.4f}",
        })
    fieldnames = ["image", "gt_count", "pred_count", "TP", "FP", "FN",
                  "precision", "recall", "F1"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[det] Saved CSV   → {output_path}")


# ── Main runner ────────────────────────────────────────────────────────────────

def run_visual_comparison(
    images_dir:      str,
    masks_dir:       str,
    test_txt:        Optional[str],
    seg_weights:     Optional[str],
    det_weights:     Optional[str],
    annotations:     Optional[str],
    labelme_dir:     Optional[str],
    output_dir:      str,
    n_images:        int     = 5,
    seg_stems:       Optional[list] = None,
    det_stems:       Optional[list] = None,
    select_strategy: str     = "diverse",
    det_conf:        float   = 0.5,
    seed:            int     = 42,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load stems ────────────────────────────────────────────────────────────
    all_stems = load_stems(test_txt)
    if not all_stems:
        # fallback: enumerate images_dir
        img_dir = Path(images_dir)
        all_stems = [p.stem for p in sorted(img_dir.iterdir())
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")]
    print(f"Total stems found: {len(all_stems)}")

    # ── GT for detection ─────────────────────────────────────────────────────
    gt_map = {}
    if annotations and Path(annotations).exists():
        gt_map = load_gt_coco(annotations)
        print(f"[det] Loaded {len(gt_map)} GT entries from COCO JSON")
    elif labelme_dir and Path(labelme_dir).exists():
        gt_map = load_gt_labelme(labelme_dir)
        print(f"[det] Loaded {len(gt_map)} GT entries from LabelMe dir")
    else:
        print("[det] WARNING — no GT annotations found; detection comparison skipped")

    # ── Select seg stems ──────────────────────────────────────────────────────
    if seg_stems:
        chosen_seg = [Path(s).stem for s in seg_stems][:n_images]
    elif select_strategy == "diverse":
        chosen_seg = _select_diverse_seg(all_stems, images_dir, masks_dir, n_images, seed)
    else:
        chosen_seg = _select_random(all_stems, n_images, seed)

    # ── Select det stems ──────────────────────────────────────────────────────
    if det_stems:
        chosen_det = [Path(s).stem for s in det_stems][:n_images]
    elif gt_map:
        if select_strategy == "diverse":
            chosen_det = _select_diverse_det(all_stems, gt_map, n_images, seed)
        else:
            chosen_det = _select_random([s for s in all_stems if s in gt_map], n_images, seed)
    else:
        chosen_det = []

    print(f"[seg] Selected {len(chosen_seg)} stems: {chosen_seg}")
    print(f"[det] Selected {len(chosen_det)} stems: {chosen_det}")

    # ── Load models ───────────────────────────────────────────────────────────
    seg_model = _load_seg_model(seg_weights, device) if chosen_seg else None
    det_model = None
    if chosen_det:
        try:
            det_model = _load_det_model(det_weights, device)
            print(f"[det] Model loaded.")
        except Exception as e:
            print(f"[det] WARNING — could not load detection model: {e}")

    # ── Segmentation comparison ───────────────────────────────────────────────
    if seg_model and chosen_seg:
        seg_records = []
        for stem in chosen_seg:
            img_path  = find_file(images_dir, stem)
            mask_path = find_file(masks_dir,  stem)
            if img_path is None:
                print(f"[seg] image not found for stem={stem}, skipping")
                continue
            if mask_path is None:
                print(f"[seg] GT mask not found for stem={stem}, skipping")
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[seg] could not read {img_path}, skipping")
                continue
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"[seg] could not read mask {mask_path}, skipping")
                continue
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred_mask = predict_seg(seg_model, img_bgr, device)
            # resize GT mask to match image dims if needed
            h, w = img_bgr.shape[:2]
            if gt_mask.shape != (h, w):
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            m = seg_metrics(pred_mask, gt_mask)
            seg_records.append({
                "stem": stem, "img_rgb": img_rgb,
                "gt_mask": gt_mask, "pred_mask": pred_mask,
                "metrics": m,
            })
            fg_dice = m["dice"][1:].mean()
            print(f"  [seg] {stem}: mean_fg_dice={fg_dice:.3f}")

        if seg_records:
            plot_seg_comparison(seg_records,
                                str(Path(output_dir) / "seg_comparison.png"))
            save_seg_csv(seg_records,
                         str(Path(output_dir) / "seg_comparison_metrics.csv"))

    # ── Detection comparison ──────────────────────────────────────────────────
    if det_model and chosen_det and gt_map:
        det_records = []
        for stem in chosen_det:
            if stem not in gt_map:
                print(f"[det] no GT for stem={stem}, skipping")
                continue
            img_path = find_file(images_dir, stem)
            if img_path is None:
                print(f"[det] image not found for stem={stem}, skipping")
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gt_boxes  = gt_map[stem]
            pred_boxes, pred_scores = predict_det(det_model, img_bgr, device, det_conf)
            tp_mask, fn_indices = match_boxes(pred_boxes, pred_scores, gt_boxes)
            tp = int(tp_mask.sum()); fp = int((~tp_mask).sum()); fn = len(fn_indices)
            print(f"  [det] {stem}: GT={len(gt_boxes)} pred={len(pred_boxes)} TP={tp} FP={fp} FN={fn}")
            det_records.append({
                "stem": stem, "img_rgb": img_rgb,
                "gt_boxes": gt_boxes, "pred_boxes": pred_boxes,
                "pred_scores": pred_scores, "tp_mask": tp_mask,
                "fn_indices": fn_indices,
            })

        if det_records:
            plot_det_comparison(det_records,
                                str(Path(output_dir) / "det_comparison.png"))
            save_det_csv(det_records,
                         str(Path(output_dir) / "det_comparison_metrics.csv"))

    print("\nDone.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Visual comparison evaluation (5 images each)")
    p.add_argument("--images-dir",  required=True,
                   help="Directory of OCT images")
    p.add_argument("--masks-dir",   default=None,
                   help="Directory of GT segmentation masks (uint8, 0-4)")
    p.add_argument("--test-txt",    default=None,
                   help="test.txt file listing stem names (one per line)")
    p.add_argument("--seg-weights", default="backend/weights/attention_unet.pth",
                   help="Path to Attention U-Net .pth weights")
    p.add_argument("--det-weights", default="backend/weights/hf_frcnn.pt",
                   help="Path to Faster R-CNN .pt weights")
    p.add_argument("--annotations", default=None,
                   help="COCO-format annotations JSON (for detection GT)")
    p.add_argument("--labelme-dir", default=None,
                   help="LabelMe point-annotation directory (alt. GT source)")
    p.add_argument("--output",      default="backend/evaluation/outputs/visual",
                   help="Output directory for figures and CSVs")
    p.add_argument("--n-images",    type=int, default=5,
                   help="Number of images per task (default 5)")
    p.add_argument("--seg-stems",   nargs="+", default=None,
                   help="Explicit list of stems to use for segmentation comparison")
    p.add_argument("--det-stems",   nargs="+", default=None,
                   help="Explicit list of stems to use for detection comparison")
    p.add_argument("--select-strategy", choices=["random", "diverse"], default="diverse",
                   help="Image selection strategy (default: diverse)")
    p.add_argument("--det-conf",    type=float, default=0.5,
                   help="Detection confidence threshold (default 0.5)")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_visual_comparison(
        images_dir      = args.images_dir,
        masks_dir       = args.masks_dir,
        test_txt        = args.test_txt,
        seg_weights     = args.seg_weights,
        det_weights     = args.det_weights,
        annotations     = args.annotations,
        labelme_dir     = args.labelme_dir,
        output_dir      = args.output,
        n_images        = args.n_images,
        seg_stems       = args.seg_stems,
        det_stems       = args.det_stems,
        select_strategy = args.select_strategy,
        det_conf        = args.det_conf,
        seed            = args.seed,
    )
