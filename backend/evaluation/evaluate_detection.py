"""
HF Detection evaluation — Faster R-CNN on RVO-Lesion test split.

Metrics
-------
COCO-style AP   : AP@0.50 (± 95 % CI), AP@0.75, mAP@[.50:.05:.95]
                  AP_small (<32²), AP_medium (32²–96²), AP_large (≥96²)
Average Recall  : AR@1, AR@10, AR@100
At opt. F1      : Precision (± 95 % CI), Recall (± 95 % CI), F1 (± 95 % CI)
FROC            : Sensitivity at FP/img ∈ {0.125, 0.25, 0.5, 1, 2, 4, 8}
                  CPM (Competition Performance Metric, mean of 7 points)
                  pAUC-FROC [0.125 – 8 FP/image] (normalised)
Count stats     : Pearson r, Spearman r, MAE, RMSE, MedAE
                  (predicted HF count vs GT count per image)

Bootstrap CI    : 1 000 image-level resamples for AP@50, Prec, Recall, F1

Outputs (written to --output directory)
-----------------------------------------
  detection_metrics.csv         scalar metrics + 95 % CI
  detection_per_image.csv       per-image TP/FP/FN/precision/recall/counts
  pr_curve.png                  Precision–Recall curve (IoU=0.50) with CI
  froc_curve.png                FROC curve with CPM operating points
  score_distribution.png        TP vs FP confidence-score histograms
  count_correlation.png         predicted vs GT HF-count scatter
  bland_altman.png              Bland-Altman count-agreement plot
  fp_distribution.png           false-positives-per-image histogram
  gt_count_distribution.png     GT HF count distribution in test set
  latex_table.tex               publication-ready LaTeX table with 95 % CI

Ground-truth source (priority order)
--------------------------------------
  1. --annotations   COCO-format JSON (from convert_labelme_to_coco.py)
  2. --labelme-dir   Raw LabelMe JSON folder (automatic fallback)

Usage
-----
  cd backend
  python -m evaluation.evaluate_detection \\
      --images-dir  ../dataset/RVO-Lesion/Image_Seg/images \\
      --labelme-dir ../dataset/RVO-Lesion/RVO_Lesion_Labelme \\
      --test-txt    ../dataset/RVO-Lesion/Image_Seg/test.txt \\
      --weights     weights/hf_frcnn.pt \\
      --output      evaluation/results/detection
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────────────────────────────────────
# FROC operating points (LUNA16 / standard lesion-detection benchmark)
# ─────────────────────────────────────────────────────────────────────────────

FROC_FP_TARGETS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# COCO size thresholds (pixels²)
SIZE_SMALL_MAX  = 32 ** 2      # 1 024
SIZE_MEDIUM_MAX = 96 ** 2      # 9 216


# ─────────────────────────────────────────────────────────────────────────────
# IoU utilities
# ─────────────────────────────────────────────────────────────────────────────

def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised pairwise IoU.
    boxes_a : (N, 4)  [x1, y1, x2, y2]
    boxes_b : (M, 4)
    returns : (N, M)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# PR / AP utilities
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_curve(
    all_scores:   np.ndarray,
    all_tp_flags: np.ndarray,
    n_gt_total:   int,
) -> tuple[np.ndarray, np.ndarray]:
    order     = np.argsort(-all_scores)
    tp_sorted = all_tp_flags[order].cumsum()
    fp_sorted = (~all_tp_flags[order].astype(bool)).cumsum()
    prec = tp_sorted / (tp_sorted + fp_sorted + 1e-9)
    rec  = tp_sorted / (n_gt_total + 1e-9)
    return prec, rec


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO 101-point interpolated Average Precision."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = sum(mpre[mrec >= t].max() if (mrec >= t).any() else 0.0
             for t in np.linspace(0.0, 1.0, 101))
    return float(ap / 101)


def f1_optimal_threshold(
    scores:   np.ndarray,
    tp_flags: np.ndarray,
    n_gt:     int,
) -> tuple[float, float, float, float]:
    """Returns (precision, recall, f1, threshold) at the F1-optimal cut."""
    order  = np.argsort(-scores)
    tp_cum = tp_flags[order].astype(float).cumsum()
    n_det  = np.arange(1, len(order) + 1, dtype=float)
    prec   = tp_cum / n_det
    rec    = tp_cum / max(n_gt, 1)
    f1     = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    best   = int(np.argmax(f1))
    return float(prec[best]), float(rec[best]), float(f1[best]), float(scores[order[best]])


# ─────────────────────────────────────────────────────────────────────────────
# Average Recall at K detections per image
# ─────────────────────────────────────────────────────────────────────────────

def average_recall_at_k(
    per_image_data: list[dict],
    iou_threshold:  float = 0.50,
    max_dets:       int   = 100,
) -> float:
    recalls = []
    for img in per_image_data:
        gt   = np.array(img["gt_boxes"],    dtype=np.float32).reshape(-1, 4)
        pred = np.array(img["pred_boxes"],  dtype=np.float32).reshape(-1, 4)[:max_dets]
        scr  = np.array(img["pred_scores"], dtype=np.float32)[:max_dets]

        if len(gt) == 0:
            continue
        if len(pred) == 0:
            recalls.append(0.0)
            continue

        iou        = box_iou_matrix(pred, gt)
        matched_gt = set()
        tp = 0
        for i in np.argsort(-scr):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= iou_threshold and j not in matched_gt:
                tp += 1
                matched_gt.add(j)
        recalls.append(tp / len(gt))
    return float(np.mean(recalls)) if recalls else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FROC metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_froc_metrics(
    fp_per_image: np.ndarray,
    sensitivity:  np.ndarray,
) -> dict:
    """
    Computes CPM score and pAUC-FROC from a FROC curve.
    fp_per_image, sensitivity must be monotonically increasing in fp.
    """
    if len(fp_per_image) == 0:
        result = {f"sens_fp{_key(t)}": np.nan for t in FROC_FP_TARGETS}
        result.update({"cpm": np.nan, "pauc_froc": np.nan})
        return result

    sens_at = {}
    for fp_t in FROC_FP_TARGETS:
        idx = np.searchsorted(fp_per_image, fp_t)
        if idx == 0:
            s = float(sensitivity[0]) if len(sensitivity) > 0 else 0.0
        elif idx >= len(sensitivity):
            s = float(sensitivity[-1])
        else:
            # Linear interpolation
            x0, x1 = fp_per_image[idx - 1], fp_per_image[idx]
            y0, y1 = sensitivity[idx - 1],  sensitivity[idx]
            t = (fp_t - x0) / (x1 - x0) if x1 > x0 else 0.0
            s = float(y0 + t * (y1 - y0))
        sens_at[fp_t] = s

    cpm = float(np.mean(list(sens_at.values())))

    # Normalised partial AUC between 0.125 and 8 FP/image
    fp_lo, fp_hi = FROC_FP_TARGETS[0], FROC_FP_TARGETS[-1]
    mask = (fp_per_image >= fp_lo) & (fp_per_image <= fp_hi)
    if mask.sum() >= 2:
        fp_seg  = np.concatenate([[fp_lo], fp_per_image[mask], [fp_hi]])
        s_lo    = float(np.interp(fp_lo, fp_per_image, sensitivity))
        s_hi    = float(np.interp(fp_hi, fp_per_image, sensitivity))
        sen_seg = np.concatenate([[s_lo], sensitivity[mask], [s_hi]])
        pauc    = float(np.trapezoid(sen_seg, fp_seg) / (fp_hi - fp_lo))
    else:
        pauc = np.nan

    result = {"cpm": cpm, "pauc_froc": pauc}
    for fp_t in FROC_FP_TARGETS:
        result[f"sens_fp{_key(fp_t)}"] = float(sens_at[fp_t])
    return result


def _key(fp: float) -> str:
    """'0.125' → '0125',  '1.0' → '10'."""
    return str(fp).replace(".", "")


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def _match_image_at_iou(img: dict, iou_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, tp_flags) for one image at given IoU threshold."""
    gt   = np.array(img["gt_boxes"],   dtype=np.float32).reshape(-1, 4)
    pred = np.array(img["pred_boxes"],  dtype=np.float32).reshape(-1, 4)
    scr  = np.array(img["pred_scores"], dtype=np.float32)

    if len(pred) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=bool)

    tp = np.zeros(len(pred), dtype=bool)
    if len(gt) > 0:
        iou        = box_iou_matrix(pred, gt)
        matched_gt = set()
        for i in range(len(pred)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= iou_threshold and j not in matched_gt:
                tp[i] = True
                matched_gt.add(j)

    return scr, tp


def bootstrap_ci_ap(
    per_image_data: list[dict],
    iou_threshold:  float = 0.50,
    n_bootstrap:    int   = 1000,
    seed:           int   = 42,
) -> tuple[float, float]:
    """Image-level bootstrap 95 % CI for AP at given IoU threshold."""
    n   = len(per_image_data)
    rng = np.random.default_rng(seed)
    boot_aps: list[float] = []

    for _ in range(n_bootstrap):
        idx     = rng.integers(0, n, size=n)
        sample  = [per_image_data[i] for i in idx]
        n_gt    = sum(len(s["gt_boxes"]) for s in sample)
        if n_gt == 0:
            continue

        all_s, all_tp = [], []
        for img in sample:
            s, tp = _match_image_at_iou(img, iou_threshold)
            all_s.extend(s.tolist())
            all_tp.extend(tp.tolist())

        if not all_s:
            continue
        prec, rec = precision_recall_curve(
            np.array(all_s, dtype=np.float32),
            np.array(all_tp, dtype=bool),
            n_gt,
        )
        boot_aps.append(compute_ap(prec, rec))

    if len(boot_aps) < 2:
        return np.nan, np.nan
    return float(np.percentile(boot_aps, 2.5)), float(np.percentile(boot_aps, 97.5))


def bootstrap_ci_prf(
    per_image_data: list[dict],
    conf_threshold: float,
    iou_threshold:  float = 0.50,
    n_bootstrap:    int   = 1000,
    seed:           int   = 42,
) -> dict[str, tuple[float, float]]:
    """
    Image-level bootstrap 95 % CI for Precision, Recall, F1
    at a fixed confidence threshold.
    """
    n   = len(per_image_data)
    rng = np.random.default_rng(seed + 1)
    boot_prec: list[float] = []
    boot_rec:  list[float] = []
    boot_f1:   list[float] = []

    for _ in range(n_bootstrap):
        idx    = rng.integers(0, n, size=n)
        sample = [per_image_data[i] for i in idx]

        tp_sum = fp_sum = fn_sum = 0
        for img in sample:
            gt    = np.array(img["gt_boxes"],   dtype=np.float32).reshape(-1, 4)
            pred  = np.array(img["pred_boxes"],  dtype=np.float32).reshape(-1, 4)
            scr   = np.array(img["pred_scores"], dtype=np.float32)

            # Filter by confidence threshold
            mask  = scr >= conf_threshold
            pred  = pred[mask]
            scr   = scr[mask]

            tp = 0
            if len(pred) > 0 and len(gt) > 0:
                iou        = box_iou_matrix(pred, gt)
                matched_gt = set()
                for i in range(len(pred)):
                    j = int(np.argmax(iou[i]))
                    if iou[i, j] >= iou_threshold and j not in matched_gt:
                        tp += 1
                        matched_gt.add(j)

            tp_sum += tp
            fp_sum += max(0, len(pred) - tp)
            fn_sum += max(0, len(gt) - tp)

        prec = tp_sum / max(tp_sum + fp_sum, 1)
        rec  = tp_sum / max(tp_sum + fn_sum, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        boot_prec.append(prec)
        boot_rec.append(rec)
        boot_f1.append(f1)

    def _ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "precision": _ci(boot_prec),
        "recall":    _ci(boot_rec),
        "f1":        _ci(boot_f1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Size-based AP
# ─────────────────────────────────────────────────────────────────────────────

def _ap_for_size_class(
    per_image_data: list[dict],
    size_label:     str,           # "small" | "medium" | "large"
    iou_threshold:  float = 0.50,
) -> float:
    """AP restricted to GT boxes in a COCO size class."""
    if size_label == "small":
        area_min, area_max = 0,               SIZE_SMALL_MAX
    elif size_label == "medium":
        area_min, area_max = SIZE_SMALL_MAX,  SIZE_MEDIUM_MAX
    else:
        area_min, area_max = SIZE_MEDIUM_MAX, float("inf")

    all_scores, all_tp, n_gt_total = [], [], 0

    for img in per_image_data:
        gt   = np.array(img["gt_boxes"],   dtype=np.float32).reshape(-1, 4)
        pred = np.array(img["pred_boxes"],  dtype=np.float32).reshape(-1, 4)
        scr  = np.array(img["pred_scores"], dtype=np.float32)

        # Filter GT to size class
        if len(gt) > 0:
            areas  = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
            gt_sel = gt[(areas >= area_min) & (areas < area_max)]
        else:
            gt_sel = gt

        n_gt_total += len(gt_sel)
        if len(pred) == 0:
            continue

        tp = np.zeros(len(pred), dtype=bool)
        if len(gt_sel) > 0:
            iou        = box_iou_matrix(pred, gt_sel)
            matched_gt = set()
            for i in range(len(pred)):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= iou_threshold and j not in matched_gt:
                    tp[i] = True
                    matched_gt.add(j)

        all_scores.extend(scr.tolist())
        all_tp.extend(tp.tolist())

    if not all_scores or n_gt_total == 0:
        return np.nan

    prec, rec = precision_recall_curve(
        np.array(all_scores, dtype=np.float32),
        np.array(all_tp, dtype=bool),
        n_gt_total,
    )
    return compute_ap(prec, rec)


# ─────────────────────────────────────────────────────────────────────────────
# Count statistics
# ─────────────────────────────────────────────────────────────────────────────

def count_agreement_stats(per_image_data: list[dict], conf_threshold: float) -> dict:
    """
    Compute agreement metrics between predicted HF count and GT count
    per image.
    """
    gt_counts: list[float] = []
    pd_counts: list[float] = []

    for img in per_image_data:
        gt  = np.array(img["gt_boxes"],   dtype=np.float32).reshape(-1, 4)
        scr = np.array(img["pred_scores"], dtype=np.float32)
        n_pred_above = int((scr >= conf_threshold).sum())
        gt_counts.append(float(len(gt)))
        pd_counts.append(float(n_pred_above))

    gt_arr = np.array(gt_counts)
    pd_arr = np.array(pd_counts)
    diff   = pd_arr - gt_arr

    r_pearson,  p_pearson  = scipy_stats.pearsonr(gt_arr, pd_arr)
    r_spearman, p_spearman = scipy_stats.spearmanr(gt_arr, pd_arr)

    return {
        "pearson_r":     float(r_pearson),
        "pearson_p":     float(p_pearson),
        "spearman_r":    float(r_spearman),
        "spearman_p":    float(p_spearman),
        "mae":           float(np.mean(np.abs(diff))),
        "rmse":          float(np.sqrt(np.mean(diff ** 2))),
        "medae":         float(np.median(np.abs(diff))),
        "mean_diff":     float(np.mean(diff)),
        "std_diff":      float(np.std(diff)),
        "gt_counts":     gt_arr,
        "pd_counts":     pd_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DetectionEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class DetectionEvaluator:
    """Accumulates predictions and computes all paper-grade metrics."""

    def __init__(self) -> None:
        self.per_image:   list[dict] = []
        self.all_gt_count: int = 0

    def update(
        self,
        image_id:    str,
        pred_boxes:  np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes:    np.ndarray,
    ) -> None:
        order = np.argsort(-pred_scores)
        self.per_image.append({
            "image_id":    image_id,
            "pred_boxes":  pred_boxes[order].tolist(),
            "pred_scores": pred_scores[order].tolist(),
            "gt_boxes":    gt_boxes.tolist(),
        })
        self.all_gt_count += len(gt_boxes)

    # ── Internal match helper ─────────────────────────────────────────────────

    def _match_at_iou(self, iou_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        all_s, all_tp = [], []
        for img in self.per_image:
            s, tp = _match_image_at_iou(img, iou_threshold)
            all_s.extend(s.tolist())
            all_tp.extend(tp.tolist())
        return (
            np.array(all_s,  dtype=np.float32),
            np.array(all_tp, dtype=bool),
        )

    # ── FROC data ─────────────────────────────────────────────────────────────

    def froc_data(self) -> tuple[np.ndarray, np.ndarray]:
        n_img = len(self.per_image)
        if n_img == 0 or self.all_gt_count == 0:
            return np.array([]), np.array([])

        scores, tp_flags = self._match_at_iou(0.50)
        if len(scores) == 0:
            return np.array([]), np.array([])

        order     = np.argsort(-scores)
        tp_sorted = tp_flags[order].astype(float).cumsum()
        fp_sorted = (~tp_flags[order]).astype(float).cumsum()

        sensitivity  = tp_sorted / self.all_gt_count
        fp_per_image = fp_sorted / n_img
        return fp_per_image, sensitivity

    # ── Per-image stats at fixed threshold ────────────────────────────────────

    def per_image_stats(self, iou_threshold: float = 0.50, conf_threshold: float = 0.5) -> list[dict]:
        rows = []
        for img in self.per_image:
            gt   = np.array(img["gt_boxes"],   dtype=np.float32).reshape(-1, 4)
            pred = np.array(img["pred_boxes"],  dtype=np.float32).reshape(-1, 4)
            scr  = np.array(img["pred_scores"], dtype=np.float32)

            # Filter by confidence
            mask = scr >= conf_threshold
            pred = pred[mask]
            scr  = scr[mask]

            tp = 0
            if len(pred) > 0 and len(gt) > 0:
                iou        = box_iou_matrix(pred, gt)
                matched_gt = set()
                for i in range(len(pred)):
                    j = int(np.argmax(iou[i]))
                    if iou[i, j] >= iou_threshold and j not in matched_gt:
                        tp += 1
                        matched_gt.add(j)

            fp = max(0, len(pred) - tp)
            fn = max(0, len(gt) - tp)
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(len(gt), 1)

            rows.append({
                "image_id": img["image_id"],
                "n_gt":     len(gt),
                "n_pred":   len(pred),
                "tp":       tp,
                "fp":       fp,
                "fn":       fn,
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
            })
        return rows

    # ── Main compute ──────────────────────────────────────────────────────────

    def compute(self, n_bootstrap: int = 1000) -> dict:
        if not self.per_image or self.all_gt_count == 0:
            warnings.warn("No data or no GT boxes — metrics undefined.")
            return {}

        # ── COCO AP across IoU thresholds ─────────────────────────────────────
        iou_thresholds = np.arange(0.50, 1.00, 0.05)
        aps = []
        for iou_t in iou_thresholds:
            s, tp = self._match_at_iou(float(iou_t))
            if len(s) == 0:
                aps.append(0.0)
                continue
            prec, rec = precision_recall_curve(s, tp, self.all_gt_count)
            aps.append(compute_ap(prec, rec))

        ap50      = float(aps[0])
        ap75      = float(aps[5])
        map_50_95 = float(np.mean(aps))

        # ── PR curve at IoU=0.50 ─────────────────────────────────────────────
        s50, tp50     = self._match_at_iou(0.50)
        prec_arr, rec_arr = precision_recall_curve(s50, tp50, self.all_gt_count)
        prec_opt, rec_opt, f1_opt, thr_opt = f1_optimal_threshold(s50, tp50, self.all_gt_count)

        # ── Average Recall ────────────────────────────────────────────────────
        ar1   = average_recall_at_k(self.per_image, iou_threshold=0.50, max_dets=1)
        ar10  = average_recall_at_k(self.per_image, iou_threshold=0.50, max_dets=10)
        ar100 = average_recall_at_k(self.per_image, iou_threshold=0.50, max_dets=100)

        # ── Size-based AP ─────────────────────────────────────────────────────
        ap_small  = _ap_for_size_class(self.per_image, "small")
        ap_medium = _ap_for_size_class(self.per_image, "medium")
        ap_large  = _ap_for_size_class(self.per_image, "large")

        # ── FROC ──────────────────────────────────────────────────────────────
        fp_per_img, sensitivity = self.froc_data()
        froc_m = compute_froc_metrics(fp_per_img, sensitivity)

        # ── Bootstrap CI ──────────────────────────────────────────────────────
        print(f"  Computing bootstrap 95 % CI ({n_bootstrap} resamples) …")
        ci_ap50  = bootstrap_ci_ap(self.per_image, iou_threshold=0.50, n_bootstrap=n_bootstrap)
        ci_prf   = bootstrap_ci_prf(self.per_image, thr_opt, iou_threshold=0.50, n_bootstrap=n_bootstrap)

        return {
            "ap50":            ap50,
            "ap75":            ap75,
            "map_50_95":       map_50_95,
            "ap_small":        ap_small,
            "ap_medium":       ap_medium,
            "ap_large":        ap_large,
            "ar1":             ar1,
            "ar10":            ar10,
            "ar100":           ar100,
            "precision":       prec_opt,
            "recall":          rec_opt,
            "f1":              f1_opt,
            "opt_threshold":   thr_opt,
            "ci_ap50":         ci_ap50,
            "ci_precision":    ci_prf["precision"],
            "ci_recall":       ci_prf["recall"],
            "ci_f1":           ci_prf["f1"],
            "n_images":        len(self.per_image),
            "n_gt_total":      self.all_gt_count,
            "_pr_prec":        prec_arr,
            "_pr_rec":         rec_arr,
            "_froc_fp":        fp_per_img,
            "_froc_sens":      sensitivity,
            **froc_m,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(m: dict) -> None:
    sep = "=" * 56

    def _ci(lo, hi):
        if np.isnan(lo) or np.isnan(hi):
            return ""
        return f"  [95%CI {lo:.4f}–{hi:.4f}]"

    print(f"\n{sep}")
    print("  HF Detection Metrics — Faster R-CNN")
    print(sep)
    print(f"  {'AP @ IoU=0.50':<32} {m['ap50']:.4f}{_ci(*m['ci_ap50'])}")
    print(f"  {'AP @ IoU=0.75':<32} {m['ap75']:.4f}")
    print(f"  {'mAP @ [.50:.05:.95]':<32} {m['map_50_95']:.4f}")
    print(f"  {'AP_small (<32²)':<32} {m['ap_small']:.4f}")
    print(f"  {'AP_medium (32²–96²)':<32} {m['ap_medium']:.4f}" if not np.isnan(m['ap_medium']) else f"  {'AP_medium (32²–96²)':<32} —")
    print(f"  {'AP_large (≥96²)':<32} {m['ap_large']:.4f}" if not np.isnan(m['ap_large']) else f"  {'AP_large (≥96²)':<32} —")
    print(f"  {'AR @ 1 det':<32} {m['ar1']:.4f}")
    print(f"  {'AR @ 10 dets':<32} {m['ar10']:.4f}")
    print(f"  {'AR @ 100 dets':<32} {m['ar100']:.4f}")
    print("-" * 56)
    print(f"  {'Precision (opt F1)':<32} {m['precision']:.4f}{_ci(*m['ci_precision'])}")
    print(f"  {'Recall    (opt F1)':<32} {m['recall']:.4f}{_ci(*m['ci_recall'])}")
    print(f"  {'F1        (opt F1)':<32} {m['f1']:.4f}{_ci(*m['ci_f1'])}")
    print(f"  {'Conf threshold (opt F1)':<32} {m['opt_threshold']:.4f}")
    print("-" * 56)
    print(f"  {'CPM':<32} {m['cpm']:.4f}")
    print(f"  {'pAUC-FROC [0.125–8 FP/img]':<32} {m['pauc_froc']:.4f}")
    for fp_t in FROC_FP_TARGETS:
        k = f"sens_fp{_key(fp_t)}"
        print(f"  {'Sens @ '+str(fp_t)+' FP/img':<32} {m[k]:.4f}")
    print("-" * 56)
    print(f"  {'Images evaluated':<32} {m['n_images']}")
    print(f"  {'Total GT boxes':<32} {m['n_gt_total']}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(m: dict, count_stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _ci_str(lo, hi):
        return f"{lo:.4f}" if not np.isnan(lo) else ""

    rows = [
        ("ap50",               m["ap50"],          m["ci_ap50"][0],         m["ci_ap50"][1]),
        ("ap75",               m["ap75"],           np.nan,                  np.nan),
        ("map_50_95",          m["map_50_95"],      np.nan,                  np.nan),
        ("ap_small",           m["ap_small"],       np.nan,                  np.nan),
        ("ap_medium",          m["ap_medium"],      np.nan,                  np.nan),
        ("ap_large",           m["ap_large"],       np.nan,                  np.nan),
        ("ar1",                m["ar1"],            np.nan,                  np.nan),
        ("ar10",               m["ar10"],           np.nan,                  np.nan),
        ("ar100",              m["ar100"],          np.nan,                  np.nan),
        ("precision_opt_f1",   m["precision"],      m["ci_precision"][0],    m["ci_precision"][1]),
        ("recall_opt_f1",      m["recall"],         m["ci_recall"][0],       m["ci_recall"][1]),
        ("f1_opt",             m["f1"],             m["ci_f1"][0],           m["ci_f1"][1]),
        ("conf_threshold_opt", m["opt_threshold"],  np.nan,                  np.nan),
        ("cpm",                m["cpm"],            np.nan,                  np.nan),
        ("pauc_froc",          m["pauc_froc"],      np.nan,                  np.nan),
    ]
    for fp_t in FROC_FP_TARGETS:
        k = f"sens_fp{_key(fp_t)}"
        rows.append((k, m[k], np.nan, np.nan))

    rows += [
        ("count_pearson_r",    count_stats["pearson_r"],  np.nan, np.nan),
        ("count_spearman_r",   count_stats["spearman_r"], np.nan, np.nan),
        ("count_mae",          count_stats["mae"],        np.nan, np.nan),
        ("count_rmse",         count_stats["rmse"],       np.nan, np.nan),
        ("count_medae",        count_stats["medae"],      np.nan, np.nan),
        ("n_images",           m["n_images"],             np.nan, np.nan),
        ("n_gt_total",         m["n_gt_total"],           np.nan, np.nan),
    ]

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_lo", "ci_hi"])
        for name, val, lo, hi in rows:
            v_str  = f"{float(val):.4f}"   if not np.isnan(float(val))  else ""
            lo_str = f"{float(lo):.4f}"    if not np.isnan(float(lo))   else ""
            hi_str = f"{float(hi):.4f}"    if not np.isnan(float(hi))   else ""
            w.writerow([name, v_str, lo_str, hi_str])

    print(f"  Saved: {output_path}")


def save_per_image_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "n_gt", "n_pred", "tp", "fp", "fn", "precision", "recall"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def save_latex(m: dict, count_stats: dict, n_images: int, output_path: Path) -> None:
    def _f(v, d=4):
        return f"{float(v):.{d}f}" if not np.isnan(float(v)) else "—"

    def _ci(lo, hi, d=4):
        if np.isnan(float(lo)) or np.isnan(float(hi)):
            return ""
        return r" \scriptsize{[" + f"{float(lo):.{d}f}–{float(hi):.{d}f}" + r"]}"

    lines = [
        r"% ─────────────────────────────────────────────────────────────────",
        r"% HF Detection results — auto-generated by evaluate_detection.py",
        r"% ─────────────────────────────────────────────────────────────────",
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{HF detection performance of Faster R-CNN on the RVO-Lesion "
            f"test set (n\\,=\\,{n_images} images). "
            r"Metrics in brackets: bootstrap 95\,\%\,CI "
            r"(1\,000 image-level resamples). "
            r"CPM: Competition Performance Metric "
            r"(mean sensitivity at 7 FROC operating points). "
            r"pAUC-FROC: normalised partial AUC$_{\text{FROC}}$ "
            r"[0.125--8\,FP/image].}"
        ),
        r"\label{tab:detection_results}",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\hline",
        r"\multicolumn{2}{l}{\textit{COCO-style Average Precision}} \\",
        f"AP @ IoU=0.50 & {_f(m['ap50'])}{_ci(*m['ci_ap50'])} \\\\",
        f"AP @ IoU=0.75 & {_f(m['ap75'])} \\\\",
        f"mAP @ [.50:.05:.95] & {_f(m['map_50_95'])} \\\\",
        f"AP$_{{\\text{{small}}}}$ (area$<$32$^2$) & {_f(m['ap_small'])} \\\\",
        r"\hline",
        r"\multicolumn{2}{l}{\textit{Average Recall}} \\",
        f"AR @ 1 detection & {_f(m['ar1'])} \\\\",
        f"AR @ 10 detections & {_f(m['ar10'])} \\\\",
        f"AR @ 100 detections & {_f(m['ar100'])} \\\\",
        r"\hline",
        r"\multicolumn{2}{l}{\textit{At optimal F1 threshold (conf=}"
        + f"{m['opt_threshold']:.3f}" + r"\textit{)}} \\",
        f"Precision & {_f(m['precision'])}{_ci(*m['ci_precision'])} \\\\",
        f"Recall    & {_f(m['recall'])}{_ci(*m['ci_recall'])} \\\\",
        f"F1 score  & {_f(m['f1'])}{_ci(*m['ci_f1'])} \\\\",
        r"\hline",
        r"\multicolumn{2}{l}{\textit{FROC analysis}} \\",
        f"CPM & {_f(m['cpm'])} \\\\",
        f"pAUC-FROC [0.125--8\\ FP/img] & {_f(m['pauc_froc'])} \\\\",
    ]
    for fp_t in FROC_FP_TARGETS:
        k = f"sens_fp{_key(fp_t)}"
        lines.append(f"Sensitivity @ {fp_t}\\,FP/img & {_f(m[k])} \\\\")

    lines += [
        r"\hline",
        r"\multicolumn{2}{l}{\textit{Count agreement (pred vs GT HF/image)}} \\",
        f"Pearson $r$ & {_f(count_stats['pearson_r'])} \\\\",
        f"Spearman $\\rho$ & {_f(count_stats['spearman_r'])} \\\\",
        f"MAE & {_f(count_stats['mae'])} \\\\",
        f"RMSE & {_f(count_stats['rmse'])} \\\\",
        f"MedAE & {_f(count_stats['medae'])} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    prec: np.ndarray,
    rec:  np.ndarray,
    ap50: float,
    ci_ap50: tuple[float, float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(rec, prec, where="post", color="#2196F3", linewidth=2,
            label=f"AP@0.50 = {ap50:.4f}")
    ax.fill_between(rec, prec, alpha=0.15, color="#2196F3", step="post")

    ci_lo, ci_hi = ci_ap50
    if not np.isnan(ci_lo):
        ax.text(0.5, 0.06,
                f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]",
                transform=ax.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Precision–Recall Curve — HF Detection (IoU=0.50)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_froc(
    fp_per_image: np.ndarray,
    sensitivity:  np.ndarray,
    froc_m:       dict,
    output_path:  Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(fp_per_image, sensitivity,
            color="#E91E63", linewidth=2.5, label="FROC curve")
    ax.fill_between(fp_per_image, sensitivity, alpha=0.12, color="#E91E63")

    # Mark standard operating points
    for fp_t in FROC_FP_TARGETS:
        k = f"sens_fp{_key(fp_t)}"
        s = froc_m.get(k, np.nan)
        if not np.isnan(s):
            ax.scatter([fp_t], [s], s=70, zorder=5, color="#C62828",
                       marker="o", edgecolors="white", linewidths=1.5)
            ax.annotate(
                f"{fp_t}\n{s:.3f}",
                xy=(fp_t, s),
                xytext=(fp_t + 0.1, s - 0.05),
                fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
            )

    cpm = froc_m.get("cpm", np.nan)
    pauc = froc_m.get("pauc_froc", np.nan)
    legend_txt = f"CPM = {cpm:.4f}    pAUC-FROC = {pauc:.4f}"
    ax.text(0.97, 0.04, legend_txt, transform=ax.transAxes,
            ha="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("Average False Positives per Image", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)",                 fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Free-Response ROC (FROC) Curve — HF Detection", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_score_distribution(
    per_image_data: list[dict],
    conf_threshold:  float,
    iou_threshold:   float = 0.50,
    output_path:     Path  = None,
) -> None:
    """TP vs FP confidence-score histograms."""
    tp_scores: list[float] = []
    fp_scores: list[float] = []

    for img in per_image_data:
        gt   = np.array(img["gt_boxes"],   dtype=np.float32).reshape(-1, 4)
        pred = np.array(img["pred_boxes"],  dtype=np.float32).reshape(-1, 4)
        scr  = np.array(img["pred_scores"], dtype=np.float32)

        if len(pred) == 0:
            continue

        tp_flags = np.zeros(len(pred), dtype=bool)
        if len(gt) > 0:
            iou        = box_iou_matrix(pred, gt)
            matched_gt = set()
            for i in range(len(pred)):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= iou_threshold and j not in matched_gt:
                    tp_flags[i] = True
                    matched_gt.add(j)

        tp_scores.extend(scr[tp_flags].tolist())
        fp_scores.extend(scr[~tp_flags].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bins = np.linspace(0, 1, 41)
    axes[0].hist(tp_scores, bins=bins, color="#4CAF50", alpha=0.8,
                 edgecolor="white", linewidth=0.4, label=f"TP (n={len(tp_scores)})")
    axes[0].hist(fp_scores, bins=bins, color="#F44336", alpha=0.6,
                 edgecolor="white", linewidth=0.4, label=f"FP (n={len(fp_scores)})")
    axes[0].axvline(conf_threshold, linestyle="--", color="black", linewidth=1.5,
                    label=f"Opt threshold={conf_threshold:.3f}")
    axes[0].set_xlabel("Confidence Score", fontsize=11)
    axes[0].set_ylabel("Count",            fontsize=11)
    axes[0].set_title("TP vs FP Score Distribution", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        x_range = np.linspace(0, 1, 300)
        if len(tp_scores) > 5:
            kde_tp = gaussian_kde(tp_scores, bw_method=0.1)
            axes[1].fill_between(x_range, kde_tp(x_range), alpha=0.6,
                                 color="#4CAF50", label="TP density")
        if len(fp_scores) > 5:
            kde_fp = gaussian_kde(fp_scores, bw_method=0.1)
            axes[1].fill_between(x_range, kde_fp(x_range), alpha=0.5,
                                 color="#F44336", label="FP density")
    except Exception:
        axes[1].hist(tp_scores, bins=bins, density=True, color="#4CAF50", alpha=0.7)
        axes[1].hist(fp_scores, bins=bins, density=True, color="#F44336", alpha=0.5)

    axes[1].axvline(conf_threshold, linestyle="--", color="black", linewidth=1.5)
    axes[1].set_xlabel("Confidence Score", fontsize=11)
    axes[1].set_ylabel("Density",          fontsize=11)
    axes[1].set_title("Score Density (TP vs FP)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Confidence Score Analysis — Faster R-CNN HF Detection", fontsize=12, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_count_correlation(count_stats: dict, output_path: Path) -> None:
    gt_arr = count_stats["gt_counts"]
    pd_arr = count_stats["pd_counts"]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(gt_arr, pd_arr, s=12, alpha=0.35, color="#2196F3", rasterized=True)

    # Identity line
    max_val = max(gt_arr.max(), pd_arr.max()) + 1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.2,
            alpha=0.6, label="Perfect agreement (y=x)")

    # Regression line
    if len(gt_arr) > 2:
        slope, intercept, r, _, _ = scipy_stats.linregress(gt_arr, pd_arr)
        x_fit = np.linspace(0, max_val, 100)
        ax.plot(x_fit, slope * x_fit + intercept, color="#E91E63", linewidth=1.8,
                label=f"Regression (r={count_stats['pearson_r']:.3f})")

    r_s = count_stats["spearman_r"]
    mae = count_stats["mae"]
    ax.text(
        0.05, 0.95,
        f"Pearson $r$={count_stats['pearson_r']:.3f}\n"
        f"Spearman $\\rho$={r_s:.3f}\n"
        f"MAE={mae:.2f}   RMSE={count_stats['rmse']:.2f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("GT HF Count per Image",         fontsize=12)
    ax.set_ylabel("Predicted HF Count per Image",  fontsize=12)
    ax.set_title("HF Count Agreement — Predicted vs Ground Truth", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_bland_altman(count_stats: dict, output_path: Path) -> None:
    gt_arr   = count_stats["gt_counts"]
    pd_arr   = count_stats["pd_counts"]
    mean_arr = (gt_arr + pd_arr) / 2.0
    diff_arr = pd_arr - gt_arr

    mean_d = float(np.mean(diff_arr))
    std_d  = float(np.std(diff_arr))
    loa_lo = mean_d - 1.96 * std_d
    loa_hi = mean_d + 1.96 * std_d

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.scatter(mean_arr, diff_arr, s=12, alpha=0.35, color="#673AB7", rasterized=True)
    ax.axhline(mean_d,  color="#E91E63", linewidth=2,   linestyle="-",
               label=f"Mean diff = {mean_d:+.2f}")
    ax.axhline(loa_hi,  color="#FF9800", linewidth=1.5, linestyle="--",
               label=f"+1.96 SD = {loa_hi:+.2f}")
    ax.axhline(loa_lo,  color="#FF9800", linewidth=1.5, linestyle="--",
               label=f"−1.96 SD = {loa_lo:+.2f}")
    ax.axhline(0,       color="black",   linewidth=0.8, linestyle="-", alpha=0.4)

    ax.fill_between([mean_arr.min(), mean_arr.max()], loa_lo, loa_hi,
                    alpha=0.07, color="#FF9800")

    ax.set_xlabel("Mean of GT and Predicted Count", fontsize=12)
    ax.set_ylabel("Predicted − GT Count",           fontsize=12)
    ax.set_title("Bland–Altman Plot — HF Count Agreement", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_fp_distribution(per_image_stats_rows: list[dict], output_path: Path) -> None:
    fps = [r["fp"] for r in per_image_stats_rows]
    fns = [r["fn"] for r in per_image_stats_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    max_fp = max(fps) if fps else 10
    bins_fp = range(0, min(max_fp + 2, 30))
    axes[0].hist(fps, bins=bins_fp, color="#F44336", alpha=0.82,
                 edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("False Positives per Image", fontsize=11)
    axes[0].set_ylabel("Image Count",               fontsize=11)
    axes[0].set_title(
        f"FP Distribution\nMean={np.mean(fps):.2f}  Median={np.median(fps):.1f}  Max={max(fps)}",
        fontsize=11,
    )
    axes[0].grid(axis="y", alpha=0.3)

    max_fn = max(fns) if fns else 10
    bins_fn = range(0, min(max_fn + 2, 30))
    axes[1].hist(fns, bins=bins_fn, color="#2196F3", alpha=0.82,
                 edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("False Negatives per Image", fontsize=11)
    axes[1].set_ylabel("Image Count",               fontsize=11)
    axes[1].set_title(
        f"FN Distribution\nMean={np.mean(fns):.2f}  Median={np.median(fns):.1f}  Max={max(fns)}",
        fontsize=11,
    )
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("FP and FN Distribution per Image — Faster R-CNN", fontsize=12, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_gt_count_distribution(per_image_data: list[dict], output_path: Path) -> None:
    gt_counts = [len(img["gt_boxes"]) for img in per_image_data]
    pos_counts = [c for c in gt_counts if c > 0]
    n_pos = len(pos_counts)
    n_neg = gt_counts.count(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pie: positive vs negative
    axes[0].pie(
        [n_pos, n_neg],
        labels=[f"Positive\n(HF present)\n{n_pos} ({100*n_pos/len(gt_counts):.1f}%)",
                f"Negative\n(no HF)\n{n_neg} ({100*n_neg/len(gt_counts):.1f}%)"],
        colors=["#FF5722", "#90A4AE"],
        startangle=90, autopct=None,
        textprops=dict(fontsize=10),
        explode=[0.04, 0],
    )
    axes[0].set_title("Positive / Negative Image Split", fontsize=12)

    # Histogram of GT count among positive images
    if pos_counts:
        max_c = max(pos_counts)
        bins  = range(1, min(max_c + 2, 30))
        axes[1].hist(pos_counts, bins=bins, color="#FF5722", alpha=0.85,
                     edgecolor="white", linewidth=0.4)
        axes[1].set_xlabel("GT HF Count (positive images)", fontsize=11)
        axes[1].set_ylabel("Image Count",                   fontsize=11)
        axes[1].set_title(
            f"GT HF Count Distribution\nMean={np.mean(pos_counts):.2f}  "
            f"Median={np.median(pos_counts):.1f}  Max={max_c}",
            fontsize=11,
        )
        axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Test-Set Dataset Statistics — HF Detection", fontsize=12, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_size_ap(m: dict, output_path: Path) -> None:
    """Bar chart comparing AP across COCO size categories."""
    categories  = ["AP@0.50\n(all)", "AP@0.75\n(all)", "mAP\n(all)",
                   "AP_small\n(<32²)", "AP_medium\n(32²–96²)", "AP_large\n(≥96²)"]
    values      = [m["ap50"], m["ap75"], m["map_50_95"],
                   m["ap_small"], m["ap_medium"], m["ap_large"]]
    colors      = ["#2196F3", "#42A5F5", "#90CAF9",
                   "#FF5722", "#FF8A65", "#FFCCBC"]

    fig, ax = plt.subplots(figsize=(10, 5))
    valid_pairs = [(c, v, col) for c, v, col in zip(categories, values, colors)
                   if not np.isnan(float(v))]
    cats, vals, cols = zip(*valid_pairs) if valid_pairs else ([], [], [])

    bars = ax.bar(cats, vals, color=cols, alpha=0.88,
                  edgecolor="black", linewidth=0.5, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Precision", fontsize=11)
    ax.set_title("AP by Metric Type and Box-Size Category — Faster R-CNN", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loading
# ─────────────────────────────────────────────────────────────────────────────

_HF_HALF = 5   # half-width of 10×10 proxy box for point annotations

def _load_gt_from_coco(annotations_json: Path) -> dict[str, np.ndarray]:
    with open(annotations_json, encoding="utf-8") as f:
        coco = json.load(f)
    id_to_stem = {img["id"]: Path(img["file_name"]).stem
                  for img in coco.get("images", [])}
    gt: dict[str, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        stem = id_to_stem.get(ann["image_id"])
        if stem is None:
            continue
        x, y, w, h = ann["bbox"]
        gt[stem].append([x, y, x + w, y + h])
    return {k: np.array(v, dtype=np.float32).reshape(-1, 4) for k, v in gt.items()}


def _load_gt_from_labelme(labelme_dir: Path) -> dict[str, np.ndarray]:
    """
    Parses LabelMe JSONs.
    Point annotations → 10×10 proxy boxes.
    Polygon/rectangle annotations → bounding box.
    """
    gt: dict[str, list] = defaultdict(list)
    for jp in labelme_dir.rglob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        img_h = data.get("imageHeight", 10000)
        img_w = data.get("imageWidth",  10000)

        for shape in data.get("shapes", []):
            if shape.get("label", "").strip().upper() != "HF":
                continue
            stype = shape.get("shape_type", "").lower()
            pts   = shape.get("points", [])

            if stype == "point" and pts:
                cx, cy = float(pts[0][0]), float(pts[0][1])
                box = [
                    max(0.0,          cx - _HF_HALF),
                    max(0.0,          cy - _HF_HALF),
                    min(float(img_w), cx + _HF_HALF),
                    min(float(img_h), cy + _HF_HALF),
                ]
            elif stype in ("rectangle", "polygon", "linestrip") and len(pts) >= 2:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                box = [max(0, min(xs)), max(0, min(ys)),
                       min(img_w, max(xs)), min(img_h, max(ys))]
            else:
                continue

            if box[2] > box[0] and box[3] > box[1]:
                gt[jp.stem].append(box)

    return {k: np.array(v, dtype=np.float32).reshape(-1, 4) for k, v in gt.items()}


def _load_test_stems(test_txt: Path) -> list[str]:
    stems = []
    with open(test_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(Path(line.replace("\\", "/")).stem)
    return stems


def _find_image(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    for p in directory.rglob(stem + ".*"):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    images_dir:       Path,
    test_txt:         Path,
    weights_path:     Optional[Path],
    output_dir:       Path,
    annotations_json: Optional[Path] = None,
    labelme_dir:      Optional[Path] = None,
    conf_threshold:   float = 0.05,
    n_bootstrap:      int   = 1000,
    max_images:       int   = 0,
    resize_eval:      int   = 0,
) -> dict:
    from app.services.detection    import load_detection_model, predict_detections
    from app.services.preprocessor import apply_clahe

    wp = str(weights_path) if weights_path and weights_path.exists() else "weights/hf_frcnn.pt"
    load_detection_model(wp)

    # Load ground truth
    if annotations_json and annotations_json.exists():
        print(f"Loading GT from COCO annotations: {annotations_json}")
        gt_map = _load_gt_from_coco(annotations_json)
    elif labelme_dir and labelme_dir.exists():
        print(f"Loading GT from LabelMe JSONs: {labelme_dir}")
        gt_map = _load_gt_from_labelme(labelme_dir)
    else:
        raise FileNotFoundError(
            "Provide --annotations (COCO JSON) or --labelme-dir to load ground truth."
        )

    stems     = _load_test_stems(test_txt)
    if max_images > 0:
        stems = stems[:max_images]
        print(f"  [--max-images {max_images}] Evaluating on first {len(stems)} images only.")

    evaluator = DetectionEvaluator()
    processed = 0
    skipped   = 0

    print(f"\nEvaluating detection on {len(stems)} test images …")

    for stem in stems:
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            continue

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enhanced = apply_clahe(img_rgb)

        # Optional: downscale before inference so the model's internal ResNet
        # operates on a smaller image. Only downscale — never upscale.
        # GT boxes are scaled proportionally so IoU calculations remain valid.
        scale = 1.0
        gt_boxes_full = gt_map.get(stem, np.zeros((0, 4), dtype=np.float32))
        if resize_eval > 0:
            H_e, W_e = enhanced.shape[:2]
            long_side = max(H_e, W_e)
            if long_side > resize_eval:
                scale    = resize_eval / long_side
                new_w    = max(1, int(W_e * scale))
                new_h    = max(1, int(H_e * scale))
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        det = predict_detections(enhanced, threshold=conf_threshold)

        pred_boxes  = np.array(det["boxes"],  dtype=np.float32).reshape(-1, 4) \
                      if len(det["boxes"]) > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.array(det["scores"], dtype=np.float32).ravel() \
                      if len(det["scores"]) > 0 else np.zeros(0, dtype=np.float32)
        # Scale GT boxes to match the (possibly resized) image coordinate space
        gt_boxes    = (gt_boxes_full * scale).astype(np.float32) if scale != 1.0 \
                      else gt_boxes_full

        evaluator.update(
            image_id    = stem,
            pred_boxes  = pred_boxes,
            pred_scores = pred_scores,
            gt_boxes    = gt_boxes,
        )

        # Explicitly free large per-image arrays so the OS can reclaim RAM.
        # Without this, Python may hold onto hundreds of MB after 200+ images.
        del img_bgr, img_rgb, enhanced, det, pred_boxes, pred_scores, gt_boxes, gt_boxes_full

        processed += 1
        if processed % 10 == 0:
            gc.collect()
        if processed % 100 == 0:
            print(f"  … {processed}/{len(stems)} images processed")

    print(f"  Done — processed {processed}, skipped {skipped}")
    gc.collect()

    # ── Compute all metrics ───────────────────────────────────────────────────
    metrics      = evaluator.compute(n_bootstrap=n_bootstrap)
    per_img_rows = evaluator.per_image_stats(
        iou_threshold   = 0.50,
        conf_threshold  = metrics.get("opt_threshold", 0.5),
    )
    count_stats  = count_agreement_stats(
        evaluator.per_image,
        conf_threshold = metrics.get("opt_threshold", 0.5),
    )

    fp_per_img  = metrics["_froc_fp"]
    sensitivity = metrics["_froc_sens"]
    froc_m      = {k: v for k, v in metrics.items()
                   if k.startswith("sens_fp") or k in ("cpm", "pauc_froc")}

    # ── Write outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    print_summary(metrics)
    save_metrics_csv(metrics, count_stats, output_dir / "detection_metrics.csv")
    save_per_image_csv(per_img_rows,        output_dir / "detection_per_image.csv")
    save_latex(metrics, count_stats, processed, output_dir / "latex_table.tex")

    plot_pr_curve(
        metrics["_pr_prec"], metrics["_pr_rec"],
        metrics["ap50"], metrics["ci_ap50"],
        output_dir / "pr_curve.png",
    )
    if len(fp_per_img) > 0:
        plot_froc(fp_per_img, sensitivity, froc_m, output_dir / "froc_curve.png")
    plot_score_distribution(
        evaluator.per_image, metrics["opt_threshold"],
        output_path = output_dir / "score_distribution.png",
    )
    plot_count_correlation(count_stats,   output_dir / "count_correlation.png")
    plot_bland_altman(count_stats,        output_dir / "bland_altman.png")
    plot_fp_distribution(per_img_rows,    output_dir / "fp_fn_distribution.png")
    plot_gt_count_distribution(evaluator.per_image, output_dir / "gt_count_distribution.png")
    plot_size_ap(metrics,                 output_dir / "size_ap.png")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HF Detection evaluation — Faster R-CNN (publication-grade)"
    )
    p.add_argument("--images-dir",     required=True,  type=Path)
    p.add_argument("--test-txt",       required=True,  type=Path)
    p.add_argument("--weights",        default=None,   type=Path)
    p.add_argument("--annotations",    default=None,   type=Path,
                   help="COCO annotations.json (preferred GT source)")
    p.add_argument("--labelme-dir",    default=None,   type=Path,
                   help="LabelMe JSON directory (fallback GT source)")
    p.add_argument("--conf-threshold", default=0.05,   type=float,
                   help="Min confidence to keep for evaluation (default 0.05)")
    p.add_argument("--output",         default=Path("evaluation/results/detection"), type=Path)
    p.add_argument("--n-bootstrap",    default=1000,   type=int,
                   help="Bootstrap resamples for CI (default 1000)")
    p.add_argument("--max-images",     default=0,      type=int,
                   help="Limit to first N images (0 = all). Use 5 for a quick smoke-test.")
    p.add_argument("--resize-eval",    default=0,      type=int,
                   help="Pre-resize images so long side ≤ this value before inference "
                        "(0 = no resize). E.g. --resize-eval 800 gives 3-5× CPU speedup.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        images_dir       = args.images_dir,
        test_txt         = args.test_txt,
        weights_path     = args.weights,
        output_dir       = args.output,
        annotations_json = args.annotations,
        labelme_dir      = args.labelme_dir,
        conf_threshold   = args.conf_threshold,
        n_bootstrap      = args.n_bootstrap,
        max_images       = args.max_images,
        resize_eval      = args.resize_eval,
    )
