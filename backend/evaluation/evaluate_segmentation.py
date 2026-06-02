"""
Segmentation evaluation — Attention U-Net on RVO-Lesion test split.

Metrics reported
----------------
Per-class  : IoU (Jaccard), Dice (DSC), Precision, Recall, Specificity,
             F1, Pixel Accuracy, HD95 (*), ASD (*)
             Bootstrap 95 % CI for Dice and IoU (1 000 resamples)
Aggregate  : mIoU excl. BG, mIoU incl. BG, FW-IoU, Mean Dice excl. BG,
             Overall Pixel Accuracy, Mean Pixel Accuracy, Cohen's κ

             (* requires scipy; skipped otherwise)

Outputs (written to --output directory)
-----------------------------------------
  segmentation_metrics.csv          per-class + aggregate values with 95 % CI
  segmentation_per_image.csv        per-image per-class Dice scores
  confusion_matrix.png              row-normalised confusion matrix heat-map
  per_class_metrics.png             IoU / Dice grouped bar chart
  violin_dice.png                   per-image Dice violin plots (fg classes)
  class_distribution.png            pixel distribution across classes
  latex_table.tex                   publication-ready LaTeX table with 95 % CI

Usage
-----
  cd backend
  python -m evaluation.evaluate_segmentation \\
      --images-dir  ../dataset/RVO-Lesion/Image_Seg/images \\
      --masks-dir   ../dataset/RVO-Lesion/Image_Seg/masks  \\
      --test-txt    ../dataset/RVO-Lesion/Image_Seg/test.txt \\
      --weights     weights/attention_unet.pth \\
      --output      evaluation/results/segmentation
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["Background", "SRF", "IRF", "ELM", "EZ"]
CLASS_COLORS = [
    "#1a1a1a",  # Background — dark
    "#2196F3",  # SRF — blue
    "#F44336",  # IRF — red
    "#FFC107",  # ELM — amber
    "#4CAF50",  # EZ  — green
]
N_CLASSES  = 5
FOREGROUND = slice(1, None)     # class indices 1-4 (exclude background)

# ─────────────────────────────────────────────────────────────────────────────
# Confusion-matrix accumulator
# ─────────────────────────────────────────────────────────────────────────────

class ConfusionMatrix:
    """Pixel-level accumulator with full metric suite including specificity and κ."""

    def __init__(self, n_classes: int = N_CLASSES):
        self.n   = n_classes
        self.mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        valid = (gt >= 0) & (gt < self.n)
        flat  = self.n * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
        counts = np.bincount(flat, minlength=self.n * self.n)
        self.mat += counts.reshape(self.n, self.n)

    def compute(self) -> dict:
        M       = self.mat.astype(float)
        tp      = np.diag(M)
        row_sum = M.sum(axis=1)     # TP + FN  — GT count per class
        col_sum = M.sum(axis=0)     # TP + FP  — pred count per class
        total   = M.sum()

        # ── Per-class metrics ─────────────────────────────────────────────────
        denom_iou  = row_sum + col_sum - tp
        iou        = np.where(denom_iou  > 0, tp / denom_iou,  np.nan)

        denom_dice = row_sum + col_sum
        dice       = np.where(denom_dice > 0, 2 * tp / denom_dice, np.nan)

        precision  = np.where(col_sum > 0, tp / col_sum, np.nan)
        recall     = np.where(row_sum > 0, tp / row_sum, np.nan)
        f1         = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            np.nan,
        )

        # Specificity = TN / (TN + FP)
        tn          = total - row_sum - col_sum + tp
        fp_count    = col_sum - tp
        specificity = np.where((tn + fp_count) > 0, tn / (tn + fp_count), np.nan)

        px_acc_cls  = np.where(row_sum > 0, tp / row_sum, np.nan)

        # ── Aggregate metrics ─────────────────────────────────────────────────
        pixel_acc    = tp.sum() / total if total > 0 else np.nan
        mean_px_acc  = float(np.nanmean(px_acc_cls))

        freq   = row_sum / total
        fwiou  = float(np.nansum(freq * iou))

        miou_fg      = float(np.nanmean(iou[FOREGROUND]))
        miou_all     = float(np.nanmean(iou))
        mean_dice_fg = float(np.nanmean(dice[FOREGROUND]))

        # Cohen's κ
        p_o   = tp.sum() / total
        p_e   = float(np.sum(row_sum * col_sum) / (total ** 2))
        kappa = float((p_o - p_e) / (1.0 - p_e)) if (1.0 - p_e) > 1e-9 else np.nan

        return {
            "iou":          iou,
            "dice":         dice,
            "precision":    precision,
            "recall":       recall,
            "specificity":  specificity,
            "f1":           f1,
            "px_acc_cls":   px_acc_cls,
            "pixel_acc":    float(pixel_acc),
            "mean_px_acc":  float(mean_px_acc),
            "miou":         miou_fg,
            "miou_with_bg": miou_all,
            "fwiou":        fwiou,
            "mean_dice":    mean_dice_fg,
            "kappa":        kappa,
        }

    def normalized(self) -> np.ndarray:
        M   = self.mat.astype(float)
        row = M.sum(axis=1, keepdims=True)
        return np.where(row > 0, M / row, 0.0)

    def pixel_distribution(self) -> np.ndarray:
        """Fraction of GT pixels per class."""
        row = self.mat.sum(axis=1).astype(float)
        s   = row.sum()
        return row / s if s > 0 else row


# ─────────────────────────────────────────────────────────────────────────────
# Per-image metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dice_per_class_single(pred: np.ndarray, gt: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    """Dice per class for a single image. NaN when both pred & GT are empty."""
    dices = np.full(n, np.nan, dtype=np.float64)
    for c in range(n):
        p_c = pred == c
        g_c = gt   == c
        denom = p_c.sum() + g_c.sum()
        if denom > 0:
            dices[c] = 2.0 * float((p_c & g_c).sum()) / float(denom)
        # If denom == 0: both empty → NaN (class absent; excluded from mean)
    return dices


def _iou_per_class_single(pred: np.ndarray, gt: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    ious = np.full(n, np.nan, dtype=np.float64)
    for c in range(n):
        p_c     = pred == c
        g_c     = gt   == c
        inter   = (p_c & g_c).sum()
        union   = (p_c | g_c).sum()
        if union > 0:
            ious[c] = float(inter) / float(union)
    return ious


# ─────────────────────────────────────────────────────────────────────────────
# Surface distance metrics (requires scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _surface_metrics_binary(
    pred_bin: np.ndarray,
    gt_bin:   np.ndarray,
) -> tuple[float, float]:
    """
    Returns (HD95, ASD) in pixels for a binary mask pair.
    HD95 = 95th-percentile symmetric Hausdorff distance.
    ASD  = mean symmetric surface distance.
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        return np.nan, np.nan

    if not pred_bin.any() and not gt_bin.any():
        return 0.0, 0.0
    if not pred_bin.any() or not gt_bin.any():
        return np.nan, np.nan

    pred_surf = pred_bin ^ binary_erosion(pred_bin)
    gt_surf   = gt_bin   ^ binary_erosion(gt_bin)

    dist_to_gt   = distance_transform_edt(~gt_bin)
    dist_to_pred = distance_transform_edt(~pred_bin)

    d_p2g = dist_to_gt[pred_surf]
    d_g2p = dist_to_pred[gt_surf]

    all_d = np.concatenate([d_p2g, d_g2p])
    hd95  = float(np.percentile(all_d, 95))
    asd   = float((d_p2g.mean() + d_g2p.mean()) / 2.0) if (len(d_p2g) > 0 and len(d_g2p) > 0) else np.nan

    return hd95, asd


def compute_surface_metrics_per_class(
    pred:      np.ndarray,
    gt:        np.ndarray,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (hd95, asd) arrays of length n_classes."""
    hd95 = np.full(n_classes, np.nan)
    asd  = np.full(n_classes, np.nan)
    for c in range(n_classes):
        hd95[c], asd[c] = _surface_metrics_binary(pred == c, gt == c)
    return hd95, asd


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values:      list[float] | np.ndarray,
    n_bootstrap: int   = 1000,
    alpha:       float = 0.05,
    seed:        int   = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95 % CI for the mean of `values`.
    NaN entries are silently dropped.
    Returns (lower, upper).
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(arr) < 2:
        m = arr[0] if len(arr) == 1 else np.nan
        return m, m
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if not (v != v) else "—"   # NaN-safe


def _pct(v: float) -> str:
    """Format as percentage string, e.g. 0.8234 → '82.34'."""
    return f"{v * 100:.2f}" if not (v != v) else "—"


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    half = (hi - lo) / 2.0
    return f"{mean * 100:.2f} ± {half * 100:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(
    metrics:       dict,
    hd95_mean:     Optional[np.ndarray],
    asd_mean:      Optional[np.ndarray],
    ci_dice:       Optional[list[tuple]],   # [(lo, hi) × N_CLASSES]
    ci_iou:        Optional[list[tuple]],
    output_path:   Path,
) -> None:
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        dice_lo, dice_hi = ci_dice[i] if ci_dice else (np.nan, np.nan)
        iou_lo,  iou_hi  = ci_iou[i]  if ci_iou  else (np.nan, np.nan)
        row = {
            "class":        name,
            "dice":         round(float(metrics["dice"][i]),        4) if not np.isnan(metrics["dice"][i])        else "",
            "dice_ci_lo":   round(float(dice_lo),                   4) if not np.isnan(dice_lo)                   else "",
            "dice_ci_hi":   round(float(dice_hi),                   4) if not np.isnan(dice_hi)                   else "",
            "iou":          round(float(metrics["iou"][i]),         4) if not np.isnan(metrics["iou"][i])         else "",
            "iou_ci_lo":    round(float(iou_lo),                    4) if not np.isnan(iou_lo)                    else "",
            "iou_ci_hi":    round(float(iou_hi),                    4) if not np.isnan(iou_hi)                    else "",
            "precision":    round(float(metrics["precision"][i]),   4) if not np.isnan(metrics["precision"][i])   else "",
            "recall":       round(float(metrics["recall"][i]),      4) if not np.isnan(metrics["recall"][i])      else "",
            "specificity":  round(float(metrics["specificity"][i]), 4) if not np.isnan(metrics["specificity"][i]) else "",
            "f1":           round(float(metrics["f1"][i]),          4) if not np.isnan(metrics["f1"][i])          else "",
            "px_acc":       round(float(metrics["px_acc_cls"][i]),  4) if not np.isnan(metrics["px_acc_cls"][i])  else "",
            "hd95":         round(float(hd95_mean[i]), 2) if hd95_mean is not None and not np.isnan(hd95_mean[i]) else "",
            "asd":          round(float(asd_mean[i]),  2) if asd_mean  is not None and not np.isnan(asd_mean[i])  else "",
        }
        rows.append(row)

    fieldnames = [
        "class", "dice", "dice_ci_lo", "dice_ci_hi",
        "iou", "iou_ci_lo", "iou_ci_hi",
        "precision", "recall", "specificity", "f1", "px_acc", "hd95", "asd",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.write("\n# Aggregate\n")
        for name, val in [
            ("pixel_accuracy",       metrics["pixel_acc"]),
            ("mean_pixel_accuracy",  metrics["mean_px_acc"]),
            ("miou_excl_bg",         metrics["miou"]),
            ("miou_incl_bg",         metrics["miou_with_bg"]),
            ("fw_iou",               metrics["fwiou"]),
            ("mean_dice_excl_bg",    metrics["mean_dice"]),
            ("cohens_kappa",         metrics["kappa"]),
        ]:
            f.write(f"{name},{val:.4f}\n")
    print(f"  Saved: {output_path}")


def save_per_image_csv(per_image_dices: list[dict], output_path: Path) -> None:
    if not per_image_dices:
        return
    fieldnames = ["stem"] + [f"dice_{c.lower()}" for c in CLASS_NAMES]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_dices)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    metrics:   dict,
    hd95_mean: Optional[np.ndarray],
    asd_mean:  Optional[np.ndarray],
    ci_dice:   Optional[list[tuple]],
    n_images:  int,
) -> None:
    hd_col  = hd95_mean is not None
    asd_col = asd_mean  is not None
    ci_col  = ci_dice   is not None

    w  = 14
    col_w = 9
    header = (
        f"{'Class':<{w}} {'Dice':>{col_w}} {'IoU':>{col_w}} "
        f"{'Prec':>{col_w}} {'Recall':>{col_w}} {'Spec':>{col_w}} {'F1':>{col_w}}"
    )
    if hd_col:  header += f" {'HD95':>{col_w}}"
    if asd_col: header += f" {'ASD':>{col_w}}"

    sep = "=" * (len(header) + 2)
    print(f"\n{sep}")
    print(f"  Segmentation Metrics — Attention U-Net   (n={n_images} test images)")
    print(sep)
    print(header)
    print("-" * (len(header) + 2))

    for i, name in enumerate(CLASS_NAMES):
        line = (
            f"{name:<{w}} "
            f"{_fmt(metrics['dice'][i]):>{col_w}} "
            f"{_fmt(metrics['iou'][i]):>{col_w}} "
            f"{_fmt(metrics['precision'][i]):>{col_w}} "
            f"{_fmt(metrics['recall'][i]):>{col_w}} "
            f"{_fmt(metrics['specificity'][i]):>{col_w}} "
            f"{_fmt(metrics['f1'][i]):>{col_w}}"
        )
        if hd_col:
            line += f" {_fmt(hd95_mean[i], 2) if not np.isnan(hd95_mean[i]) else '—':>{col_w}}"
        if asd_col:
            line += f" {_fmt(asd_mean[i],  2) if not np.isnan(asd_mean[i])  else '—':>{col_w}}"
        print(line)

    print("-" * (len(header) + 2))

    # Mean Dice: global confusion-matrix estimate; per-class CI from per-image bootstrap
    print(f"  {'Mean Dice (excl BG)':<30} {metrics['mean_dice']:.4f}  [global confusion-matrix]")
    if ci_col:
        for c_idx in range(1, N_CLASSES):
            lo, hi = ci_dice[c_idx]
            if not np.isnan(lo):
                print(f"    {CLASS_NAMES[c_idx]:<26} per-img CI [{lo:.4f}, {hi:.4f}]")

    print(f"  {'mIoU (excl BG)':<30} {metrics['miou']:.4f}")
    print(f"  {'FW-IoU':<30} {metrics['fwiou']:.4f}")
    print(f"  {'Pixel Accuracy':<30} {metrics['pixel_acc']:.4f}")
    print(f"  {'Mean Pixel Accuracy':<30} {metrics['mean_px_acc']:.4f}")
    print(f"  {'Cohen\'s κ':<30} {metrics['kappa']:.4f}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def save_latex(
    metrics:     dict,
    hd95_mean:   Optional[np.ndarray],
    asd_mean:    Optional[np.ndarray],
    ci_dice:     Optional[list[tuple]],
    ci_iou:      Optional[list[tuple]],
    n_images:    int,
    output_path: Path,
) -> None:
    hd_col  = hd95_mean is not None
    asd_col = asd_mean  is not None

    # Number of data columns (after Class column)
    n_data_cols = 6 + int(hd_col) + int(asd_col)
    col_spec    = "l" + "c" * n_data_cols

    header_row = (
        r"\textbf{Class} & "
        r"\textbf{Dice (\%)} & "
        r"\textbf{IoU (\%)} & "
        r"\textbf{Precision (\%)} & "
        r"\textbf{Recall (\%)} & "
        r"\textbf{Spec. (\%)} & "
        r"\textbf{F1 (\%)}"
    )
    if hd_col:  header_row += r" & \textbf{HD\textsubscript{95} (px)}"
    if asd_col: header_row += r" & \textbf{ASD (px)}"
    header_row += r" \\"

    lines = [
        r"% ─────────────────────────────────────────────────────────────",
        r"% Segmentation results — auto-generated by evaluate_segmentation.py",
        r"% ─────────────────────────────────────────────────────────────",
        r"\begin{table*}[ht]",
        r"\centering",
        (
            r"\caption{Segmentation performance of Attention U-Net on the "
            f"RVO-Lesion test set (n\\,=\\,{n_images} images). "
            r"Dice and IoU: global confusion-matrix value\,\% [per-image bootstrap "
            r"95\,\%\,CI, 1\,000 resamples]. "
            r"HD\textsubscript{95}: 95th-percentile Hausdorff distance. "
            r"ASD: average symmetric surface distance. "
            r"Best foreground values are \textbf{bold}.}"
        ),
        r"\label{tab:segmentation_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\hline",
        header_row,
        r"\hline",
    ]

    # Per-class rows
    best_dice = np.nanargmax(metrics["dice"][FOREGROUND]) + 1
    best_iou  = np.nanargmax(metrics["iou"][FOREGROUND])  + 1

    for i, name in enumerate(CLASS_NAMES):
        dice_v  = metrics["dice"][i]
        iou_v   = metrics["iou"][i]
        prec_v  = metrics["precision"][i]
        rec_v   = metrics["recall"][i]
        spec_v  = metrics["specificity"][i]
        f1_v    = metrics["f1"][i]

        # Format Dice with CI.
        # The confusion-matrix Dice and the per-image bootstrap CI are different
        # estimators; we report both values explicitly rather than ±half-width
        # (which would be misleading when the bar sits outside the CI).
        if ci_dice and not np.isnan(dice_v):
            lo, hi   = ci_dice[i]
            if not (np.isnan(lo) or np.isnan(hi)):
                dice_str = f"{dice_v*100:.2f} [{lo*100:.2f}--{hi*100:.2f}]"
            else:
                dice_str = _pct(dice_v)
        else:
            dice_str = _pct(dice_v)

        if ci_iou and not np.isnan(iou_v):
            lo, hi  = ci_iou[i]
            if not (np.isnan(lo) or np.isnan(hi)):
                iou_str = f"{iou_v*100:.2f} [{lo*100:.2f}--{hi*100:.2f}]"
            else:
                iou_str = _pct(iou_v)
        else:
            iou_str = _pct(iou_v)

        # Bold best fg class
        if i == best_dice and i > 0:
            dice_str = r"\textbf{" + dice_str + r"}"
        if i == best_iou and i > 0:
            iou_str = r"\textbf{" + iou_str + r"}"

        vals = [dice_str, iou_str, _pct(prec_v), _pct(rec_v), _pct(spec_v), _pct(f1_v)]
        if hd_col:
            vals.append(_fmt(hd95_mean[i], 2) if not np.isnan(hd95_mean[i]) else "—")
        if asd_col:
            vals.append(_fmt(asd_mean[i],  2) if not np.isnan(asd_mean[i])  else "—")

        lines.append(f"{name} & " + " & ".join(vals) + r" \\")

    # Aggregate rows
    lines.append(r"\hline")
    lines.append(
        f"mDice (excl.\\ BG) & "
        r"\multicolumn{" + str(n_data_cols) + r"}{c}{"
        + f"{metrics['mean_dice']*100:.2f}" + r"\,\%} \\"
    )
    lines.append(
        f"mIoU (excl.\\ BG) & "
        r"\multicolumn{" + str(n_data_cols) + r"}{c}{"
        + f"{metrics['miou']*100:.2f}" + r"\,\%} \\"
    )
    lines.append(
        f"FW-IoU & "
        r"\multicolumn{" + str(n_data_cols) + r"}{c}{"
        + f"{metrics['fwiou']*100:.2f}" + r"\,\%} \\"
    )
    lines.append(
        f"Pixel Accuracy & "
        r"\multicolumn{" + str(n_data_cols) + r"}{c}{"
        + f"{metrics['pixel_acc']*100:.2f}" + r"\,\%} \\"
    )
    lines.append(
        r"Cohen's $\kappa$ & "
        r"\multicolumn{" + str(n_data_cols) + r"}{c}{"
        + f"{metrics['kappa']:.4f}" + r"} \\"
    )
    lines += [
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_normalized: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=10)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class",      fontsize=11)
    ax.set_title("Normalised Confusion Matrix — Attention U-Net", fontsize=12, pad=12)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val   = cm_normalized[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_class_metrics(
    metrics:     dict,
    ci_dice:     Optional[list[tuple]],
    ci_iou:      Optional[list[tuple]],
    output_path: Path,
) -> None:
    fg_names  = CLASS_NAMES[1:]
    iou_vals  = [float(metrics["iou"][i + 1])  for i in range(4)]
    dice_vals = [float(metrics["dice"][i + 1]) for i in range(4)]
    prec_vals = [float(metrics["precision"][i + 1]) for i in range(4)]
    rec_vals  = [float(metrics["recall"][i + 1])    for i in range(4)]

    dice_err = None
    iou_err  = None
    if ci_dice:
        # Clamp to ≥ 0: confusion-matrix Dice and per-image bootstrap CI are
        # different estimators so the bar height may sit outside the CI bounds.
        dice_err = np.array([
            [max(0.0, v - ci_dice[i+1][0]), max(0.0, ci_dice[i+1][1] - v)]
            for i, v in enumerate(dice_vals)
        ]).T
    if ci_iou:
        iou_err = np.array([
            [max(0.0, v - ci_iou[i+1][0]),  max(0.0, ci_iou[i+1][1]  - v)]
            for i, v in enumerate(iou_vals)
        ]).T

    x     = np.arange(len(fg_names))
    width = 0.2
    colors = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]
    labels = ["IoU", "Dice", "Precision", "Recall"]
    data   = [iou_vals, dice_vals, prec_vals, rec_vals]
    errs   = [iou_err,  dice_err,  None,      None]

    fig, ax = plt.subplots(figsize=(11, 5))

    for k, (lbl, vals, err, col) in enumerate(zip(labels, data, errs, colors)):
        offset = (k - 1.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=lbl, color=col, alpha=0.88,
            yerr=err, capsize=3, error_kw=dict(elinewidth=1, capthick=1, ecolor="black"),
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(fg_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-Class Segmentation Metrics — Attention U-Net", fontsize=12, pad=10)
    ax.axhline(metrics["miou"],       linestyle="--", color="#4878d0", alpha=0.5, linewidth=1.2,
               label=f"mIoU={metrics['miou']:.3f}")
    ax.axhline(metrics["mean_dice"],  linestyle="--", color="#ee854a", alpha=0.5, linewidth=1.2,
               label=f"mDice={metrics['mean_dice']:.3f}")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_violin_dice(
    per_image_dices: list[np.ndarray],
    output_path:     Path,
) -> None:
    """Violin + strip plot of per-image Dice for each foreground class."""
    if not per_image_dices:
        return

    arr = np.array(per_image_dices, dtype=np.float64)   # (N_images, N_CLASSES)
    fg_data  = [arr[:, c][~np.isnan(arr[:, c])] for c in range(1, N_CLASSES)]
    fg_names = CLASS_NAMES[1:]

    fig, ax = plt.subplots(figsize=(9, 5))

    parts = ax.violinplot(
        fg_data, positions=range(len(fg_names)),
        showmedians=True, showextrema=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(CLASS_COLORS[i + 1])
        pc.set_alpha(0.7)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_edgecolor("#333333")
            parts[key].set_linewidth(1.5)

    # Scatter jittered points
    rng = np.random.default_rng(0)
    for i, d in enumerate(fg_data):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(i + jitter, d, s=3, alpha=0.25, color=CLASS_COLORS[i + 1], zorder=2)

    # Means
    for i, d in enumerate(fg_data):
        mean_v = np.nanmean(d)
        ax.scatter(i, mean_v, s=60, marker="D", color="white",
                   edgecolor="#333333", linewidth=1.2, zorder=5, label="Mean" if i == 0 else "")

    ax.set_xticks(range(len(fg_names)))
    ax.set_xticklabels(fg_names, fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Dice (DSC)", fontsize=11)
    ax.set_title("Per-Image Dice Distribution — Attention U-Net", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_class_distribution(pixel_dist: np.ndarray, output_path: Path) -> None:
    """Pie + bar chart of pixel distribution across classes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    pct_labels = [f"{CLASS_NAMES[i]}\n{pixel_dist[i]*100:.2f}%" for i in range(N_CLASSES)]
    explode    = [0.03] * N_CLASSES
    wedges, texts = axes[0].pie(
        pixel_dist, labels=pct_labels, colors=CLASS_COLORS,
        explode=explode, startangle=140,
        textprops=dict(fontsize=9),
    )
    axes[0].set_title("Pixel Class Distribution", fontsize=12)

    # Bar chart (log scale for class imbalance)
    bars = axes[1].bar(CLASS_NAMES, pixel_dist * 100, color=CLASS_COLORS, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("% of total pixels (log scale)", fontsize=10)
    axes[1].set_title("Class Imbalance (log scale)", fontsize=12)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=10)
    for bar in bars:
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.3f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Dataset Pixel Statistics — RVO-Lesion Test Set", fontsize=13, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_surface_metrics(
    hd95_mean: np.ndarray,
    asd_mean:  np.ndarray,
    output_path: Path,
) -> None:
    """Side-by-side bar chart comparing HD95 and ASD per foreground class."""
    fg_names = CLASS_NAMES[1:]
    hd95_fg  = [float(hd95_mean[i + 1]) for i in range(4)]
    asd_fg   = [float(asd_mean[i + 1])  for i in range(4)]

    x     = np.arange(len(fg_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, hd95_fg, width, label="HD₉₅ (px)", color="#9C27B0", alpha=0.82)
    bars2 = ax.bar(x + width/2, asd_fg,  width, label="ASD (px)",  color="#FF9800", alpha=0.82)

    for bar in bars1 + bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(fg_names, fontsize=11)
    ax.set_ylabel("Distance (pixels)", fontsize=11)
    ax.set_title("Surface Distance Metrics per Class — Attention U-Net", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_class_precision_recall(metrics: dict, output_path: Path) -> None:
    """Horizontal precision-recall bar chart per foreground class."""
    fg_names = CLASS_NAMES[1:]
    prec = [float(metrics["precision"][i+1]) for i in range(4)]
    rec  = [float(metrics["recall"][i+1])    for i in range(4)]
    spec = [float(metrics["specificity"][i+1]) for i in range(4)]
    f1   = [float(metrics["f1"][i+1])         for i in range(4)]

    y     = np.arange(len(fg_names))
    height = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (lbl, vals, col) in enumerate(
        zip(
            ["Precision", "Recall", "Specificity", "F1"],
            [prec, rec, spec, f1],
            ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"],
        )
    ):
        offset = (k - 1.5) * height
        ax.barh(y + offset, vals, height=height, label=lbl, color=col, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(fg_names, fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("Precision, Recall, Specificity & F1 — per Class", fontsize=12, pad=10)
    ax.axvline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Data-loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_stems(test_txt: Path) -> list[str]:
    stems = []
    with open(test_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.append(Path(line.replace("\\", "/")).stem)
    return stems


def _find_file(directory: Path, stem: str) -> Optional[Path]:
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
    images_dir:   Path,
    masks_dir:    Path,
    test_txt:     Path,
    weights_path: Optional[Path],
    output_dir:   Path,
    compute_hd:   bool = True,
    n_bootstrap:  int  = 1000,
) -> dict:
    from app.services.preprocessor import apply_clahe
    from app.services.segmentation import load_segmentation_model, predict_segmentation

    wp = str(weights_path) if weights_path and weights_path.exists() else "weights/attention_unet.pth"
    load_segmentation_model(wp)

    stems = _load_stems(test_txt)
    print(f"\nEvaluating segmentation on {len(stems)} test images …")

    confmat          = ConfusionMatrix(N_CLASSES)
    per_image_dices: list[np.ndarray] = []     # (N_images, N_CLASSES)
    per_image_ious:  list[np.ndarray] = []
    per_image_rows:  list[dict] = []

    hd95_accum  = np.zeros(N_CLASSES) if compute_hd else None
    asd_accum   = np.zeros(N_CLASSES) if compute_hd else None
    surf_count  = np.zeros(N_CLASSES)
    processed   = 0
    skipped     = 0

    for stem in stems:
        img_path  = _find_file(images_dir, stem)
        mask_path = _find_file(masks_dir,  stem)

        if img_path is None or mask_path is None:
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped += 1
            continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enhanced = apply_clahe(img_rgb)

        gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            skipped += 1
            continue
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = gt.astype(np.uint8)

        pred = predict_segmentation(enhanced)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        confmat.update(pred, gt)

        # Per-image dice and IoU for bootstrap CI
        dice_img = _dice_per_class_single(pred, gt)
        iou_img  = _iou_per_class_single(pred, gt)
        per_image_dices.append(dice_img)
        per_image_ious.append(iou_img)
        per_image_rows.append({
            "stem": stem,
            **{f"dice_{c.lower()}": round(float(dice_img[i]), 6) if not np.isnan(dice_img[i]) else ""
               for i, c in enumerate(CLASS_NAMES)},
        })

        if compute_hd:
            hd95_img, asd_img = compute_surface_metrics_per_class(pred, gt)
            for c in range(N_CLASSES):
                if not np.isnan(hd95_img[c]):
                    hd95_accum[c] += hd95_img[c]
                    asd_accum[c]  += asd_img[c] if not np.isnan(asd_img[c]) else 0.0
                    surf_count[c] += 1

        processed += 1
        if processed % 100 == 0:
            print(f"  … {processed}/{len(stems)} images processed")

    print(f"  Done — processed {processed}, skipped {skipped}")

    # ── Aggregate confusion-matrix metrics ────────────────────────────────────
    metrics = confmat.compute()

    # ── Surface distance means ────────────────────────────────────────────────
    hd95_mean: Optional[np.ndarray] = None
    asd_mean:  Optional[np.ndarray] = None
    if compute_hd:
        hd95_mean = np.where(surf_count > 0, hd95_accum / np.maximum(surf_count, 1), np.nan)
        asd_mean  = np.where(surf_count > 0, asd_accum  / np.maximum(surf_count, 1), np.nan)

    # ── Bootstrap 95 % CI ─────────────────────────────────────────────────────
    print(f"  Computing bootstrap 95 % CI ({n_bootstrap} resamples) …")
    ci_dice: list[tuple[float, float]] = []
    ci_iou:  list[tuple[float, float]] = []
    dice_arr = np.array(per_image_dices)   # (N, 5)
    iou_arr  = np.array(per_image_ious)

    for c in range(N_CLASSES):
        ci_dice.append(bootstrap_ci(dice_arr[:, c], n_bootstrap=n_bootstrap))
        ci_iou.append(bootstrap_ci(iou_arr[:, c],  n_bootstrap=n_bootstrap))

    # ── Pixel distribution ────────────────────────────────────────────────────
    pixel_dist = confmat.pixel_distribution()

    # ── Outputs ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    print_summary(metrics, hd95_mean, asd_mean, ci_dice, processed)
    save_metrics_csv(metrics, hd95_mean, asd_mean, ci_dice, ci_iou,
                     output_dir / "segmentation_metrics.csv")
    save_per_image_csv(per_image_rows,
                       output_dir / "segmentation_per_image.csv")
    save_latex(metrics, hd95_mean, asd_mean, ci_dice, ci_iou, processed,
               output_dir / "latex_table.tex")

    plot_confusion_matrix(confmat.normalized(),
                          output_dir / "confusion_matrix.png")
    plot_per_class_metrics(metrics, ci_dice, ci_iou,
                           output_dir / "per_class_metrics.png")
    plot_violin_dice(per_image_dices,
                     output_dir / "violin_dice.png")
    plot_class_distribution(pixel_dist,
                            output_dir / "class_distribution.png")
    plot_per_class_precision_recall(metrics,
                                    output_dir / "precision_recall_specificity.png")
    if compute_hd and hd95_mean is not None and asd_mean is not None:
        plot_surface_metrics(hd95_mean, asd_mean,
                             output_dir / "surface_metrics.png")

    return {
        **metrics,
        "hd95_mean": hd95_mean,
        "asd_mean":  asd_mean,
        "ci_dice":   ci_dice,
        "ci_iou":    ci_iou,
        "n_images":  processed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segmentation evaluation — Attention U-Net (publication-grade)"
    )
    p.add_argument("--images-dir",  required=True,  type=Path,
                   help="Directory with raw OCT images")
    p.add_argument("--masks-dir",   required=True,  type=Path,
                   help="Directory with ground-truth mask PNG files")
    p.add_argument("--test-txt",    required=True,  type=Path,
                   help="Path to test.txt split file")
    p.add_argument("--weights",     default=None,   type=Path,
                   help="Path to attention_unet.pth (default: weights/attention_unet.pth)")
    p.add_argument("--output",      default=Path("evaluation/results/segmentation"), type=Path,
                   help="Output directory for all results")
    p.add_argument("--no-hd95",     action="store_true",
                   help="Skip surface distance metrics (faster; no scipy needed)")
    p.add_argument("--n-bootstrap", default=1000, type=int,
                   help="Bootstrap resamples for CI (default 1000)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        images_dir   = args.images_dir,
        masks_dir    = args.masks_dir,
        test_txt     = args.test_txt,
        weights_path = args.weights,
        output_dir   = args.output,
        compute_hd   = not args.no_hd95,
        n_bootstrap  = args.n_bootstrap,
    )
