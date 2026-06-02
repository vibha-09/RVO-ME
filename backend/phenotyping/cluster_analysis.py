"""
Cluster count analysis — finds the optimal k for each feature subspace
using BIC/AIC (GMM) and Silhouette score (K-Means).

Run from project root:
    python backend/phenotyping/cluster_analysis.py
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_PROJECT    = _HERE.parent.parent
DATASET_DIR = _PROJECT / "dataset" / "RVO-Lesion"
MASKS_DIR   = DATASET_DIR / "Image_Seg" / "masks"
TRAIN_TXT   = DATASET_DIR / "Image_Seg" / "train.txt"
LABELME_DIR = DATASET_DIR / "RVO_Lesion_Labelme"

LABEL_SRF, LABEL_IRF, LABEL_ELM, LABEL_EZ = 1, 2, 3, 4
HF_MAX = 50.0

# ── Biomarker extraction (same as training script) ────────────────────────────

def load_stems(txt_path):
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return [Path(l.replace("\\", "/")).stem for l in lines]

def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None: return None
    if mask.ndim == 3: mask = mask[:, :, 0]
    return mask.astype(np.uint8)

def count_hf(stem):
    pid = stem.rsplit("_", 1)[0]
    jp  = LABELME_DIR / pid / f"{stem}.json"
    if not jp.exists(): return 0
    with open(jp, encoding="utf-8") as f:
        data = json.load(f)
    return sum(1 for s in data.get("shapes", [])
               if s.get("label","").upper() == "HF" and s.get("points"))

def extract(mask, hf_count):
    H, W  = mask.shape
    total = float(H * W)
    irf   = np.sum(mask == LABEL_IRF) / total
    srf   = np.sum(mask == LABEL_SRF) / total
    elm   = float(np.sum(np.any(mask == LABEL_ELM, axis=0))) / W
    ez    = float(np.sum(np.any(mask == LABEL_EZ,  axis=0))) / W
    hfd   = min(hf_count / HF_MAX, 1.0)
    return dict(irf_area=irf, srf_area=srf, elm_integrity=elm,
                ez_integrity=ez, hf_density=hfd,
                fluid_total=irf+srf, structure_score=(elm+ez)/2)

# ── Load patient-level data ───────────────────────────────────────────────────

print("Loading biomarkers from training set...")
stems = load_stems(TRAIN_TXT)
rows  = []
for i, stem in enumerate(stems):
    mp = MASKS_DIR / f"{stem}.png"
    if not mp.exists(): continue
    mask = load_mask(mp)
    if mask is None: continue
    rows.append({"patient_id": stem.rsplit("_",1)[0], **extract(mask, count_hf(stem))})
    if i % 500 == 0: print(f"  {i}/{len(stems)}...", end="\r")

FEATURES = ["irf_area","srf_area","elm_integrity","ez_integrity",
            "hf_density","fluid_total","structure_score"]

df = (pd.DataFrame(rows)
        .groupby("patient_id", as_index=False)
        .agg({**{f:"mean" for f in FEATURES}}))
print(f"\nPatients loaded: {len(df)}")

# ── Helper: score k candidates ────────────────────────────────────────────────

def analyse(name, X_raw, k_range=range(2, 9)):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"\n{'='*60}")
    print(f"  {name}  ({X_raw.shape[1]} features, {len(X)} patients)")
    print(f"{'='*60}")
    print(f"  {'k':>3}  {'BIC':>10}  {'AIC':>10}  {'Silhouette':>12}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*12}")

    best_bic, best_sil = {}, {}
    bic_vals, sil_vals = [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=20, random_state=42, max_iter=500)
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)

        labels = gmm.predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0

        bic_vals.append(bic)
        sil_vals.append(sil)
        print(f"  {k:>3}  {bic:>10.1f}  {aic:>10.1f}  {sil:>12.4f}")

    # Best BIC = lowest; best silhouette = highest
    best_k_bic = k_range[int(np.argmin(bic_vals))]
    best_k_sil = k_range[int(np.argmax(sil_vals))]

    print(f"\n  >> Best k by BIC:        {best_k_bic}")
    print(f"  >> Best k by Silhouette: {best_k_sil}")

    # Show what BIC "elbow" looks like
    deltas = [bic_vals[i] - bic_vals[i-1] for i in range(1, len(bic_vals))]
    elbow_idx = int(np.argmin(np.diff(deltas))) + 1 if len(deltas) > 1 else 0
    elbow_k   = list(k_range)[elbow_idx]
    print(f"  >> BIC elbow at k:        {elbow_k}")

    return best_k_bic, best_k_sil

# ── Run analysis on each feature subspace ────────────────────────────────────

print("\n\n" + "="*60)
print("  ANALYSIS 1: ALL BIOMARKERS (overall phenotyping)")
print("="*60)
analyse("ALL 7 FEATURES", df[FEATURES].values)

analyse(
    "FLUID ONLY — SRF + IRF (treatment planning)",
    df[["srf_area", "irf_area"]].values
)

analyse(
    "STRUCTURE ONLY — ELM + EZ (prognosis)",
    df[["elm_integrity", "ez_integrity"]].values
)

analyse(
    "INFLAMMATION ONLY — HF density (monitoring)",
    df[["hf_density"]].values,
    k_range=range(2, 6)
)

# ── Distribution summary ──────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  FEATURE DISTRIBUTIONS (training patients)")
print("="*60)
desc = df[FEATURES].describe().loc[["mean","std","min","25%","50%","75%","max"]]
print(desc.round(5).to_string())
