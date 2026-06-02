"""
Four-predictor phenotyping training pipeline — local version.

Trains four independent GMMs, each on its own clinically-relevant feature
subspace, with k determined by BIC/Silhouette analysis on this dataset:

    1. Overall Phenotype  k=3  5 independent features   3 clinical patterns
    2. Fluid Severity     k=4  SRF + IRF                Anti-VEGF guidance
    3. Prognosis          k=3  ELM + EZ                 Visual recovery outlook
    4. Inflammation       k=2  HF density               Corticosteroid flag

Usage (from project root):
    python backend/phenotyping/train_phenotyping.py

Outputs to backend/weights/:
    scaler_overall.pkl     gmm_overall.pkl
    scaler_fluid.pkl       gmm_fluid.pkl
    scaler_prognosis.pkl   gmm_prognosis.pkl
    scaler_inflammation.pkl  gmm_inflammation.pkl
    cluster_maps.json
"""

from __future__ import annotations
import json, os, pickle, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_PROJECT    = _HERE.parent.parent
DATASET_DIR = _PROJECT / "dataset" / "RVO-Lesion"
MASKS_DIR   = DATASET_DIR / "Image_Seg" / "masks"
TRAIN_TXT   = DATASET_DIR / "Image_Seg" / "train.txt"
TEST_TXT    = DATASET_DIR / "Image_Seg" / "test.txt"
LABELME_DIR = DATASET_DIR / "RVO_Lesion_Labelme"
WEIGHTS_DIR = _HERE.parent / "weights"

# ── Model configuration (k values derived from BIC + Silhouette analysis) ────
LABEL_SRF, LABEL_IRF, LABEL_ELM, LABEL_EZ = 1, 2, 3, 4
HF_MAX = 50.0
RANDOM_SEED = 42

ALL_FEATURES           = ["irf_area","srf_area","elm_integrity","ez_integrity",
                           "hf_density","fluid_total","structure_score"]
# Independent features only — fluid_total and structure_score are linear combinations of
# the above, so including them in the GMM feature matrix creates near-singular covariance.
OVERALL_FEATURES       = ["irf_area","srf_area","elm_integrity","ez_integrity","hf_density"]
FLUID_FEATURES         = ["srf_area","irf_area"]
PROGNOSIS_FEATURES     = ["elm_integrity","ez_integrity"]
INFLAMMATION_FEATURES  = ["hf_density"]

MODELS = {
    # name → (features, k, rank_feature, ascending, cov_type)
    # ascending=True  → cluster 0 = lowest value of rank_feature
    # ascending=False → cluster 0 = highest value of rank_feature
    # cov_type: "tied" for small minority clusters; "full" when all clusters are well-populated
    "overall":      (OVERALL_FEATURES,      3, "fluid_total",  False, "full"),
    "fluid":        (FLUID_FEATURES,        4, "fluid_total",  True,  "full"),
    "prognosis":    (PROGNOSIS_FEATURES,    3, "ez_integrity", True,  "full"),
    "inflammation": (INFLAMMATION_FEATURES, 3, "hf_density",   True,  "full"),
}

# ── Clinical labels per cluster index ─────────────────────────────────────────
LABELS = {
    "overall":      {0: "Exudative with Structural Compromise",
                     1: "Exudative with Preserved Photoreceptors",
                     2: "Dry Macula, Intact Retinal Layers"},
    "fluid":        {0: "IRF Only",              1: "IRF + Trace SRF",
                     2: "Mixed IRF + SRF",       3: "SRF Dominant"},
    "prognosis":    {0: "Severe Disruption",     1: "Partial Disruption",
                     2: "Intact Photoreceptors"},
    "inflammation": {0: "Low Inflammation", 1: "Moderate Inflammation", 2: "High Inflammation"},
}


# ── Biomarker extraction ──────────────────────────────────────────────────────

def load_stems(txt_path: Path) -> list[str]:
    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [Path(ln.replace("\\", "/")).stem for ln in lines]


def load_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return (mask[:, :, 0] if mask.ndim == 3 else mask).astype(np.uint8)


def count_hf(stem: str) -> int:
    pid  = stem.rsplit("_", 1)[0]
    jp   = LABELME_DIR / pid / f"{stem}.json"
    if not jp.exists():
        return 0
    with open(jp, encoding="utf-8") as f:
        data = json.load(f)
    return sum(1 for s in data.get("shapes", [])
               if s.get("label", "").upper() == "HF" and s.get("points"))


def extract_biomarkers(mask: np.ndarray, hf_count: int) -> dict:
    H, W  = mask.shape
    total = float(H * W)
    irf   = np.sum(mask == LABEL_IRF) / total
    srf   = np.sum(mask == LABEL_SRF) / total
    elm   = float(np.sum(np.any(mask == LABEL_ELM, axis=0))) / W
    ez    = float(np.sum(np.any(mask == LABEL_EZ,  axis=0))) / W
    hfd   = min(hf_count / HF_MAX, 1.0)
    return {
        "irf_area":        float(irf),
        "srf_area":        float(srf),
        "elm_integrity":   float(elm),
        "ez_integrity":    float(ez),
        "hf_density":      float(hfd),
        "fluid_total":     float(irf + srf),
        "structure_score": float((elm + ez) / 2),
    }


def process_split(stems: list[str], split_name: str) -> pd.DataFrame:
    rows, skipped = [], 0
    for i, stem in enumerate(stems):
        if i % 300 == 0:
            print(f"  [{split_name}] {i}/{len(stems)}...", end="\r")
        mp = MASKS_DIR / f"{stem}.png"
        if not mp.exists():
            skipped += 1
            continue
        mask = load_mask(mp)
        if mask is None:
            skipped += 1
            continue
        rows.append({"patient_id": stem.rsplit("_", 1)[0], "split": split_name,
                     **extract_biomarkers(mask, count_hf(stem))})
    print(f"  [{split_name:5s}] processed {len(rows)}, skipped {skipped}          ")
    return pd.DataFrame(rows)


# ── Cluster remapping ─────────────────────────────────────────────────────────

def make_remap(raw_labels: np.ndarray, df: pd.DataFrame,
               rank_feature: str, ascending: bool) -> dict[int, int]:
    """
    Map raw GMM cluster IDs to clinical IDs ordered by mean of rank_feature.
    ascending=True  → clinical 0 = lowest mean (e.g. least fluid)
    ascending=False → clinical 0 = highest mean (e.g. most fluid / Active Disease)
    """
    n = int(raw_labels.max()) + 1
    means = [df.loc[raw_labels == c, rank_feature].mean() for c in range(n)]
    rank  = np.argsort(means)          # ascending order of means
    if not ascending:
        rank = rank[::-1]
    return {int(rank[i]): i for i in range(n)}


# ── Train one GMM ─────────────────────────────────────────────────────────────

def train_gmm(name: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
              features: list[str], k: int,
              rank_feature: str, ascending: bool,
              cov_type: str = "full") -> dict:
    """
    Fit StandardScaler + GMM on train patients, remap clusters, return artefacts.
    """
    print(f"\n  [{name.upper()}]  features={features}  k={k}")

    X_train = df_train[features].values
    X_test  = df_test[features].values

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                          n_init=20, random_state=RANDOM_SEED, max_iter=500)
    gmm.fit(X_train_sc)
    print(f"     BIC={gmm.bic(X_train_sc):.1f}  AIC={gmm.aic(X_train_sc):.1f}"
          f"  converged={gmm.converged_}")

    raw_train = gmm.predict(X_train_sc)
    raw_test  = gmm.predict(X_test_sc)

    remap = make_remap(raw_train, df_train, rank_feature, ascending)

    train_clusters = np.array([remap[c] for c in raw_train])
    test_clusters  = np.array([remap[c] for c in raw_test])

    # Centroids in original (unscaled) feature space
    centroids = {}
    for cid in range(k):
        m = train_clusters == cid
        centroids[str(cid)] = {f: float(df_train.loc[m, f].mean()) for f in features}

    # Print per-cluster stats
    label_map = LABELS[name]
    print(f"     {'Cluster':<5}  {'Label':<25}  {'N_train':>7}  {rank_feature:>16}")
    print(f"     {'-'*5}  {'-'*25}  {'-'*7}  {'-'*16}")
    for cid in range(k):
        m   = train_clusters == cid
        n   = int(m.sum())
        val = float(df_train.loc[m, rank_feature].mean())
        print(f"     {cid:<5}  {label_map[cid]:<25}  {n:>7}  {val:>16.5f}")

    return {
        "scaler":         scaler,
        "gmm":            gmm,
        "remap":          {str(k_): v for k_, v in remap.items()},
        "centroids":      centroids,
        "train_clusters": train_clusters,
        "test_clusters":  test_clusters,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  RVO-ME Four-Predictor Phenotyping — Local Training")
    print("=" * 65)

    print("\nLoading splits...")
    train_stems = load_stems(TRAIN_TXT)
    test_stems  = load_stems(TEST_TXT)
    print(f"  train: {len(train_stems)} images | test: {len(test_stems)} images")

    print("\nExtracting biomarkers from ground-truth masks...")
    df_train_imgs = process_split(train_stems, "train")
    df_test_imgs  = process_split(test_stems,  "test")

    if df_train_imgs.empty:
        print("ERROR: no images processed — check MASKS_DIR path.")
        sys.exit(1)

    print("\nAggregating to patient level (mean across all B-scans per patient)...")
    def agg(df):
        return df.groupby("patient_id", as_index=False).agg(
            {**{f: "mean" for f in ALL_FEATURES}, "split": "first"})

    df_train = agg(df_train_imgs)
    df_test  = agg(df_test_imgs)
    print(f"  train: {len(df_train)} patients | test: {len(df_test)} patients")

    print("\nTraining GMMs...")
    results     = {}
    cluster_map = {}

    for name, (features, k, rank_feature, ascending, cov_type) in MODELS.items():
        res = train_gmm(name, df_train, df_test, features, k, rank_feature, ascending, cov_type)
        results[name] = res
        cluster_map[name] = {
            "k":        k,
            "features": features,
            "remap":    res["remap"],
            "centroids": res["centroids"],
        }

    print("\nSaving artefacts...")
    for name, res in results.items():
        sp = WEIGHTS_DIR / f"scaler_{name}.pkl"
        gp = WEIGHTS_DIR / f"gmm_{name}.pkl"
        with open(sp, "wb") as f:
            pickle.dump(res["scaler"], f)
        with open(gp, "wb") as f:
            pickle.dump(res["gmm"], f)
        print(f"  {sp.name} ({sp.stat().st_size/1024:.1f} KB)"
              f"  +  {gp.name} ({gp.stat().st_size/1024:.1f} KB)")

    mp = WEIGHTS_DIR / "cluster_maps.json"
    with open(mp, "w") as f:
        json.dump(cluster_map, f, indent=2)
    print(f"  {mp.name}")

    print(f"\nDone. Restart the backend to load new models.")
    print("=" * 65)


if __name__ == "__main__":
    main()
