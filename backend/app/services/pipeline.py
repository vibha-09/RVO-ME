"""
Four-predictor biomarker phenotyping pipeline — inference module.

Each predictor is an independent GMM trained on its own feature subspace:

    overall      k=3   5 features        → Exudative+Compromised (fluid~7.9%) / Exudative+Preserved (fluid~4.6%) / Dry Macula (fluid~3.0%)
    fluid        k=4   SRF + IRF        → Anti-VEGF guidance (4 severity tiers)
    prognosis    k=3   ELM + EZ         → Visual recovery outlook
    inflammation k=2   HF density       → Corticosteroid flag

Required artefacts in backend/weights/:
    scaler_overall.pkl     gmm_overall.pkl
    scaler_fluid.pkl       gmm_fluid.pkl
    scaler_prognosis.pkl   gmm_prognosis.pkl
    scaler_inflammation.pkl  gmm_inflammation.pkl
    cluster_maps.json

Generate with: python backend/phenotyping/train_phenotyping.py
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from typing import Optional

import numpy as np

# ── Feature subspaces (must match training script exactly) ────────────────────
ALL_FEATURES          = ["irf_area","srf_area","elm_integrity","ez_integrity",
                          "hf_density","fluid_total","structure_score"]
# Overall GMM uses only the 5 independent features — fluid_total and structure_score
# are linear combinations of the others and cause near-singular covariance.
OVERALL_GMM_FEATURES  = ["irf_area","srf_area","elm_integrity","ez_integrity","hf_density"]
FLUID_FEATURES        = ["srf_area","irf_area"]
PROGNOSIS_FEATURES    = ["elm_integrity","ez_integrity"]
INFLAMMATION_FEATURES = ["hf_density"]

HF_MAX = 50.0  # must match train_phenotyping.py

# ── Clinical labels ───────────────────────────────────────────────────────────
OVERALL_LABELS = {
    0: "Exudative with Structural Compromise",
    1: "Exudative with Preserved Photoreceptors",
    2: "Dry Macula, Intact Retinal Layers",
}

FLUID_LABELS = {
    0: "IRF Only",
    1: "IRF + Trace SRF",
    2: "Mixed IRF + SRF",
    3: "SRF Dominant",
}
FLUID_RECOMMENDATIONS = {
    0: ("Pure intraretinal fluid — no subretinal component detected. "
        "Standard Anti-VEGF protocol. IRF-only pattern typically responds well "
        "to Anti-VEGF; monitor central retinal thickness at each visit."),
    1: ("Predominantly intraretinal fluid with trace subretinal component. "
        "Anti-VEGF loading (3× monthly) recommended. "
        "Re-evaluate SRF resolution at 3 months as a secondary treatment response marker."),
    2: ("Mixed intraretinal and subretinal fluid in roughly equal proportions. "
        "Anti-VEGF loading phase required. SRF typically resolves slower than IRF — "
        "use SRF clearance as the primary treatment endpoint at 3-month review."),
    3: ("SRF-dominant fluid pattern indicating active subretinal exudation. "
        "Aggressive Anti-VEGF loading required. High SRF burden is associated with "
        "longer treatment duration. Consider switch or extended protocol "
        "if SRF persists beyond 3 injections."),
}

PROGNOSIS_LABELS = {
    0: "Severe Disruption",
    1: "Partial Disruption",
    2: "Intact Photoreceptors",
}
PROGNOSIS_OUTLOOKS = {
    0: ("Guarded prognosis. Significant ELM/EZ disruption detected. "
        "Risk of permanent visual impairment. Prompt and aggressive "
        "treatment is critical to limit further photoreceptor loss."),
    1: ("Moderate prognosis. Partial EZ layer disruption present. "
        "Visual recovery is possible with timely and consistent "
        "Anti-VEGF treatment. Monitor EZ recovery at each visit."),
    2: ("Favorable prognosis. Intact ELM and EZ layers indicate preserved "
        "photoreceptor function. Good potential for visual acuity recovery "
        "with appropriate treatment."),
}

INFLAMMATION_LABELS = {
    0: "Low Inflammation",
    1: "Moderate Inflammation",
    2: "High Inflammation",
}
INFLAMMATION_RECOMMENDATIONS = {
    0: ("Low hyperreflective foci burden. Standard Anti-VEGF protocol "
        "is appropriate. No inflammatory escalation required."),
    1: ("Moderate hyperreflective foci burden — active inflammatory activity detected. "
        "Proceed with Anti-VEGF loading and monitor HF count at each visit. "
        "Consider corticosteroid adjunct if HF does not reduce after 3 injections."),
    2: ("High hyperreflective foci count indicates severe active inflammation. "
        "High HF burden is strongly associated with Anti-VEGF resistance. "
        "Corticosteroid adjunct (e.g., dexamethasone implant) is recommended "
        "alongside Anti-VEGF therapy."),
}

# ── Weight paths ──────────────────────────────────────────────────────────────
_BASE        = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.normpath(os.path.join(_BASE, "..", "..", "weights"))

_ARTEFACT_FILES = [
    "scaler_overall.pkl",    "gmm_overall.pkl",
    "scaler_fluid.pkl",      "gmm_fluid.pkl",
    "scaler_prognosis.pkl",  "gmm_prognosis.pkl",
    "scaler_inflammation.pkl", "gmm_inflammation.pkl",
    "cluster_maps.json",
]

# Lazy-loaded singletons
_scalers    = {}   # name → StandardScaler
_gmms       = {}   # name → GaussianMixture
_maps       = {}   # name → {"remap": {str→int}, "centroids": {str→{feat→float}}}
_models_ok  = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    segmentation_mask: np.ndarray,
    detection_results: dict,
) -> dict:
    """
    Compute all 7 biomarker features from model outputs.

    Args:
        segmentation_mask: (H,W) uint8  0=BG 1=SRF 2=IRF 3=ELM 4=EZ
        detection_results: dict with 'boxes' key → list/array of HF boxes

    Returns:
        dict with keys == ALL_FEATURES (raw fractions, not percentages)
    """
    H, W  = segmentation_mask.shape
    total = float(H * W)

    irf = np.sum(segmentation_mask == 2) / total
    srf = np.sum(segmentation_mask == 1) / total
    elm = float(np.sum(np.any(segmentation_mask == 3, axis=0))) / W
    ez  = float(np.sum(np.any(segmentation_mask == 4, axis=0))) / W
    hfd = min(len(detection_results.get("boxes", [])) / HF_MAX, 1.0)

    return {
        "irf_area":        float(irf),
        "srf_area":        float(srf),
        "elm_integrity":   float(elm),
        "ez_integrity":    float(ez),
        "hf_density":      float(hfd),
        "fluid_total":     float(irf + srf),
        "structure_score": float((elm + ez) / 2),
    }


def predict_pipeline(
    features: dict,
    patient_id: Optional[str] = None,
) -> dict:
    """
    Run all four independent predictors on one image's biomarkers.

    Returns:
        {
          "overall":      {cluster, label, description, severity_score, probability},
          "fluid":        {cluster, severity, irf_percent, srf_percent,
                           recommendation, probability},
          "prognosis":    {cluster, tier, elm_integrity, ez_integrity,
                           outlook, probability},
          "inflammation": {cluster, level, hf_count, hf_density,
                           recommendation, probability},
          "explanation":  str,
          "ml_mode":      bool
        }
    """
    ml_ok = _load_models()

    if ml_ok:
        overall      = _predict_overall(features)
        fluid        = _predict_fluid(features)
        prognosis    = _predict_prognosis(features)
        inflammation = _predict_inflammation(features)
        explanation  = _build_explanation(features, overall, fluid, prognosis, inflammation)
        ml_mode      = True
    else:
        overall      = _fallback_overall(features)
        fluid        = _fallback_fluid(features)
        prognosis    = _fallback_prognosis(features)
        inflammation = _fallback_inflammation(features)
        explanation  = _fallback_explanation(features)
        ml_mode      = False

    return {
        "overall":      overall,
        "fluid":        fluid,
        "prognosis":    prognosis,
        "inflammation": inflammation,
        "explanation":  explanation,
        "ml_mode":      ml_mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_models() -> bool:
    global _scalers, _gmms, _maps, _models_ok

    if _models_ok is not None:
        return _models_ok

    missing = [f for f in _ARTEFACT_FILES
               if not os.path.exists(os.path.join(_WEIGHTS_DIR, f))]
    if missing:
        warnings.warn(
            f"[pipeline] Artefacts not found: {missing}. Using rule-based fallback. "
            "Run: python backend/phenotyping/train_phenotyping.py"
        )
        _models_ok = False
        return False

    try:
        for name in ["overall", "fluid", "prognosis", "inflammation"]:
            sp = os.path.join(_WEIGHTS_DIR, f"scaler_{name}.pkl")
            gp = os.path.join(_WEIGHTS_DIR, f"gmm_{name}.pkl")
            with open(sp, "rb") as f:
                _scalers[name] = pickle.load(f)
            with open(gp, "rb") as f:
                _gmms[name] = pickle.load(f)

        with open(os.path.join(_WEIGHTS_DIR, "cluster_maps.json")) as f:
            raw = json.load(f)
        for name, info in raw.items():
            _maps[name] = {
                "remap":     {k: int(v) for k, v in info["remap"].items()},
                "centroids": info["centroids"],
                "features":  info["features"],
            }
    except Exception as exc:
        warnings.warn(f"[pipeline] Failed to load artefacts: {exc}. Using fallback.")
        _models_ok = False
        return False

    _models_ok = True
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-predictor inference
# ─────────────────────────────────────────────────────────────────────────────

def _gmm_predict(name: str, feat_keys: list[str], features: dict):
    """Scale features, run GMM, remap cluster, return (cluster, probability_dict)."""
    x       = np.array([[features[f] for f in feat_keys]])
    x_sc    = _scalers[name].transform(x)
    raw     = int(_gmms[name].predict(x_sc)[0])
    proba   = _gmms[name].predict_proba(x_sc)[0]
    remap   = _maps[name]["remap"]
    cluster = remap[str(raw)]
    # Build probability dict keyed by clinical cluster id
    prob_by_clinical = {
        remap[str(i)]: round(float(proba[i]), 3)
        for i in range(len(proba))
    }
    return cluster, prob_by_clinical


def _build_overall_description(cluster: int, features: dict) -> str:
    srf = features["srf_area"] * 100
    irf = features["irf_area"] * 100
    ft  = features["fluid_total"] * 100
    elm = features["elm_integrity"] * 100
    ez  = features["ez_integrity"] * 100
    hf  = int(round(features["hf_density"] * HF_MAX))

    if srf > irf * 3.0:
        fluid_phrase = f"SRF-dominant ({srf:.1f}% SRF, {irf:.1f}% IRF, {ft:.1f}% total)"
    elif irf > srf * 3.0:
        fluid_phrase = f"IRF-dominant ({irf:.1f}% IRF, {srf:.1f}% SRF, {ft:.1f}% total)"
    elif ft < 0.5:
        fluid_phrase = f"minimal fluid ({ft:.1f}% total)"
    else:
        fluid_phrase = f"mixed fluid ({srf:.1f}% SRF, {irf:.1f}% IRF, {ft:.1f}% total)"

    mean_integrity = (elm + ez) / 2.0
    if mean_integrity >= 70:
        layer_phrase = f"well-preserved photoreceptor layers (ELM {elm:.0f}%, EZ {ez:.0f}%)"
    elif mean_integrity >= 45:
        layer_phrase = f"partially disrupted photoreceptor layers (ELM {elm:.0f}%, EZ {ez:.0f}%)"
    else:
        layer_phrase = f"severely disrupted photoreceptor layers (ELM {elm:.0f}%, EZ {ez:.0f}%)"

    # Align HF wording with the binary Inflammation GMM (boundary ≈ 2 HF / hf_density=0.04)
    if hf == 0:
        hf_phrase = "no hyperreflective foci"
    elif hf <= 2:
        hf_phrase = f"{hf} hyperreflective foc{'us' if hf == 1 else 'i'} (low inflammatory activity)"
    else:
        hf_phrase = f"{hf} hyperreflective foci (elevated inflammatory burden)"

    if cluster == 0:
        return (
            f"Active exudative disease: {fluid_phrase}, {layer_phrase}, {hf_phrase}. "
            "Structural degeneration present — urgent Anti-VEGF intervention required. "
            "Prognosis is guarded; EZ/ELM recovery depends on speed of fluid resolution. "
            "Consider corticosteroid adjunct if HF burden persists after Anti-VEGF loading."
        )
    elif cluster == 1:
        return (
            f"Active exudative disease: {fluid_phrase}, {layer_phrase}, {hf_phrase}. "
            "Photoreceptor architecture shows relative preservation — good prognosis with "
            "prompt Anti-VEGF loading. Monitor EZ/ELM integrity and HF count at each visit."
        )
    else:
        # Cluster 2: Dry Macula — branch by photoreceptor status and HF
        if mean_integrity < 45:
            # Chronic atrophic RVO-ME: fluid resolved but photoreceptor damage persists
            return (
                f"Resolved exudative disease with residual structural damage: "
                f"{fluid_phrase}, {layer_phrase}, {hf_phrase}. "
                "Fluid has resolved but photoreceptor integrity is permanently compromised — "
                "visual recovery is limited. Focus on preventing disease reactivation; "
                "monitor closely for any return of fluid."
            )
        elif hf > 2:
            # Dry macula but active inflammation — risk of reactivation
            return (
                f"Dry macular state: {fluid_phrase}, {layer_phrase}, {hf_phrase}. "
                "Fluid is resolved; however, elevated HF burden indicates ongoing inflammatory "
                "activity and risk of disease reactivation. Close monitoring recommended — "
                "re-image at shorter intervals and escalate if fluid re-emerges."
            )
        else:
            return (
                f"Dry macular state: {fluid_phrase}, {layer_phrase}, {hf_phrase}. "
                "Stable or recovering retinal architecture. "
                "Continue observation; re-image if visual acuity changes."
            )


def _overall_label(cluster: int, features: dict) -> str:
    mean_integrity = (features["elm_integrity"] + features["ez_integrity"]) / 2.0
    high_hf = int(round(features["hf_density"] * HF_MAX)) > 2

    if cluster == 0:
        return ("Exudative, Structural Compromise & Active Inflammation"
                if high_hf else "Exudative with Structural Compromise")
    elif cluster == 1:
        return ("Exudative, Preserved Photoreceptors & Active Inflammation"
                if high_hf else "Exudative with Preserved Photoreceptors")
    else:  # cluster 2
        if mean_integrity < 0.45:
            return "Resolved Disease — Residual Structural Damage"
        elif high_hf:
            return "Dry Macula — Active Inflammatory Risk"
        else:
            return "Dry Macula, Intact Retinal Layers"


def _predict_overall(features: dict) -> dict:
    cluster, proba = _gmm_predict("overall", OVERALL_GMM_FEATURES, features)
    mean_integrity = (features["elm_integrity"] + features["ez_integrity"]) / 2.0
    # Correction: cluster 1 ("Preserved Photoreceptors") requires both high fluid AND
    # intact ELM/EZ. If mean integrity < 0.45, photoreceptors are actually compromised
    # → reclassify to cluster 0 ("Structural Compromise").
    if cluster == 1 and mean_integrity < 0.45:
        cluster = 0
    # Severity is always computed from raw feature values, NOT from cluster probability.
    severity = _severity_score(features)
    return {
        "cluster":        cluster,
        "label":          _overall_label(cluster, features),
        "description":    _build_overall_description(cluster, features),
        "severity_score": round(float(severity), 3),
        "probability":    proba.get(cluster, 0.0),
    }


def _predict_fluid(features: dict) -> dict:
    cluster, proba = _gmm_predict("fluid", FLUID_FEATURES, features)
    srf = features["srf_area"]
    irf = features["irf_area"]
    # Override cluster label when SRF clearly dominates (ratio ≥ 3:1) but GMM placed
    # the case in a mixed or IRF-dominant bin — cluster 3 (SRF Dominant) has very few
    # training samples so the GMM under-captures this pattern statistically.
    display_cluster = cluster
    if srf > irf * 3.0 and cluster < 3:
        display_cluster = 3
    return {
        "cluster":        display_cluster,
        "severity":       FLUID_LABELS[display_cluster],
        "irf_percent":    round(irf * 100, 3),
        "srf_percent":    round(srf * 100, 3),
        "recommendation": FLUID_RECOMMENDATIONS[display_cluster],
        "probability":    proba.get(cluster, 0.0),
    }


def _predict_prognosis(features: dict) -> dict:
    cluster, proba = _gmm_predict("prognosis", PROGNOSIS_FEATURES, features)
    return {
        "cluster":       cluster,
        "tier":          PROGNOSIS_LABELS[cluster],
        "elm_integrity": round(features["elm_integrity"], 3),
        "ez_integrity":  round(features["ez_integrity"], 3),
        "outlook":       PROGNOSIS_OUTLOOKS[cluster],
        "probability":   proba.get(cluster, 0.0),
    }


def _predict_inflammation(features: dict) -> dict:
    cluster, proba = _gmm_predict("inflammation", INFLAMMATION_FEATURES, features)
    hf_count = int(round(features["hf_density"] * HF_MAX))
    return {
        "cluster":        cluster,
        "level":          INFLAMMATION_LABELS[cluster],
        "hf_count":       hf_count,
        "hf_density":     round(features["hf_density"], 4),
        "recommendation": INFLAMMATION_RECOMMENDATIONS[cluster],
        "probability":    proba.get(cluster, 0.0),
    }


def _build_explanation(features, overall, fluid, prognosis, inflammation) -> str:
    lines = [
        "GMM Biomarker Analysis:",
        f"  Overall   : {overall['label']} (p={overall['probability']:.2f})",
        f"  Fluid     : {fluid['severity']} "
        f"(IRF={fluid['irf_percent']:.3f}%, SRF={fluid['srf_percent']:.3f}%,"
        f" p={fluid['probability']:.2f})",
        f"  Prognosis : {prognosis['tier']} "
        f"(ELM={prognosis['elm_integrity']:.3f}, EZ={prognosis['ez_integrity']:.3f},"
        f" p={prognosis['probability']:.2f})",
        f"  Inflam.   : {inflammation['level']} "
        f"(HF count~{inflammation['hf_count']}, p={inflammation['probability']:.2f})",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallbacks (when artefacts are absent)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_overall(features: dict) -> dict:
    ft = features["fluid_total"]
    ez = features["ez_integrity"]
    # Two-axis rule derived from trained cluster centroids:
    #   Cluster 0: fluid_total ~0.0788, Cluster 1: ~0.0455, Cluster 2: ~0.0298
    #   Boundary 0↔1 at (0.0788+0.0455)/2 ≈ 0.062
    #   Boundary 1↔2 at (0.0455+0.0298)/2 ≈ 0.038
    #   EZ secondary split: compromised (<0.45) vs preserved (≥0.45)
    if ft > 0.062:
        cluster = 0 if ez < 0.45 else 1   # exudative: compromised vs preserved
    elif ft > 0.038:
        cluster = 1                         # exudative with preserved structure
    else:
        cluster = 2                         # dry macula
    return {
        "cluster":        cluster,
        "label":          _overall_label(cluster, features),
        "description":    _build_overall_description(cluster, features),
        "severity_score": round(_severity_score(features), 3),
        "probability":    None,
    }


def _fallback_fluid(features: dict) -> dict:
    srf = features["srf_area"]
    irf = features["irf_area"]
    ft  = features["fluid_total"]
    if   ft < 0.001: cluster = 0
    elif ft < 0.025: cluster = 1
    elif ft < 0.055: cluster = 2
    else:            cluster = 3
    display_cluster = 3 if (srf > irf * 3.0 and cluster < 3) else cluster
    return {
        "cluster":        display_cluster,
        "severity":       FLUID_LABELS[display_cluster],
        "irf_percent":    round(irf * 100, 3),
        "srf_percent":    round(srf * 100, 3),
        "recommendation": FLUID_RECOMMENDATIONS[display_cluster],
        "probability":    None,
    }


def _fallback_prognosis(features: dict) -> dict:
    ez = features["ez_integrity"]
    if   ez < 0.30: cluster = 0
    elif ez < 0.65: cluster = 1
    else:           cluster = 2
    return {
        "cluster":       cluster,
        "tier":          PROGNOSIS_LABELS[cluster],
        "elm_integrity": round(features["elm_integrity"], 3),
        "ez_integrity":  round(features["ez_integrity"], 3),
        "outlook":       PROGNOSIS_OUTLOOKS[cluster],
        "probability":   None,
    }


def _fallback_inflammation(features: dict) -> dict:
    # Thresholds from trained k=3 centroid midpoints:
    #   Low ~0.9 HF (0.0174), Moderate ~2.9 HF (0.0581), High ~8.8 HF (0.1751)
    #   Boundary 0↔1 = (0.0174+0.0581)/2 = 0.038  →  ~2 HF
    #   Boundary 1↔2 = (0.0581+0.1751)/2 = 0.117  →  ~6 HF
    hfd = features["hf_density"]
    if   hfd < 0.038: cluster = 0
    elif hfd < 0.117: cluster = 1
    else:             cluster = 2
    hf_count = int(round(hfd * HF_MAX))
    return {
        "cluster":        cluster,
        "level":          INFLAMMATION_LABELS[cluster],
        "hf_count":       hf_count,
        "hf_density":     round(hfd, 4),
        "recommendation": INFLAMMATION_RECOMMENDATIONS[cluster],
        "probability":    None,
    }


def _fallback_explanation(features: dict) -> str:
    return (
        "Rule-based fallback (run train_phenotyping.py for GMM models).\n"
        f"  fluid_total={features['fluid_total']:.4f}  "
        f"ez_integrity={features['ez_integrity']:.3f}  "
        f"hf_density={features['hf_density']:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Severity score (used only by overall predictor)
# ─────────────────────────────────────────────────────────────────────────────

def _severity_score(features: dict) -> float:
    # Multi-component composite severity — independent of cluster label.
    #   50% fluid burden  (total fluid / 0.10 cap — above Active centroid of 0.085)
    #   30% structural loss (1 − mean ELM/EZ integrity)
    #   20% inflammation  (normalised HF density)
    # Expected ranges: truly quiescent ~0.10-0.25, moderate active ~0.45-0.65, severe ~0.75+
    fluid_norm  = min(features["fluid_total"] / 0.10, 1.0)
    struct_loss = 1.0 - features["structure_score"]
    hf_norm     = features["hf_density"]
    severity    = 0.5 * fluid_norm + 0.3 * struct_loss + 0.2 * hf_norm
    return float(np.clip(severity, 0.0, 1.0))
