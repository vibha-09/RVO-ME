import numpy as np

def quantify_biomarkers(segmentation_mask: np.ndarray, detection_results: dict):
    """
    Computes numerical metrics from the ML models.
    """
    total_pixels = segmentation_mask.size
    
    # 0: Background, 1: SRF, 2: IRF, 3: ELM, 4: EZ
    srf_pixels = np.sum(segmentation_mask == 1)
    irf_pixels = np.sum(segmentation_mask == 2)
    srf_percent = (srf_pixels / total_pixels) * 100
    irf_percent = (irf_pixels / total_pixels) * 100
    total_fluid_percent = srf_percent + irf_percent
    
    # HF Count
    hf_count = len(detection_results['boxes'])
    
    # ELM and EZ Integrity
    elm_pixels = np.sum(segmentation_mask == 3)
    ez_pixels = np.sum(segmentation_mask == 4)
    
    # Simplified integrity calculation (assuming full width should ideally be non-zero)
    # This is a naive implementation: ratio of columns with at least one pixel belonging to the layer.
    _, width = segmentation_mask.shape
    
    elm_cols = np.any(segmentation_mask == 3, axis=0)
    ez_cols = np.any(segmentation_mask == 4, axis=0)
    
    elm_integrity = np.sum(elm_cols) / width if width > 0 else 0
    ez_integrity = np.sum(ez_cols) / width if width > 0 else 0
    
    return {
        "srf_area_pixels": float(srf_pixels),
        "srf_area_percent": round(float(srf_percent), 2),
        "irf_area_pixels": float(irf_pixels),
        "irf_area_percent": round(float(irf_percent), 2),
        "total_fluid_area_percent": round(float(total_fluid_percent), 2),
        "hf_count": int(hf_count),
        "elm_integrity": round(float(elm_integrity), 2),
        "ez_integrity": round(float(ez_integrity), 2)
    }
