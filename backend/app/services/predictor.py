def predict_treatment(metrics: dict):
    """
    Rule-based treatment severity prediction based on biomarkers.
    """
    srf_perc = metrics.get('srf_area_percent', 0)
    irf_perc = metrics.get('irf_area_percent', 0)
    hf_count = metrics.get('hf_count', 0)
    elm_int = metrics.get('elm_integrity', 1.0)
    ez_int = metrics.get('ez_integrity', 1.0)
    
    # In an OCT image, the retina itself only represents a small portion of the total pixels.
    # Therefore, 1-2% total image area as fluid is actually very huge clinically!
    
    # 1. Severe Conditions
    if srf_perc > 0.5 or irf_perc > 2.0 or hf_count > 6 or ez_int < 0.2:
        return {
            "severity": "Severe",
            "recommendation": "Urgent treatment needed",
            "reasoning": f"Critical indicators met: High fluid volume or severe structural disruption (EZ: {ez_int:.2f})."
        }
        
    # 2. Moderate Conditions
    elif srf_perc > 0.1 or irf_perc > 0.5 or hf_count >= 3 or elm_int < 0.4:
        return {
            "severity": "Moderate",
            "recommendation": "Treatment recommended",
            "reasoning": f"Moderate fluid accumulation (IRF: {irf_perc}%) and/or ELM layer disruption detected."
        }
        
    # 3. Mild Conditions
    elif irf_perc > 0 or srf_perc > 0 or hf_count > 0:
        return {
            "severity": "Mild",
            "recommendation": "Monitor closely",
            "reasoning": "Trace amounts of fluid or isolated hyperreflective foci present. Continue observation."
        }
        
    # 4. Normal / Expected
    else:
        # If no fluid, no HF, and layers are relatively intact
        return {
            "severity": "Normal",
            "recommendation": "No active treatment needed",
            "reasoning": "No pathological fluid or spots detected. Retinal lines appear sufficiently stable."
        }
