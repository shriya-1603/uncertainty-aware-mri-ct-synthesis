"""
dose_proxy.py - Clinical Dosimetry Proxies for Radiotherapy

Translates normalized algorithmic CT outputs back into physical Hounsfield Units,
subsequently mapping them to relative electron Stopping Power Ratios (SPR) and
estimating rudimentary proton radiotherapy attenuation via Pencil Beam Water
Equivalent Thickness (WET) equations.

Medical ML Rationale:
A 50 HU error in soft tissue (-120 to 200 HU) heavily skews the SPR mapping by 
almost 1.0 -> 1.10. However, a 50 HU error in air (-1000 HU) means absolutely 
nothing for dose attenuation. Using these clinical physical proxies proves whether 
the generative improvements translate into actual proton therapy margins.
"""

import numpy as np
from typing import Dict

# =============================================================================
# Physical Mappings
# =============================================================================

# Bounds rigidly replicated from dataset limits
CT_HU_MIN = -1000.0
CT_HU_MAX = 3000.0

def norm_to_hu(volume: np.ndarray) -> np.ndarray:
    """Inverts the dataset CT normalisation: [-1, 1] -> [-1000, 3000]."""
    span = CT_HU_MAX - CT_HU_MIN
    hu = ((volume + 1.0) / 2.0) * span + CT_HU_MIN
    return np.clip(hu, CT_HU_MIN, CT_HU_MAX)

def hu_to_spr(hu: np.ndarray) -> np.ndarray:
    """
    Applies the piecewise linear HU -> SPR mapping (Schneider 1996 approx).
    Explicitly follows the analytical slopes outlined in the design spec.
    """
    spr = np.zeros_like(hu, dtype=np.float32)
    
    # 1. Air Mask
    m_air = hu < -950
    spr[m_air] = 0.001
    
    # 2. Lung Interface
    m_lung = (hu >= -950) & (hu < -120)
    spr[m_lung] = 0.001 + (hu[m_lung] - -950) * ((0.29 - 0.001) / (-120.0 - -950.0))
    
    # 3. Soft Tissue 
    m_soft = (hu >= -120) & (hu <= 200)
    spr[m_soft] = 0.92 + (hu[m_soft] - -120) * ((1.10 - 0.92) / (200.0 - -120.0))
    
    # 4. Dense Bone
    m_bone = hu > 200
    hu_bone_clamped = np.clip(hu[m_bone], 200, 3000)
    spr[m_bone] = 1.10 + (hu_bone_clamped - 200) * ((2.50 - 1.10) / (3000.0 - 200.0))
    
    return spr

def simulate_pencil_beam(spr: np.ndarray, voxel_size_mm: float = 1.0) -> np.ndarray:
    """
    Calculates primary attenuation dose using the integral Water Equivalent Path.
    Assumes straight-line beam delivery along the y-axis (axis=0) of the slice.
    """
    # WET = Integral of SPR along the beam path
    wet = np.cumsum(spr, axis=0) * voxel_size_mm
    
    # Simple exponential attenuation: Base Dose (1.0) * e^(-mu * WET)
    mu_water = 0.04 # approx attenuation coeff per mm
    dose = 1.0 * np.exp(-mu_water * wet)
    return dose

# =============================================================================
# Safety Coverage Validations
# =============================================================================

def aggregate_coverage_error(err_map: np.ndarray, std_map: np.ndarray, coverage: float) -> float:
    """
    Simulates clinical rejection: masks out regions where generative uncertainty
    is mathematically too high, calculating average absolute errors strictly on
    the accepted safe geometries.
    """
    err_flat = np.abs(err_map.ravel())
    std_flat = std_map.ravel()
    
    if coverage >= 1.0:
        return float(np.mean(err_flat))
        
    threshold = np.percentile(std_flat, coverage * 100.0)
    safe_mask = std_flat <= threshold
    
    if safe_mask.sum() == 0:
        return 0.0
    return float(np.mean(err_flat[safe_mask]))

def evaluate_clinical_proxies(pred_norm: np.ndarray, gt_norm: np.ndarray, std_map: np.ndarray) -> Dict[str, float]:
    """
    Runs the complete proxy loop for an inferred patient slice.
    Yields 6 values: SPR & Dose error at 100%, 75%, and 50% coverages respectively.
    """
    pred_hu = norm_to_hu(pred_norm)
    gt_hu = norm_to_hu(gt_norm)
    
    pred_spr = hu_to_spr(pred_hu)
    gt_spr = hu_to_spr(gt_hu)
    
    spr_err = np.abs(pred_spr - gt_spr)
    
    pred_dose = simulate_pencil_beam(pred_spr)
    gt_dose = simulate_pencil_beam(gt_spr)
    
    dose_err = np.abs(pred_dose - gt_dose)
    
    return {
        "SPR_Error_100%": aggregate_coverage_error(spr_err, std_map, 1.0),
        "SPR_Error_75%":  aggregate_coverage_error(spr_err, std_map, 0.75),
        "SPR_Error_50%":  aggregate_coverage_error(spr_err, std_map, 0.50),
        
        "Dose_Error_100%": aggregate_coverage_error(dose_err, std_map, 1.0),
        "Dose_Error_75%":  aggregate_coverage_error(dose_err, std_map, 0.75),
        "Dose_Error_50%":  aggregate_coverage_error(dose_err, std_map, 0.50)
    }
