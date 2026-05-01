"""
evaluation.py - Inference Metrics & Uncertainty Quality Analysis

Implements standard reconstruction metrics (MAE, RMSE, PSNR, SSIM) alongside
rigorous epistemic uncertainty scoring metrics (Spearman, Risk-Coverage, Calibration).

Medical ML Rationale:
Standard reconstruction metrics treat all errors equally. In radiotherapy dose
prediction, recognizing *where* the model is uncertain is sometimes more 
important than the absolute error. Spearman rank evaluates whether high model 
uncertainty reliably maps to physical reconstruction errors.
"""

import logging
import math
from typing import Dict, List, Any

import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

# =============================================================================
# Standard Reconstruction Metrics
# =============================================================================

def compute_reconstruction_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Computes standard physical reconstruction numbers.
    CT normalisation spans [-1, 1], thus data_range is 2.0.
    """
    # Flat vectors computationally faster for MAE/RMSE
    err = pred.ravel() - gt.ravel()
    
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    
    # Avoid div/0 if perfect prediction
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * math.log10((2.0 ** 2) / mse)
        
    s_val = float(ssim(
        gt.squeeze(), 
        pred.squeeze(), 
        data_range=2.0,
        channel_axis=None if gt.squeeze().ndim == 2 else 0
    ))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": s_val
    }

# =============================================================================
# Uncertainty Quality Metrics
# =============================================================================

def compute_spearman(err: np.ndarray, std: np.ndarray) -> float:
    """
    1. Spearman Correlation: 
    Does the model output higher standard deviations where errors are larger?
    Uses random subsampling (10,000 pixels) to avoid extreme O(NlogN) slow downs.
    """
    e_flat = np.abs(err.ravel())
    s_flat = std.ravel()
    
    # Subsample for speed without losing statistical significance
    if len(e_flat) > 10000:
        idx = np.random.choice(len(e_flat), 10000, replace=False)
        e_flat = e_flat[idx]
        s_flat = s_flat[idx]
        
    corr, _ = stats.spearmanr(s_flat, e_flat)
    # Spearmanr can return NaN if std map is perfectly flat (e.g. deterministic failure)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_calibration(err: np.ndarray, std: np.ndarray, bins: int = 10) -> float:
    """
    2. Calibration Slope:
    Segments the uncertainty into 10 bins. Computes the slope of the best-fit line
    between predicted standard deviation and actual empirical error. Perfectly calibrated
    distributions yield a slope of 1.0.
    """
    e_flat = np.abs(err.ravel())
    s_flat = std.ravel()
    
    if s_flat.max() == s_flat.min():
        return 0.0 # flat
        
    # Digitize into bins
    bin_edges = np.linspace(s_flat.min(), s_flat.max(), bins + 1)
    bin_items = np.digitize(s_flat, bin_edges) - 1
    
    err_means, std_means = [], []
    for b in range(bins):
        mask = (bin_items == b)
        if mask.sum() > 0:
            err_means.append(np.mean(e_flat[mask]))
            std_means.append(np.mean(s_flat[mask]))
            
    if len(err_means) > 1:
        slope, _, _, _, _ = stats.linregress(std_means, err_means)
        return float(slope)
    return 0.0


def compute_risk_coverage(err: np.ndarray, std: np.ndarray) -> float:
    """
    3. Risk-Coverage Area Under Curve:
    Sort pixels by uncertainty. Greedily remove the most uncertain pixels.
    If uncertainty is a good proxy for error, the MAE of the remaining pixels
    should drop rapidly. Returns the normalized Area Under this risk curve.
    """
    e_flat = np.abs(err.ravel())
    s_flat = std.ravel()
    
    # Sort by standard deviation descending
    sort_idx = np.argsort(-s_flat)
    sorted_err = e_flat[sort_idx]
    
    # Compute cumulative sums to dynamically calculate means
    cum_sum = np.cumsum(sorted_err[::-1])[::-1] # sum from i to end
    counts = np.arange(len(sorted_err), 0, -1)
    
    mae_curve = cum_sum / counts
    
    # Normalize AUC to [0, 1] relative to the baseline start MAE
    baseline_mae = mae_curve[0]
    if baseline_mae == 0:
        return 1.0
        
    auc = np.trapz(mae_curve, dx=1.0/len(mae_curve)) / baseline_mae
    return float(auc)


def compute_uncertainty_quality(pred: np.ndarray, gt: np.ndarray, std: np.ndarray) -> Dict[str, float]:
    """Evaluates all required epistemic validity measures."""
    err = pred - gt
    return {
        "Spearman-rho": compute_spearman(err, std),
        "Calibration-Slope": compute_calibration(err, std),
        "Risk-Coverage-AUC": compute_risk_coverage(err, std)
    }

# =============================================================================
# Aggregation across K-Folds
# =============================================================================

def evaluate_kfold(fold_results: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Translates individual fold lists into the rigidly requested:
      {metric: {"mean": float, "std": float, "per_fold": list}}
    Format string example natively provided.
    """
    if not fold_results:
        return {}
        
    keys = fold_results[0].keys()
    agg_report = {}
    
    for k in keys:
        vals = [f[k] for f in fold_results if k in f]
        mean_v = np.mean(vals)
        std_v  = np.std(vals)
        
        agg_report[k] = {
            "mean": float(mean_v),
            "std": float(std_v),
            "per_fold": vals,
            "_formatted": f"{mean_v:.3f} ± {std_v:.3f}"
        }
        
    return agg_report
