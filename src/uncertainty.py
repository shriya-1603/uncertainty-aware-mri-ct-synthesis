"""
uncertainty.py - Epistemic and Aleatoric Uncertainty Quantifiers

Implements:
1. Monte Carlo Dropout (MCD)
2. Deep Ensembles (DE)
3. Test-Time Augmentation (TTA)

Medical ML Rationale:
CT synthesis purely interpolates structural approximations from MR boundaries.
Highlighting regions with high generative uncertainty provides clinicians with
a direct confidence map, guarding against synthetic hallucinations during
radiotherapy dose planning.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

# =============================================================================
# U1: Monte Carlo Dropout
# =============================================================================

def get_mc_dropout_uncertainty(
    model: nn.Module,
    mr_tensor: torch.Tensor,
    ct_tensor: torch.Tensor,
    t_passes: int = 20
) -> Dict[str, np.ndarray]:
    """
    U1: Monte Carlo Dropout.
    Requires model to have an `enable_dropout()` method to unlock layers manually 
    masked by `.eval()` ensuring BN stats are frozen but dropouts randomize.
    """
    model.eval()
    if hasattr(model, 'enable_dropout'):
        model.enable_dropout()
    else:
        raise NotImplementedError("Model lacks 'enable_dropout()' safety method.")
    
    samples = []
    with torch.no_grad():
        for _ in range(t_passes):
            pred = model(mr_tensor)
            samples.append(pred.cpu().numpy())
            
    samples = np.concatenate(samples, axis=0) # [T, 1, H, W]
    mean = np.mean(samples, axis=0, keepdims=True)
    std = np.std(samples, axis=0, keepdims=True)
    
    return {
        "method": "mc_dropout",
        "mri": mr_tensor.cpu().numpy(),
        "ct_gt": ct_tensor.cpu().numpy(),
        "mean": mean,
        "std": std,
        "samples": samples
    }


# =============================================================================
# U2: Deep Ensembles
# =============================================================================

def get_deep_ensemble_uncertainty(
    models: List[nn.Module],
    mr_tensor: torch.Tensor,
    ct_tensor: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    U2: Deep Ensembles.
    Averaged deterministic pass over N totally independent model weights.
    """
    samples = []
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(mr_tensor)
            samples.append(pred.cpu().numpy())
            
    samples = np.concatenate(samples, axis=0)
    mean = np.mean(samples, axis=0, keepdims=True)
    std = np.std(samples, axis=0, keepdims=True)
    
    return {
        "method": "deep_ensemble",
        "mri": mr_tensor.cpu().numpy(),
        "ct_gt": ct_tensor.cpu().numpy(),
        "mean": mean,
        "std": std,
        "samples": samples
    }


# =============================================================================
# U3: Test-Time Augmentation
# =============================================================================

def get_tta_uncertainty(
    model: nn.Module,
    mr_tensor: torch.Tensor,
    ct_tensor: torch.Tensor,
    t_passes: int = 10
) -> Dict[str, np.ndarray]:
    """
    U3: Test-Time Augmentation (TTA).
    Evaluated with strict model.eval() standard configuration. 
    Stochastic physical variations (flips, 90-rotations) + mild 5% intensity noise.
    Outputs are rigidly reversed onto the original coordinate map to compute stable std.
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(t_passes):
            aug_mr = mr_tensor.clone()
            
            # Action flags for inverse transform
            h_flip = torch.rand(1).item() > 0.5
            v_flip = torch.rand(1).item() > 0.5
            k_rot = torch.randint(0, 4, (1,)).item()
            
            if h_flip:
                aug_mr = torch.flip(aug_mr, dims=[-1])
            if v_flip:
                aug_mr = torch.flip(aug_mr, dims=[-2])
            if k_rot > 0:
                aug_mr = torch.rot90(aug_mr, k=k_rot, dims=[-2, -1])
            
            # +/- 5% Intensity Noise
            noise = torch.randn_like(aug_mr) * 0.05
            aug_mr = torch.clamp(aug_mr + noise, 0.0, 1.0)
            
            # Predict
            pred = model(aug_mr)
            
            # Inverse transforms to align anatomically
            if k_rot > 0:
                pred = torch.rot90(pred, k=-k_rot, dims=[-2, -1])
            if v_flip:
                pred = torch.flip(pred, dims=[-2])
            if h_flip:
                pred = torch.flip(pred, dims=[-1])
                
            samples.append(pred.cpu().numpy())
            
    samples = np.concatenate(samples, axis=0)
    mean = np.mean(samples, axis=0, keepdims=True)
    std = np.std(samples, axis=0, keepdims=True)
    
    return {
        "method": "tta",
        "mri": mr_tensor.cpu().numpy(),
        "ct_gt": ct_tensor.cpu().numpy(),
        "mean": mean,
        "std": std,
        "samples": samples
    }
