"""
utils.py - Core Project Utilities & System Management

Provides generic system configurations, standardized logging structures,
graphic defaults, and synthetic dataset generation for CI smoke-tests.

Medical ML Rationale:
Consistent logging formatting enables programmatic parsing of massive
cross-validation sweeps. Synthetic data ensures researchers can validate 
pipeline modifications entirely offline without mounting heavy clinical subsets.
"""

import logging
import sys
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Logging Infrastructure
# =============================================================================

def setup_logger(name: str) -> logging.Logger:
    """Configures project-wide consistent stdout streaming."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            '%(asctime)s  %(levelname)-8s %(name)-12s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        
    return logger

# =============================================================================
# Plotting Enhancements
# =============================================================================

def init_plot_aesthetics():
    """Applies a strict, modern style-sheet native to Seaborn."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rc('font', family='sans-serif', size=11)
    plt.rc('axes', titlesize=14, labelsize=12, labelweight='bold')
    plt.rc('legend', fontsize=11)
    plt.rc('figure', autolayout=True)

# =============================================================================
# Synthetic Data Injection (Smoke Testing)
# =============================================================================

def generate_synthetic_patient(patient_dir: Path, n_slices: int = 5, size: int = 256):
    """
    Builds structurally identical mock NIfTI volumes populated with deterministically
    noisy matrices mapping to distinct intensity gradients.
    """
    patient_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    
    # Generate mock MRI: range [-300, 300] random noise, physically centered cube
    mr_grid = np.random.normal(loc=0.0, scale=100.0, size=(size, size, n_slices)).astype(np.float32)
    mr_grid[size//4:size//4*3, size//4:size//4*3, :] += 500  # distinct "body"
    mr_nifti = nib.Nifti1Image(mr_grid, affine)
    nib.save(mr_nifti, str(patient_dir / "mr.nii.gz"))
    
    # Generate mock CT: explicitly structured to test Schneider HU boundaries
    # Some pixels below -950 (Air), some soft tissue, some bone.
    ct_grid = np.random.uniform(low=-1000, high=2500, size=(size, size, n_slices)).astype(np.float32)
    ct_nifti = nib.Nifti1Image(ct_grid, affine)
    nib.save(ct_nifti, str(patient_dir / "ct.nii.gz"))
    
    # Generate Mock Mask: purely logical valid threshold
    mask_grid = (mr_grid > 100).astype(np.float32)
    mask_nifti = nib.Nifti1Image(mask_grid, affine)
    nib.save(mask_nifti, str(patient_dir / "mask.nii.gz"))

def create_smoke_test_environment(target_root: Path, n_patients: int = 3):
    """
    Injects exactly n_patients into the isolated test directory safely, 
    clobbering it if it already exists to purge old geometric corruptions.
    """
    logger = logging.getLogger(__name__)
    if target_root.exists():
        shutil.rmtree(target_root)
        
    for p in range(1, n_patients + 1):
        pid = f"DUMMY_PA_{p:03d}"
        generate_synthetic_patient(target_root / pid)
        
    logger.info(f"Injecting smoke-test environment containing {n_patients} artificial slices.")
