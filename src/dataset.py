"""
dataset.py - Data Loading & Patient-Level Cross-Validation Splitter

Medical ML Rationale:
Data leakage is a severe risk in medical imaging. If splits occur at the slice
level, slices from the same patient could appear in both train and validation
sets, leading to falsely inflated metrics. The KFoldPatientSplitter ensures
splits strictly occur at the patient level.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Constants
MRI_PLOW = 0.5
MRI_PHIGH = 99.5
CT_HU_MIN = -1000.0
CT_HU_MAX = 3000.0

def normalise_mri(volume: np.ndarray) -> np.ndarray:
    """Robust percentile-based min-max normalisation for MRI to [0, 1]."""
    lo = float(np.percentile(volume, MRI_PLOW))
    hi = float(np.percentile(volume, MRI_PHIGH))
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo + 1e-8)
    return volume.astype(np.float32)

def normalise_ct(volume: np.ndarray) -> np.ndarray:
    """Linear HU normalisation for CT clipped to [-1000, 3000] and scaled to [-1, 1]."""
    volume = np.clip(volume, CT_HU_MIN, CT_HU_MAX)
    span = CT_HU_MAX - CT_HU_MIN
    volume = (volume - CT_HU_MIN) / span * 2.0 - 1.0
    return volume.astype(np.float32)

def pad_or_crop_center(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Ensure consistent spatial dimensions for UNet batching (divisible by 16)."""
    h, w = img.shape
    th, tw = target_size, target_size

    # Pad if smaller
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h > 0 or pad_w > 0:
        pt, pb = pad_h // 2, pad_h - (pad_h // 2)
        pl, pr = pad_w // 2, pad_w - (pad_w // 2)
        # Use edge padding to avoid harsh artifacts against zero background
        img = np.pad(img, ((pt, pb), (pl, pr)), mode='edge')

    # Crop if larger
    h, w = img.shape
    if h > th or w > tw:
        sh, sw = (h - th) // 2, (w - tw) // 2
        img = img[sh:sh+th, sw:sw+tw]
        
    return img

class MRICTDataset(Dataset):
    """
    Single-task (Brain) MRI to CT Dataset.
    Loads pre-filtered patient IDs to strictly enforce patient-level splits.
    Memory-efficient: reads slices from disk.
    """
    def __init__(self, data_root: Path, patient_ids: List[str], augment: bool = False):
        super().__init__()
        self.data_root = Path(data_root)
        self.patient_ids = patient_ids
        self.augment = augment
        self._index: List[Dict[str, Any]] = []
        
        self._build_index()

    def _build_index(self) -> None:
        logger.info(f"Building Dataset index for {len(self.patient_ids)} patients...")
        
        for pid in self.patient_ids:
            pdir = self.data_root / pid
            mr_path = pdir / "mr.nii.gz"
            ct_path = pdir / "ct.nii.gz"
            mask_path = pdir / "mask.nii.gz"
            
            if not (mr_path.exists() and ct_path.exists()):
                continue
                
            # Open header to find dimensions without loading entire volume
            mr_img = nib.load(mr_path)
            mr_shape = mr_img.shape
            
            # Assume axial slices are on axis 2
            n_slices = mr_shape[2]
            
            for i in range(n_slices):
                self._index.append({
                    "mr_path": mr_path,
                    "ct_path": ct_path,
                    "pid": pid,
                    "slice_idx": i
                })
                
        logger.info(f"Dataset ready: {len(self._index)} slices indexed.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        item = self._index[idx]
        
        # Load specific slice from disk
        mr_img = nib.load(item["mr_path"])
        ct_img = nib.load(item["ct_path"])
        
        # Slicer: get_fdata() on a specific slice is memory efficient
        mr_slice = mr_img.dataobj[:, :, item["slice_idx"]]
        ct_slice = ct_img.dataobj[:, :, item["slice_idx"]]
        
        mr_np = np.array(mr_slice, dtype=np.float32)
        ct_np = np.array(ct_slice, dtype=np.float32)
        
        mr_np = normalise_mri(mr_np)
        ct_np = normalise_ct(ct_np)
        
        if self.augment and random.random() < 0.5:
            mr_np = np.fliplr(mr_np)
            ct_np = np.fliplr(ct_np)
            
        mr_np = pad_or_crop_center(mr_np, 256)
        ct_np = pad_or_crop_center(ct_np, 256)

        # PyTorch requires (C, H, W) where C=1
        mr_t = torch.from_numpy(np.ascontiguousarray(mr_np[np.newaxis])).to(torch.float32)
        ct_t = torch.from_numpy(np.ascontiguousarray(ct_np[np.newaxis])).to(torch.float32)
        
        return {
            "mr": mr_t,
            "ct": ct_t,
            "patient_id": item["pid"],
            "slice_idx": item["slice_idx"]
        }


class KFoldPatientSplitter:
    """
    Manages deterministic 3-way splits (Train/Val/Test) at the patient level.
    Ensures a dedicated hold-out test set is never seen by the Cross-Validation folds.
    """
    def __init__(self, data_root: str | Path, k: int = 3, test_ratio: float = 0.12, seed: int = 42):
        self.data_root = Path(data_root)
        self.k = k
        self.seed = seed
        self.test_ratio = test_ratio
        
        all_ids = self._get_patient_ids()
        if not all_ids:
            logger.warning(f"No patient directories found in {self.data_root}.")
            self.test_ids = []
            self.pool_ids = []
            self.folds = []
            return

        # 1. Deterministic hold-out test set extraction
        random.seed(self.seed)
        shuffled = list(all_ids)
        random.shuffle(shuffled)
        
        n_test = max(1, int(len(shuffled) * self.test_ratio))
        self.test_ids = sorted(shuffled[:n_test])
        self.pool_ids = sorted(shuffled[n_test:])
        
        logger.info(f"3-Way Split: {len(all_ids)} total patients -> {len(self.pool_ids)} Pool, {len(self.test_ids)} Test")

        # 2. Setup K-Fold on the remaining Pool
        if self.k > 1:
            kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
            self.folds = list(kf.split(self.pool_ids))
        else:
            # Fallback for k=1 (single deterministic train/val split 80/20 in the pool)
            random.seed(self.seed)
            p_shuffled = list(self.pool_ids)
            random.shuffle(p_shuffled)
            split_idx = int(0.8 * len(p_shuffled))
            self.folds = [(list(range(split_idx)), list(range(split_idx, len(p_shuffled))))]

    def _get_patient_ids(self) -> List[str]:
        if not self.data_root.exists():
            return []
        pids = [d.name for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        return sorted(pids)

    def get_fold_ids(self, fold_k: int) -> Tuple[List[str], List[str]]:
        """Return (train_ids, val_ids) for the requested fold from the pool."""
        if fold_k >= len(self.folds):
            raise ValueError(f"Fold {fold_k} out of range for k={self.k}")
        train_idx, val_idx = self.folds[fold_k]
        train_ids = [self.pool_ids[i] for i in train_idx]
        val_ids = [self.pool_ids[i] for i in val_idx]
        return train_ids, val_ids

    def get_fold_loaders(self, fold_k: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Build train/val dataloaders for a specific fold."""
        train_ids, val_ids = self.get_fold_ids(fold_k)
        logger.info(f"Fold {fold_k}: {len(train_ids)} train pts | {len(val_ids)} val pts")
        
        train_ds = MRICTDataset(self.data_root, train_ids, augment=True)
        val_ds   = MRICTDataset(self.data_root, val_ids,   augment=False)
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                       num_workers=2, pin_memory=True, persistent_workers=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, 
                       num_workers=2, pin_memory=True, persistent_workers=True)
        )

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Build the dedicated hold-out test set loader."""
        logger.info(f"Loading Test Set: {len(self.test_ids)} patients")
        test_ds = MRICTDataset(self.data_root, self.test_ids, augment=False)
        return DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                          num_workers=2, pin_memory=True)
