"""
inference.py - Execution Router for Uncertainty Estimation

Coordinates the extraction of uncertainty maps utilizing MC Dropout,
Deep Ensembles, and Test-Time Augmentation across validation sets.

Memory-efficient: strictly streams slice-level evaluations directly to highly 
organized patient directories on disk, never holding the complete test envelope 
in main memory to avoid MPS swap-thrashing.
"""

import logging
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import KFoldPatientSplitter
from train import get_device, load_fold_model, load_ensemble
from uncertainty import get_mc_dropout_uncertainty, get_deep_ensemble_uncertainty, get_tta_uncertainty

logger = logging.getLogger(__name__)

def run_inference(
    arch: str,
    fold: int,
    data_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    mc_t: int = 20,
    tta_t: int = 10,
    ensemble: int = 0,
    mode: str = "val"
) -> None:
    """
    Main execution loop extracting deterministic predictions and uncertainty bounds.
    Can operate on either a validation fold or the dedicated hold-out test set.
    """
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"--- Running Inference | Arch: {arch} | Mode: {mode} | Fold: {fold} ---")
    
    # 1. Build Loader dynamically based on mode
    splitter = KFoldPatientSplitter(data_root=data_root, k=3, seed=42)
    
    if mode == "test":
        val_loader = splitter.get_test_loader(batch_size=1)
        save_tag = "test"
    else:
        _, val_loader = splitter.get_fold_loaders(fold, batch_size=1)
        save_tag = f"fold{fold}"
    
    # 2. Boot networks onto MPS natively in eval mode
    try:
        primary_model = load_fold_model(arch, fold, checkpoint_dir, tag="primary")
    except FileNotFoundError:
        logger.error(f"Cannot find primary checkpoint for fold {fold}. Aborting.")
        return

    ensemble_models = []
    if ensemble > 0:
        logger.info(f"Hydrating {ensemble} deep ensemble networks...")
        ensemble_models = load_ensemble(arch, fold, checkpoint_dir, n_members=ensemble)
        
    pbar = tqdm(val_loader, desc=f"Inferencing {save_tag}")
    
    save_root = output_dir / f"{arch}_{save_tag}"
    
    for batch in pbar:
        mr_tensor = batch["mr"].to(device, dtype=torch.float32)
        ct_tensor = batch["ct"].to(device, dtype=torch.float32)
        pid = batch["patient_id"][0]
        sl_idx = batch["slice_idx"][0].item()
        
        # Base Prediction
        with torch.no_grad():
            base_pred = primary_model(mr_tensor).cpu().numpy()
            
        case_block = {
            "patient_id": pid,
            "slice_idx": sl_idx,
            "mri": mr_tensor.cpu().numpy(),
            "ct_gt": ct_tensor.cpu().numpy(),
            "base_pred": base_pred,
            "uncertainty": {}
        }
        
        # =========================================================
        # U1: MC Dropout
        # =========================================================
        case_block["uncertainty"]["mc_dropout"] = get_mc_dropout_uncertainty(
            model=primary_model, mr_tensor=mr_tensor, ct_tensor=ct_tensor, t_passes=mc_t
        )
        
        # =========================================================
        # U2: Deep Ensembles (Optional)
        # =========================================================
        if ensemble > 0:
            case_block["uncertainty"]["deep_ensemble"] = get_deep_ensemble_uncertainty(
                models=ensemble_models, mr_tensor=mr_tensor, ct_tensor=ct_tensor
            )
            
        # =========================================================
        # U3: Test Time Augmentation
        # =========================================================
        case_block["uncertainty"]["tta"] = get_tta_uncertainty(
            model=primary_model, mr_tensor=mr_tensor, ct_tensor=ct_tensor, t_passes=tta_t
        )
        
        # =========================================================
        # OOM Memory Guard
        # =========================================================
        case_dir = save_root / pid
        case_dir.mkdir(parents=True, exist_ok=True)
        dump_path = case_dir / f"slice_{sl_idx:04d}.pkl"
        
        with open(dump_path, "wb") as f:
            pickle.dump(case_block, f)
            
    logger.info(f"Target evaluations successfully persisted per slice inside {save_root}")
