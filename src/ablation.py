"""
ablation.py - Architectural Sweep Coordinator

Orchestrates the massive automated training loops across all UNet variants
(Base, CBAM, SA, CBAM+SA) and across all CV folds to empirically construct 
the core MS thesis ablation models.

Medical ML Rationale:
Automating the sequential evaluation ensures that identical randomized splits 
and scheduler conditions are applied systematically across architectures. This
eradicates human-induced inconsistencies when reporting structural validity 
between the Base architecture and the non-local Attention variants.
"""

import logging
from pathlib import Path
from typing import Dict

from train import train_kfold
from model import MODEL_REGISTRY

logger = logging.getLogger(__name__)

def run_ablation_study(
    data_root: Path,
    checkpoint_dir: Path,
    epochs: int = 15,
    batch_size: int = 8,
    cv_folds: int = 3
) -> Dict[str, Dict[int, float]]:
    """
    Sequentially drives patient-level CV training across the 4 core networks.
    """
    logger.info("=====================================================")
    logger.info(f"   STARTING MASTER ABLATION SWEEP | {cv_folds}-Fold CV")
    logger.info("=====================================================")
    
    ablation_results = {}
    
    for arch in MODEL_REGISTRY.keys():
        logger.info(f"\n" + "="*50)
        logger.info(f">>> EVALUATING VARIANT: [ {arch.upper()} ]")
        logger.info("="*50)
        
        arch_results = {}
        
        for fold in range(cv_folds):
            logger.info(f"\n>> Deploying {arch} onto Fold {fold}/{cv_folds-1}")
            try:
                best_val_l1 = train_kfold(
                    arch=arch,
                    fold_k=fold,
                    data_root=data_root,
                    checkpoint_dir=checkpoint_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    cv_folds=cv_folds,
                    seed=42,       # Enforce identical patient splitting across variants
                    tag="primary"
                )
                arch_results[fold] = best_val_l1
                
            except Exception as e:
                logger.error(f"Catastrophic failure in {arch} Fold {fold}: {e}")
                arch_results[fold] = float("inf")
                
        ablation_results[arch] = arch_results
        
    logger.info("\n" + "="*50)
    logger.info("   ABLATION COMPLETION L1-MATRIX (Best Val)")
    logger.info("="*50)
    for arch, val_dict in ablation_results.items():
        logger.info(f"[{arch.ljust(15)}] -> {val_dict}")
        
    return ablation_results
