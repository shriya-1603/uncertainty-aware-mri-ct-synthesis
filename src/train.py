"""
train.py - Core Training Loops and Cross-Validation Dispatcher

Handles K-Fold patient-level training and deterministic ensemble training mechanisms.

Medical ML Rationale:
Training individual folds maintains strict separation of patient data to avoid
radiological leakage. Using L1 loss explicitly models physical preservation of 
HU values necessary for downstream clinical dose calculations, contrasting with
L2 loss which disproportionately punishes high-intensity bone/metal structures.
"""

import sys
import logging
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import KFoldPatientSplitter
from model import build_model, UNetBase

# Conditional TPU Support
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# Hardware Strategy
# =============================================================================

def get_device(verbose=True):
    # 1. Check for TPU (XLA)
    if XLA_AVAILABLE:
        try:
            device = xm.xla_device()
            if verbose: logger.info(f"Using device: TPU (XLA) | {device}")
            return device
        except Exception:
            pass # Fallback to CUDA/MPS/CPU

    # 2. Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose: logger.info("Using device: CUDA (NVIDIA GPU)")
    # 3. Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose: logger.info("Using device: MPS (Apple Silicon GPU)")
    # 4. Fallback to CPU
    else:
        device = torch.device("cpu")
        if verbose: logger.info("Using device: CPU")
    return device

# =============================================================================
# Training Engine
# =============================================================================

def train_epoch(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device,
    epoch: int = 0,
    epochs: int = 0,
    arch: str = "",
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    model.train()
    running_loss = 0.0
    is_tty = sys.stdout.isatty()
    n_batches = len(loader)
    
    pbar = tqdm(loader, desc="  Train Mode", leave=False, disable=not is_tty)
    for i, batch in enumerate(pbar):
        mr = batch["mr"].to(device, dtype=torch.float32, non_blocking=True)
        ct = batch["ct"].to(device, dtype=torch.float32, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 1. Forward Pass (Autocast context for AMP)
        # Only use autocast if on CUDA/TPU and scaler/XLA is enabled
        use_amp = (device.type == "cuda" and scaler is not None)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(mr)
            loss = criterion(pred, ct)
        
        # 2. Backward Pass & Optimizer Step
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if XLA_AVAILABLE and device.type == "xla":
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
        
        running_loss += loss.item() * mr.size(0)
        
        # --- Neat Colab Logging ---
        if not is_tty and (i + 1) % 50 == 0:
            avg_l1 = running_loss / ((i + 1) * loader.batch_size)
            print(f"  [{arch}] Epoch {epoch}/{epochs} | Batch {i+1}/{n_batches} | Train L1: {avg_l1:.4f}", flush=True)
        
        # Synchronization and Performance Monitoring (TTY only)
        if is_tty:
            if (i + 1) % 50 == 0:
                if device.type == "mps":
                    torch.mps.synchronize()
                    mem = f"{torch.mps.current_allocated_memory()/1e9:.1f}GB"
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                    mem = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                elif XLA_AVAILABLE and device.type == "xla":
                    xm.mark_step()
                    mem = "XLA"
                else:
                    mem = "N/A"
                pbar.set_postfix({"L1": f"{loss.item():.4f}", "mem": mem})
            else:
                pbar.set_postfix({"L1": f"{loss.item():.4f}"})
        
    if device.type == "mps": torch.mps.synchronize()
    elif device.type == "cuda": torch.cuda.synchronize()
    elif XLA_AVAILABLE and device.type == "xla": xm.mark_step()
    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate_epoch(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    epoch: int = 0,
    epochs: int = 0,
    arch: str = ""
) -> float:
    # Ensure standard BN and Dropout are locked down during evaluation sweeps
    model.eval()
    running_loss = 0.0
    is_tty = sys.stdout.isatty()
    n_batches = len(loader)
    
    pbar = tqdm(loader, desc="  Valid Mode", leave=False, disable=not is_tty)
    for i, batch in enumerate(pbar):
        mr = batch["mr"].to(device, dtype=torch.float32, non_blocking=True)
        ct = batch["ct"].to(device, dtype=torch.float32, non_blocking=True)
        
        pred = model(mr)
        loss = criterion(pred, ct)
        
        running_loss += loss.item() * mr.size(0)
        
        if not is_tty and (i + 1) % 50 == 0:
            avg_l1 = running_loss / ((i + 1) * loader.batch_size)
            print(f"  [{arch}] Epoch {epoch}/{epochs} | Val Batch {i+1}/{n_batches} | Val L1: {avg_l1:.4f}", flush=True)

        if XLA_AVAILABLE and device.type == "xla" and (i + 1) % 50 == 0:
            xm.mark_step()
            
    if XLA_AVAILABLE and device.type == "xla": xm.mark_step()
    return running_loss / len(loader.dataset)

def save_checkpoint(
    model: nn.Module, 
    param_dict: dict, 
    path: Path, 
    optimizer: Optional[optim.Optimizer] = None, 
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    epoch: int = 0
):
    """Serialize model parameters and training state (opt/sch/epoch) to disk safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "arch": param_dict["arch"],
        "fold": param_dict["fold"],
        "epoch": epoch
    }
    if optimizer:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler:
        state["scheduler_state"] = scheduler.state_dict()
        
    if XLA_AVAILABLE and isinstance(model.device if hasattr(model, 'device') else next(model.parameters()).device, torch.device) and (next(model.parameters()).device.type == "xla"):
        xm.save(state, str(path))
    elif XLA_AVAILABLE and str(next(model.parameters()).device).startswith('xla'):
        xm.save(state, str(path))
    else:
        torch.save(state, str(path))

# =============================================================================
# K-Fold & Ensemble Routines
# =============================================================================

def train_kfold(
    arch: str,
    fold_k: int,
    data_root: Path,
    checkpoint_dir: Path,
    epochs: int = 15,
    batch_size: int = 8,
    cv_folds: int = 3,
    seed: int = 42,
    tag: str = "primary",
    resume: bool = False
) -> float:
    """
    Executes a complete training cycle for a specific fold isolated by Patient ID.
    Supports resumption from the latest available checkpoint.
    """
    device = get_device()
    logger.info(f"--- Launching Fold {fold_k} | {arch} | Device {device} | Resume {resume} ---")
    
    # 1. Reproducibility
    torch.manual_seed(seed)
    
    # 2. Build Loaders safely
    splitter = KFoldPatientSplitter(data_root=data_root, k=cv_folds, seed=seed)
    train_loader, val_loader = splitter.get_fold_loaders(fold_k, batch_size=batch_size)
    
    # 3. Assemble Network & Optimisers
    model = build_model(arch).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Gradient Scaler for AMP (Mixed Precision)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    start_epoch = 1
    best_loss = float("inf")
    
    # 4. Resume Logic
    if resume:
        ckpt_path = checkpoint_dir / f"latest_{tag}_fold{fold_k}.pth"
        if not ckpt_path.exists():
            ckpt_path = checkpoint_dir / f"best_{tag}_fold{fold_k}.pth"
            
        if ckpt_path.exists():
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            # Always load to CPU first to avoid device-tagging issues (especially for TPU/XLA)
            state = torch.load(str(ckpt_path), map_location='cpu')
            model.load_state_dict(state["model_state"])
            if "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])
            if "scheduler_state" in state:
                scheduler.load_state_dict(state["scheduler_state"])
            start_epoch = state.get("epoch", 0) + 1
            logger.info(f"Restarting at Epoch {start_epoch}")
        else:
            logger.warning(f"Resume requested but no checkpoint found at {checkpoint_dir}. Starting fresh.")
    
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 epoch=epoch, epochs=epochs, arch=arch, scaler=scaler)
        val_loss = validate_epoch(model, val_loader, criterion, device,
                                   epoch=epoch, epochs=epochs, arch=arch)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch:03d}/{epochs} | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f}")
        
        # Checkpoint: Latest (Every Epoch)
        if device.type == "mps": torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()
        elif XLA_AVAILABLE and device.type == "xla": xm.mark_step()
        
        latest_path = checkpoint_dir / f"latest_{tag}_fold{fold_k}.pth"
        save_checkpoint(model, {"arch": arch, "fold": fold_k}, latest_path, optimizer, scheduler, epoch)
        
        # Checkpoint: Best
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = checkpoint_dir / f"best_{tag}_fold{fold_k}.pth"
            save_checkpoint(model, {"arch": arch, "fold": fold_k}, best_path, optimizer, scheduler, epoch)
            
        # Checkpoint: Milestones (Every 10)
        if epoch % 10 == 0:
            int_path = checkpoint_dir / f"epoch_{tag}_fold{fold_k}_{epoch:03d}.pth"
            save_checkpoint(model, {"arch": arch, "fold": fold_k}, int_path, optimizer, scheduler, epoch)
            
    return best_loss


def train_ensemble(
    arch: str,
    target_fold: int,
    data_root: Path,
    checkpoint_dir: Path,
    n_members: int = 5,
    epochs: int = 50,
    batch_size: int = 8,
    cv_folds: int = 3
) -> None:
    """
    Trains decoupled models forming a Deep Ensemble precisely localized onto a single best fold.
    """
    for n in range(n_members):
        logger.info(f"==> Training Ensemble Member {n+1}/{n_members} (Seed {n})")
        train_kfold(
            arch=arch,
            fold_k=target_fold,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            epochs=epochs,
            batch_size=batch_size,
            cv_folds=cv_folds,
            seed=n, # independent loader seeds!
            tag=f"ensemble_m{n}"
        )

# =============================================================================
# Checkpoint Loaders
# =============================================================================

def load_fold_model(arch: str, fold: int, checkpoint_dir: Path, tag: str = "primary", epoch: Optional[int] = None) -> nn.Module:
    """Hydrates a UNet state evaluated to testing mode."""
    device = get_device()
    model = build_model(arch).to(device)
    
    if epoch:
        ckpt_path = checkpoint_dir / f"epoch_{tag}_fold{fold}_{epoch:03d}.pth"
    else:
        ckpt_path = checkpoint_dir / f"best_{tag}_fold{fold}.pth"
        
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing state dict: {ckpt_path}")
        
    # Always load to CPU first for cross-platform portability 
    state = torch.load(str(ckpt_path), map_location='cpu')
    model.load_state_dict(state["model_state"])
    model.eval()
    return model

def load_ensemble(arch: str, fold: int, checkpoint_dir: Path, n_members: int = 5) -> List[nn.Module]:
    """Hydrates the complete stack of independently seeded deep ensembles."""
    models = []
    for n in range(n_members):
        model = load_fold_model(arch, fold, checkpoint_dir, tag=f"ensemble_m{n}")
        models.append(model)
    return models
