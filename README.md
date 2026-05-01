# Uncertainty-Aware MRI-to-CT Synthesis

**Attention Ablation and Uncertainty Method Comparison on SynthRAD2023 Brain Subset**

This repository contains the complete implementation for translating MRI structures into physically accurate Computed Tomography (CT) Hounsfield Units via deep learning. It rigidly structures execution around epistemic verification, evaluating exactly how well deep generative uncertainty flags physical dose proxy errors prior to clinical proton delivery. 

## Project Constraints
- **Hardware Profile:** Optimized exclusively for Apple Silicon (`mps`). Hard memory caps natively implemented inside `dataset.py` to prevent OOM panics.
- **Task Strictness:** Operates exclusively upon the SynthRAD2023 `Task1` (Brain) subset. No diffusion cascades. No discriminator losses. Pure deterministically bounded L1 metrics. 
- **Ensemble Policy:** Deep Ensembles execute strictly atop the *best independent fold* to retain methodological integrity whilst adhering rigidly to standard 1-week timeline feasibility caps. 

---

## Workspace Architecture

```
mri_ct_synthesis/
├── pyproject.toml
├── setup.sh                  (uv environment initializer)
├── data/brain/               (SynthRAD2023 parent mount mapping)
├── checkpoints/              (safely un-tracked binary drop)
├── outputs/
│   ├── figures/              (rendered seaborn blocks)
│   └── logs/                 (serialized epoch telemetry)
└── src/                      
    ├── dataset.py            (Strict Patient-Level Splitter implementations)
    ├── model.py              (Option C UNet permutations)
    ├── train.py
    ├── uncertainty.py        (Option A mechanisms: MCD, DE, TTA)
    ├── inference.py
    ├── evaluation.py
    ├── dose_proxy.py         (Schneider Clinical Validators + Pencil Beam approximations)
    ├── ablation.py           (Automated Sweeps)
    ├── utils.py
    └── main.py               (Typer Dispatch)
```

---

## 🚀 Execution Guide (Typer CLI)

The pipeline is mathematically wrapped around `uv` and `typer` to prevent parameters deviating maliciously outside of Jupyter blocks. Execute commands functionally using `uv run python -m src.main <command>`.

### 1. Framework Initiation
Bootstraps the identical logical folders natively spanning disk.
```bash
uv run python -m src.main setup
```

### 2. Standard Training & Folding (Example)
Executes deterministic fold rotation mapping to `unet_cbam_sa`.
```bash
uv run python -m src.main train --arch unet_cbam_sa --epochs 30 --cv-folds 3
```

### 3. Deep Ensemble Escalation
Deploys decoupled identical networks onto the clinically optimal fold.
```bash
uv run python -m src.main train --arch unet_cbam_sa --epochs 30 --ensemble 5 --cv-folds 1 --fold 0
```

### 4. Deterministic Ablation Execution
Launches an automated sweep iterating the identical CV-splits flawlessly across `unet`, `unet_cbam`, `unet_sa`, and `unet_cbam_sa`.
```bash
uv run python -m src.main ablation --epochs 30 --cv-folds 3
```

### 5. Uncertainty Generation (Inference)
Translates the `checkpoints` structures evaluating outputs utilizing purely sequential dumps (`.pkl`) rigidly conserving system memory allocation.
```bash
uv run python -m src.main infer --arch unet_cbam_sa --mc-t 20 --tta-t 10 --fold 0
```

### 6. Clinical Assessment Loop
Renders empirical evaluations bridging raw algorithmic errors translating them straight to clinical proxy bounds.
```bash
uv run python -m src.main evaluate --cv-folds 3
uv run python -m src.main dose --fold 0
uv run python -m src.main visualize
```

---

*Authored for MS Project Synthesis Evaluation.*
