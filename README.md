# Uncertainty-Aware MRI-to-CT Synthesis

> Attention Ablation & Uncertainty Quantification on the SynthRAD2023 Brain Subset  
> MS Capstone · Georgia State University · Spring 2026

---

## Overview

Radiotherapy planning requires Hounsfield Unit (HU) accurate CT data for dose calculation. Acquiring both MRI and CT scans increases patient radiation exposure, cost, and registration error — particularly critical for pediatric patients.

This project builds a deep learning pipeline that synthesizes physically accurate CT scans directly from MRI, enabling **MRI-only radiotherapy workflows**. Beyond synthesis accuracy, the system includes an **uncertainty quantification engine** that generates voxel-wise confidence maps alongside every prediction — flagging where the model is likely to be wrong before a clinician ever reviews the output.

---

## Visualizations

### Synthesis Output — MRI to Synthetic CT
*MRI Input · Ground Truth CT · Predicted CT (CBAM) · Error Map · Ensemble Uncertainty*

![Synthesis Output Grid](outputs/figures/synthesis_output_grid.png)

> Each row is a different brain slice. The error map highlights prediction deviation in HU. The uncertainty map shows where the model lacks confidence — note how uncertainty concentrates at bone-soft tissue boundaries, exactly where HU accuracy is most critical.

---

### Uncertainty Quantification — Error vs Predicted Uncertainty
![Uncertainty Correlation](outputs/figures/uncertainty_correlation.png)

> Spearman ρ = 0.7886 (p < 0.0001). The model's uncertainty maps reliably predict actual voxel-level error, enabling clinically actionable confidence signals.

---

## Key Results

| Model | MAE (Normalized) | MAE (HU) | PSNR | SSIM |
|---|---|---|---|---|
| Base U-Net (M1) | 0.0262 | 52.4 HU | 28.01 dB | 0.8603 |
| **CBAM U-Net (M2) — Winner** | **0.0257** | **51.4 HU** | **28.08 dB** | **0.8656** |
| Self-Attention U-Net (M3) | Did not converge competitively | — | — | — |
| Hybrid U-Net (M4) | Marginal over baseline | — | — | — |

![Official Leaderboard](outputs/figures/leaderboard.png)

- **Clinical benchmark**: State-of-the-art brain sCT architectures target 30–60 HU MAE for viable dose planning. All variants are competitive within this range.
- **Uncertainty engine**: Spearman correlation ρ = 0.7886 (p < 0.0001) between predicted uncertainty and actual voxel error — proving the model reliably identifies its own failure regions.

---

## Why U-Net and Not a GAN?

GANs risk generating **anatomical hallucinations** — synthetic bone or air pockets that appear realistic but are physically incorrect. In radiotherapy dose planning, this is a patient safety issue, not just a metrics problem.

U-Net with **L1 loss** enforces strict Hounsfield Unit conservation, ensuring every pixel prediction is physically grounded.

---

## Architecture & Ablation Study

Four U-Net variants were systematically compared via **3-fold cross-validation** (21 model checkpoints total) to isolate the optimal attention mechanism for physical density mapping:

| Variant | Mechanism |
|---|---|
| M1 — Base U-Net | 4-level encoder-decoder, residual double convolutions |
| M2 — CBAM U-Net | Channel + Spatial attention gates at each decoder skip connection |
| M3 — Self-Attention U-Net | Transformer-inspired Non-Local block at 1/16 resolution bottleneck |
| M4 — Hybrid | M2 decoder gates + M3 bottleneck transformer |

**Winner: CBAM U-Net (M2)**  
CBAM's local attention mechanism dynamically re-weights features at complex bone-soft tissue interfaces — precisely where HU accuracy matters most for dose calculation.

---

## Uncertainty Quantification Engine

Two complementary uncertainty estimation methods:

- **MC Dropout** — Epistemic uncertainty via stochastic forward passes at inference
- **Deep Ensembles** — Voxel-wise variance maps across 5 independently trained networks on the best-performing fold

The resulting uncertainty maps correlate strongly with actual prediction error (ρ = 0.79), enabling **clinically actionable confidence signals** alongside every synthetic CT.

---

## Project Structure

```
mri_ct_synthesis/
├── pyproject.toml
├── setup.sh                  # uv environment initializer
├── data/brain/               # SynthRAD2023 Task1 Brain subset
├── checkpoints/              # Model checkpoints (untracked)
├── outputs/
│   ├── figures/              # Visualizations
│   └── logs/                 # Epoch telemetry
└── src/
    ├── dataset.py            # Patient-level splitter, memory-safe data loading
    ├── model.py              # U-Net variants (M1–M4)
    ├── train.py
    ├── uncertainty.py        # MC Dropout, Deep Ensembles, TTA
    ├── inference.py
    ├── evaluation.py
    ├── dose_proxy.py         # Schneider clinical validators, pencil beam approximations
    ├── ablation.py           # Automated architecture sweeps
    ├── utils.py
    └── main.py               # Typer CLI dispatcher
```

---

## Setup & Usage

This project uses [`uv`](https://github.com/astral-sh/uv) for environment management and [`typer`](https://typer.tiangolo.com/) for CLI dispatch.

**Hardware**: Optimized for Apple Silicon (`mps`) with memory-safe data loading for 16GB unified memory. Scaled to Kaggle GPU for full cross-validation and ensemble runs.

### 1. Environment Setup

```bash
bash setup.sh
```

### 2. Train with Cross-Validation

```bash
uv run python -m src.main train --arch unet_cbam --epochs 30 --cv-folds 3
```

### 3. Run Full Ablation Study

```bash
uv run python -m src.main ablation --epochs 30 --cv-folds 3
```

### 4. Train Deep Ensemble

```bash
uv run python -m src.main train --arch unet_cbam --epochs 30 --ensemble 5 --cv-folds 1 --fold 0
```

### 5. Generate Uncertainty Maps

```bash
uv run python -m src.main infer --arch unet_cbam --mc-t 20 --tta-t 10 --fold 0
```

### 6. Clinical Evaluation

```bash
uv run python -m src.main evaluate --cv-folds 3
uv run python -m src.main dose --fold 0
uv run python -m src.main visualize
```

---

## Dataset

**SynthRAD2023 — Task 1 (Brain Subset)**  
A clinical benchmark dataset for MRI-to-CT synthesis in radiotherapy planning.  
Access via the official SynthRAD2023 challenge: [synthrad2023.grand-challenge.org](https://synthrad2023.grand-challenge.org)

---

## Author

**Shriya Kotala**  
MS in Computer Science · Georgia State University · Spring 2026  
[LinkedIn](https://www.linkedin.com/in/shriya-kotala) · [GitHub](https://github.com/shriya-kotala)
