# Installation & Setup Guide

This guide walks you through setting up the project environment and datasets for training and evaluation.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Clone Repository](#1-clone-repository)
3. [Python Environment Setup](#2-python-environment-setup)
4. [Verify Installation](#3-verify-installation)
5. [Download Datasets](#4-download-datasets)
6. [Prepare Datasets](#5-prepare-datasets)
7. [Download Pretrained Weights](#6-download-pretrained-weights)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware (Recommended)
- **GPU**: NVIDIA GPU with CUDA compute capability ≥ 7.0 (RTX series or better)
- **Memory**: ≥ 24GB VRAM for 4×GPU training; ≥ 8GB for single GPU
- **CPU**: 8+ cores recommended
- **Disk**: ≥ 100GB for datasets and checkpoints
- **OS**: Linux (Ubuntu 20.04+ tested), macOS (limited GPU support), Windows WSL2

### Software
- **CUDA**: 12.4 or compatible version
- **cuDNN**: 9.0 or compatible
- **Python**: 3.10, 3.11 (see `pyproject.toml`)

### CPU-Only Mode
- Works but significantly slower (~10-50× slower than GPU)
- Requires `torch` CPU build in environment

---

## 1. Clone Repository

```bash
git clone https://github.com/edison44851/Detection-and-Recognition-From-Aerial-Drones-FYP.git
cd Detection-and-Recognition-From-Aerial-Drones-FYP
```

---

## 2. Python Environment Setup

### Option A: Using `uv` (Recommended - Fastest)

If `uv` is not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate environment:
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\activate  (Windows)

# Install dependencies
uv pip install -e .
```

### Option B: Using Conda

```bash
conda create -n fyp python=3.10
conda activate fyp

# Install PyTorch with CUDA 12.4 support
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# Install remaining dependencies
pip install -e .
```

### Option C: Using venv (Standard Python)

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\activate  (Windows)

# Install PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -e .
```

### Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output (GPU):
```
PyTorch: 2.6.0
CUDA Available: True
Device: NVIDIA RTX 4090
```

---

## 3. Verify Installation

Run the quick verification script:

```bash
python3 tools/quick_train_check.py
```

**Expected output:**
```
✓ Trainer initialized successfully
✓ Dataset loaded: 2 samples
✓ Model forward pass completed
✓ Installation verified!
```

If you see errors, check the [Troubleshooting](#troubleshooting) section.

### Optional: Verify Detection Head Setup

```bash
python3 tools/verify_score_fix.py
```

This validates detection scores are in proper range [0, 1] without compression artifacts.

---

## 4. Download Datasets

### DroneRGBT Dataset

**Source**: https://github.com/VisDrone/DroneRGBT

1. Visit the repository and follow their download instructions
2. Create `.data/` directory in project root:
   ```bash
   mkdir -p .data/DroneRGBT
   ```
3. Extract downloaded files to this structure:
   ```
   .data/DroneRGBT/
   ├── Train/
   │   ├── RGB/              # RGB images
   │   ├── Infrared/        # Thermal images (named *R.jpg)
   │   └── GT_/             # XML annotations (named *R.xml)
   └── Test/
       ├── RGB/
       ├── Infrared/
       └── GT_/
   ```

### RGBT-CC Dataset (Optional)

**Source**: https://github.com/chen-judge/RGBTCrowdCounting

1. Download from the official repository
2. Extract to:
   ```
   .data/RGBT-CC/
   ├── train/               # *_RGB.jpg, *_T.jpg, *_GT.json
   ├── val/
   └── test/
   ```

---

## 5. Prepare Datasets

Convert raw datasets to internal format (`*_RGB.jpg`, `*_T.jpg`, `*_GT.npy`):

### DroneRGBT Conversion

```bash
python3 tools/convert_dronergbt.py \
  --src-root .data/DroneRGBT \
  --out-root .data/DroneRGBT_converted
```

**Output structure:**
```
.data/DroneRGBT_converted/
├── train/
│   ├── 1_RGB.jpg, 1_T.jpg, 1_GT.npy
│   ├── 2_RGB.jpg, 2_T.jpg, 2_GT.npy
│   └── ...
└── test/
    └── ...
```

### RGBT-CC Conversion (Optional)

```bash
python3 tools/convert_rgbtcc.py \
  --src-root .data/RGBT-CC \
  --out-root .data/RGBT-CC_converted
```

### Compute Thermal Statistics (RGBT-CC Only)

Required for proper thermal image normalization:

```bash
# For training split only
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --split train

# For all splits
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --all-splits
```

**Example output:**
```
Computing thermal statistics for .data/RGBT-CC_converted/train/
Mean: [0.492, 0.168, 0.430]
Std: [0.317, 0.174, 0.191]
```

Use these values when training on RGBT-CC. The computed statistics are applied automatically by the dataset loader when training on `RGBT-CC_converted`.

---

## 6. Download Pretrained Weights

### Phase 6.5 Checkpoint (Recommended)

Download from Hugging Face:
```bash
# Using huggingface_hub CLI
pip install huggingface_hub

mkdir -p checkpoints_phase6/phase6.5_better_bias
huggingface-cli download Edison2525/Detection-and-Recognition-From-Aerial-Drones-FYP \
  phase6_5_best_model_epoch_68.pth \
  --repo-type model \
  --local-dir checkpoints_phase6/phase6.5_better_bias
```

Or manually:
1. Visit: https://huggingface.co/Edison2525/Detection-and-Recognition-From-Aerial-Drones-FYP
2. Download `phase6_5_best_model_epoch_68.pth`
3. Place in `checkpoints_phase6/phase6.5_better_bias/`

### Verify Checkpoint

```bash
python3 -c "
import torch
ckpt = torch.load('checkpoints_phase6/phase6.5_better_bias/phase6_5_best_model_epoch_68.pth')
print('Checkpoint keys:', ckpt.keys() if isinstance(ckpt, dict) else 'Direct state dict')
print('Training epochs completed:', ckpt.get('epoch', 'Unknown'))
"
```

---

## Troubleshooting

### CUDA/GPU Issues

**Problem**: `CUDA out of memory`
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB on cuda:0
```

**Solutions**:
```bash
# 1. Reduce batch size
# Lower BATCH_SIZE in tools/train_entry.sh, then run:
bash tools/train_entry.sh

# 2. Freeze more components
# Enable the freeze variables in tools/train_entry.sh, then run:
bash tools/train_entry.sh

# 3. Use CPU mode (last resort)
# Set DEVICE in tools/train_entry.sh, then run:
bash tools/train_entry.sh
```

**Problem**: `No CUDA devices detected`
```
CUDA Available: False
```

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Or use CPU-only PyTorch
pip install torch torchvision torchaudio
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Ensure environment is activated
```bash
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\activate  (Windows)

# Reinstall
uv pip install -e .
```

**Problem**: `ImportError: cannot import name 'F' from 'timm'`

**Solution**: Check TIMM version
```bash
python -c "import timm; print(timm.__version__)"

# Should be >= 1.0.0
pip install -U timm
```

### Dataset Issues

**Problem**: `FileNotFoundError: .data/DroneRGBT_converted/train/*_RGB.jpg`

**Solutions**:
```bash
# Verify conversion completed
ls -la .data/DroneRGBT_converted/train/ | head -20

# Re-run conversion
python3 tools/convert_dronergbt.py \
  --src-root .data/DroneRGBT \
  --out-root .data/DroneRGBT_converted \
  --force  # overwrite if exists
```

**Problem**: Thermal image loading fails (RGBT-CC)

**Solution**: Ensure thermal statistics are computed:
```bash
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --all-splits
```

### Training Issues

**Problem**: `NaN` loss during training

**Solutions**:
```bash
# 1. Lower learning rate
# Lower HEAD_LR in tools/train_entry.sh, then run:
bash tools/train_entry.sh

# 2. Increase training epochs
# Raise MAX_EPOCH in tools/train_entry.sh, then run:
bash tools/train_entry.sh

# 3. Use Phase 6.5 config as reference (see docs/TRAINING.md)
```

**Problem**: Low detection accuracy

**Solutions**:
```bash
# 1. Verify dataset conversion
python3 tools/quick_train_check.py

# 2. Increase training epochs
# Raise MAX_EPOCH in tools/train_entry.sh, then run:
bash tools/train_entry.sh

# 3. Use Phase 6.5 config as reference (see docs/TRAINING.md)
```

### Environment Variables

Set these for reproducible runs:

```bash
export PYTHONHASHSEED=0
export CUDA_LAUNCH_BLOCKING=1  # Safer error reporting
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs
```

### Getting Help

1. Check [docs/FAQ.md](FAQ.md) for common questions
2. Review [docs/KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) for known problems
3. Inspect error logs in `.tmp_posttrain/` directory
4. Compare your config with examples in README.md
