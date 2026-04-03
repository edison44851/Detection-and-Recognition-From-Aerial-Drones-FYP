# Tools & Utilities Reference

This directory contains standalone utility scripts for dataset conversion, evaluation, and diagnostics.

## Table of Contents

- [Dataset Conversion](#dataset-conversion)
- [Training & Inference](#training--inference)
- [Diagnostics & Analysis](#diagnostics--analysis)
- [Comparison & Visualization](#comparison--visualization)

---

## Dataset Conversion

### `convert_dronergbt.py`

**Purpose**: Convert DroneRGBT dataset from XML annotations to internal format

**Usage**:
```bash
python3 tools/convert_dronergbt.py \
  --src-root .data/DroneRGBT \
  --out-root .data/DroneRGBT_converted
```

**Required Arguments**:
- `--src-root`: Path to downloaded DroneRGBT directory (contains `Train/`, `Test/` subdirs)
- `--out-root`: Output directory for converted dataset

**Optional Arguments**:
- `--force`: Overwrite existing output directory
- `--num-workers`: Parallel workers for conversion (default: 4)

**Input Structure**:
```
DroneRGBT/
├── Train/
│   ├── RGB/           # RGB images
│   ├── Infrared/      # Thermal images (*R.jpg)
│   └── GT_/           # Annotations (*R.xml)
└── Test/
    ├── RGB/
    ├── Infrared/
    └── GT_/
```

**Output Structure**:
```
DroneRGBT_converted/
├── train/
│   ├── 1_RGB.jpg, 1_T.jpg, 1_GT.npy
│   └── ...
└── test/
    └── ...
```

**Output Format**:
- `{id}_RGB.jpg`: RGB image (uint8, BGR)
- `{id}_T.jpg`: Thermal image (uint8, single channel)
- `{id}_GT.npy`: Ground truth points (Nx2 array, [x, y] coordinates)

---

### `convert_rgbtcc.py`

**Purpose**: Convert RGBT-CC dataset from JSON annotations to internal format

**Usage**:
```bash
python3 tools/convert_rgbtcc.py \
  --src-root .data/RGBT-CC \
  --out-root .data/RGBT-CC_converted
```

**Required Arguments**:
- `--src-root`: Path to downloaded RGBT-CC directory
- `--out-root`: Output directory for converted dataset

**Optional Arguments**:
- `--force`: Overwrite existing output
- `--num-workers`: Parallel conversion workers (default: 4)

**Input Structure**:
```
RGBT-CC/
├── train/        # *_RGB.jpg, *_T.jpg, *_GT.json
├── val/          # Validation split
└── test/
```

**Output Structure**:
```
RGBT-CC_converted/
├── train/
├── val/
└── test/
```

---

### `compute_thermal_stats.py`

**Purpose**: Compute per-channel mean and standard deviation for thermal images (RGBT-CC only)

**Why Needed**: Thermal images require normalization for consistent training performance

**Usage**:
```bash
# Single split
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --split train

# All splits
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --all-splits
```

**Arguments**:
- `--data-dir`: Path to converted dataset
- `--split`: Compute for single split (train/val/test)
- `--all-splits`: Compute for all splits simultaneously

**Output Example**:
```
Computing thermal statistics for .data/RGBT-CC_converted/train/
Processing 2500 images...
Mean: [0.492, 0.168, 0.430]
Std: [0.317, 0.174, 0.191]
```

**Use in Training**:

The dataset loader automatically applies RGBT-CC thermal normalization when `--data-dir` points to `RGBT-CC_converted`. No additional flags needed.

---

## Training & Inference

### `train_entry.sh`

**Purpose**: Convenient wrapper around `torchrun` and `Fine-tune/train.py`

**Usage**:
```bash
# Configure variables in tools/train_entry.sh, then run:
bash tools/train_entry.sh
```

**Key Script Variables**:
- `DATA_DIR`: Dataset path
- `NPROC`: Number of GPU processes
- `DEVICE`: GPU indices (for example, `0,1,2,3`)
- `BATCH_SIZE`: Batch size per GPU
- `MAX_EPOCH`: Maximum epochs
- `RESUME`: Checkpoint to resume from
- `SAVE_DIR`: Output checkpoint directory

**Phase 6 Defaults** (automatically handled by script):
The following flags are **enabled by default** in this script for Phase 6 models and do not need to be specified:
- `--use-deconv`: Deconvolution upsampling (always on)
- `--use-fpn`: FPN feature pyramid network (always on)
- `--keypoint-mode`: Keypoint-only detection mode (always on)
- `--use-bce-logits`: BCEWithLogitsLoss (always on)
- `--det-use-gn`: GroupNorm in detection head (always on)

Simply run `bash tools/train_entry.sh` after updating the script variables.

**Common Script Variables**:
- `TASK`: `counting` or `detection`
- `HEAD_CONV`: Head channel width
- `DET_SIGMA`: Gaussian sigma
- `FREEZE_BACKBONE`: Freeze Swin backbone
- `FREEZE_UNET`: Freeze U-Net fusion

---

## Diagnostics & Analysis

### `quick_train_check.py`

**Purpose**: Verify training setup without running full training

**What It Does**:
1. Loads trainer with minimal config
2. Replaces dataset with 2-sample subset
3. Runs one forward pass and gradient step
4. Validates no shape mismatches or missing dependencies

**Usage**:
```bash
python3 tools/quick_train_check.py
```

**Expected Output**:
```
✓ Trainer initialized successfully
✓ Dataset loaded: 2 samples
✓ Model forward pass completed
✓ Loss computation successful
✓ Installation verified!
```

**When to Use**:
- After environment setup (verify compatibility)
- After code changes (catch early errors)
- Before launching long training runs

---

### `verify_score_fix.py`

**Purpose**: Validate detection head score ranges and detect compression artifacts

**Why Needed**: Phase 4 bug caused score compression (scores clamped to 0.1-0.45). This script detects such issues.

**Usage**:
```bash
python3 tools/verify_score_fix.py
```

**Output Example**:
```
Loading checkpoint: checkpoints_phase6/phase6.5_better_bias/phase6_5_best_model_epoch_68.pth
Running inference on 100 test images...

Score Statistics:
  Min: 0.001
  Max: 0.999
  Mean: 0.350
  Std: 0.285
  
✓ PASS: Score distribution normal (range [0, 1])
✓ PASS: No evidence of compression or clamping
```

**Warning Signs** (indicates issues):
- Min score > 0.05 (compressed low end)
- Max score < 0.95 (compressed high end)
- Score std < 0.15 (too narrow distribution)

---

### `run_posttrain_diagnostics.sh`

**Purpose**: Comprehensive multi-mode evaluation with parameter sweeps

**Usage**:
```bash
bash tools/run_posttrain_diagnostics.sh
```

**Or with custom output directory**:
```bash
# Update the variables at the top of tools/run_posttrain_grid.sh, then run:
bash tools/run_posttrain_grid.sh
```

**What It Evaluates**:
1. **Three inference modes**: RAW, TILES, ORIG
2. **Parameter sweeps**:
   - Score thresholds: 0.1, 0.2, ..., 0.9
   - NMS radii: 1.0, 2.0, 4.0, 8.0
   - Tile sizes: 256, 512 (for TILES mode)
3. **Metrics per mode**: AP, Precision, Recall, F1
4. **Per-prediction analysis**: CSVs with scores and GT distances

**Output Structure**:
```
.tmp_posttrain/
├── RAW/
│   ├── metrics_summary.csv
│   ├── scores.csv
│   └── visualizations/
├── TILES/
│   └── ...
├── ORIG/
│   └── ...
└── comparison.png  # PR curves for all modes
```

**Time Estimate**: ~30-60 min on single GPU for 1806 images

---

### `run_posttrain_grid.sh`

**Purpose**: Parametric version of diagnostics with custom settings

**Usage**:
```bash
bash tools/run_posttrain_grid.sh
```

**Example**:
```bash
# Update CKPT, DATA_DIR, OUT_ROOT, NUM, and DOWNSAMPLE in the script, then run:
bash tools/run_posttrain_grid.sh
```

---

## Comparison & Visualization

### `calculate_ap_pr_curve.py`

**Purpose**: Compute AP and PR curves from predictions and ground truth

**Supports Multiple Input Formats**:
- JSON (Detectron2, YOLO)
- CSV (Phase model predictions)

**Usage**:

**From baseline JSON**:
```bash
python3 tools/calculate_ap_pr_curve.py --json path/to/eval.json
```

**From Phase CSV**:
```bash
python3 tools/calculate_ap_pr_curve.py \
  --csv path/to/scores.csv \
  --gt-count 54391  # Total GT objects
```

**Compare Multiple Models**:
```bash
python3 tools/calculate_ap_pr_curve.py \
  --compare baseline1.json baseline2.json phase_scores.csv \
  --names "YOLO26s RGB" "YOLO26s Thermal" "Phase 6.3" \
  --plot pr_comparison.png
```

**Output**:
```
YOLO26s Thermal:
  AP@8px: 0.6210
  Precision: 0.6685
  Recall: 0.8111
  F1: 0.7329

Phase 6.3:
  AP@8px: 0.7018
  Precision: 0.7868
  Recall: 0.7204
  F1: 0.7521
```

**Options**:
- `--json`: JSON file with predictions
- `--csv`: CSV file with prediction scores
- `--gt-count`: Total ground truth count (for CSV)
- `--distance-threshold`: Matching distance (default: 8px)
- `--plot`: Save PR curve visualization
- `--output-json`: Save results as JSON

---

### `generate_comparison_charts.py`

**Purpose**: Generate comparison visualizations across phases and datasets

**Usage**:
```bash
python3 tools/generate_comparison_charts.py \
  --results-dir .tmp_posttrain/ \
  --output comparison_charts.png
```

**Generates**:
- Phase-by-phase AP progression
- Precision-Recall trade-offs
- Mode comparison (RAW vs TILES vs ORIG)
- Threshold sensitivity analysis

---

## Dataset Utilities

### `convert_droneRGBT_to_coco.py`

**Purpose**: Convert DroneRGBT to COCO format (for external tools)

**Usage**:
```bash
python3 tools/convert_droneRGBT_to_coco.py \
  --src-root .data/DroneRGBT_converted \
  --output coco_annotations.json
```

**Output**: COCO-format JSON for Detectron2, MMDetection, etc.

---

### `convert_droneRGBT_ultralytics.py`

**Purpose**: Convert DroneRGBT to YOLO format

**Usage**:
```bash
python3 tools/convert_droneRGBT_ultralytics.py \
  --src-root .data/DroneRGBT_converted \
  --output-dir .data/DroneRGBT_yolo
```

---

### `create_masked_dronergbt.py`

**Purpose**: Create masked variants for ablation studies (RGB-only, Thermal-only)

**Usage**:
```bash
python3 tools/create_masked_dronergbt.py \
  --src-root .data/DroneRGBT_converted \
  --output-rgb-only .data/DroneRGBT_rgb_only \
  --output-thermal-only .data/DroneRGBT_thermal_only
```

---

## Quick Reference

| Tool | Purpose | Time |
|------|---------|------|
| `quick_train_check.py` | Verify setup | ~30 sec |
| `verify_score_fix.py` | Check score ranges | ~2 min |
| `compute_thermal_stats.py` | Thermal normalization | ~5 min |
| `run_posttrain_diagnostics.sh` | Full evaluation | 30-60 min |
| `calculate_ap_pr_curve.py` | Compute AP | ~10 min |

