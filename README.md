# Detection and Recognition From Aerial Drones

This repository is a final-year project (FYP) of graduates at the City University of Hong Kong (CityUHK), supervised by [Prof. Chun Pong LAU](https://scholars.cityu.edu.hk/en/persons/cplau27/). The project aims to develop a robust and efficient solution for detecting and recognizing individuals from aerial RGBT imagery using drones.

The base of this project is built upon the CVPR2025 paper [Free Lunch Enhancements for Multi-modal Crowd Counting](https://github.com/HenryCilence/Free-Lunch-Multimodal-Counting) codebase (Meng et al., 2025). It builds on the original implementation and adds a **CenterNet-style keypoint detection branch** with FPN multi-scale architecture for aerial RGBT imagery.

![Model Architecture](image/FYP-High-level.png "Model Architecture")

## Quick Navigation

### 📖 **Documentation** (Recommended Reading Order)

1. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System design and components
   - U-Net broker, Swin backbone, CenterNet detection head
   - Feature extraction options (full vs broker-only)
   - Inference pipeline and evaluation modes

2. **[DEVELOPMENT.md](docs/DEVELOPMENT.md)** — Development timeline (Phases 1-6)
   - Complete evolution from baseline → Phase 6.5
   - What worked, what failed, and why
   - Key findings and lessons per phase

3. **[TRAINING.md](docs/TRAINING.md)** — How to train
   - Recommended configurations (Phase 6.5, lightweight, multi-scale)
   - Dataset-specific setups (DroneRGBT, RGBT-CC)
   - Hyperparameter tuning, troubleshooting, reproducibility

4. **[INFERENCE.md](docs/INFERENCE.md)** — How to evaluate
   - Three evaluation modes (RAW, TILES, ORIG) explained
   - Parameter tuning (score threshold, NMS radius)
   - Output analysis and metrics interpretation

5. **[KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** — What NOT to do
   - Phase 4 analysis: 6 problematic features and why they're disabled
   - Verification script to check your setup
   - Correct alternatives for FP reduction

6. **[LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md)** — Key insights
   - Why stride-4, 2-layer heads, and heatmap bias matter
   - Why simple fusion beats complex; why theory guides design
   - Principles for future work

### ⚡ **Quick Links**

- [Current Status](#current-status-2026-02-08) — Latest checkpoint and metrics
- [Quick Start](#quick-start) — Single-command training
- [Getting Started](#getting-started) — Setup, datasets, and verification
- [Training Examples](#training--inference-guide) — Copy-paste configurations
- [Datasets](#datasets) — Dataset details and results
- [Troubleshooting](#troubleshooting) — Common issues and fixes

---

## Contents

- Current Status
- Quick Start
- Training & Inference Guide
- Datasets
- Troubleshooting
- Citation & Acknowledgments
- Future Work

## Current Status (2026-02-08)
- **✅ Phase 6.5 Adopted**: Better Bias Initialization (-4.6→-2.0) - **KEPT despite lower RAW/TILES AP**
- **Rationale**: Better TP confidence distribution (more balanced, not concentrated <50% quadrant) + significantly improved ORIG mode (+14.5% AP)
- **Current Best Checkpoint**: `checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth` (AP=0.5622 RAW, 0.5281 TILES, 0.4502 ORIG)
- **AP@15px (Phase 6.5)**: RAW 0.7148, TILES 0.6669, ORIG 0.5590 (more tolerant TP matching)
- **Previous Checkpoint (Phase 6.3)**: `checkpoints_phase6/phase6.3_full_features/best_model_epoch_40.pth` (AP=0.5908 RAW, 0.5622 TILES, 0.3932 ORIG) - archived
- **❌ Phase 6.4 Failed & Reverted**: Better Adaptor (3×3+3×3+1×1) caused catastrophic failure (AP dropped 97%, massive overfitting)
- **Key Trade-off**: Accepted -4.8%/-6.1% RAW/TILES AP reduction for better confidence calibration and +14.5% ORIG improvement

## Quick Start

We strongly recommend using the convenience launcher `tools/train_entry.sh` for training. It wraps `torchrun` safely, sets sensible defaults, and forwards extra flags to `Fine-tune/train.py`.

## Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd FYP
```

### 2. Download Datasets

#### DroneRGBT Dataset
- **Source**: [DroneRGBT Official Repository](https://github.com/VisDrone/DroneRGBT)
- **Structure after download**:
  ```
  .data/DroneRGBT/
  ├── Train/
  │   ├── RGB/              # RGB images
  │   ├── Infrared/        # Thermal images (named *R.jpg)
  │   └── GT_/             # XML annotation files (named *R.xml)
  └── Test/
      ├── RGB/
      ├── Infrared/
      └── GT_/
  ```

#### RGBT-CC Dataset
- **Source**: [RGBT-CC Official Repository](https://github.com/chen-judge/RGBTCrowdCounting)
- **Structure after download**:
  ```
  .data/RGBT-CC/
  ├── train/               # *_RGB.jpg, *_T.jpg, *_GT.json
  ├── val/                 # Validation split
  └── test/                # *_RGB.jpg, *_T.jpg, *_GT.json
  ```

### 3. Download Pretrained Weights

This project extends the **Free-Lunch multimodal counting** framework:
- **Source**: [Free-Lunch Repository](https://github.com/HenryCilence/Free-Lunch-Multimodal-Counting)
- **Required files**: Pretrained backbone weights for Swin Transformer
- Download and place in `checkpoints/` directory (weights are automatically loaded during training if backbone is frozen)

**Current best checkpoint** (Phase 6.5):
- Located at: `checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth`
- Use with `--resume` flag to fine-tune from this checkpoint

### 4. Convert Datasets to Internal Format

Convert raw datasets to the flat `*_RGB.jpg`, `*_T.jpg`, `*_GT.npy` format used by our training pipeline.

#### DroneRGBT Conversion
```bash
python3 tools/convert_dronergbt.py \
  --src-root .data/DroneRGBT \
  --out-root .data/DroneRGBT_converted
```
**Output**: 
- `.data/DroneRGBT_converted/train/` — Training images + annotations
- `.data/DroneRGBT_converted/test/` — Test images + annotations
- Each image generates: `{id}_RGB.jpg`, `{id}_T.jpg`, `{id}_GT.npy`

#### RGBT-CC Conversion
```bash
python3 tools/convert_rgbtcc.py \
  --src-root .data/RGBT-CC \
  --out-root .data/RGBT-CC_converted
```
**Output**:
- `.data/RGBT-CC_converted/train/`, `.../val/`, `.../test/` — All splits converted
- Same file naming scheme as DroneRGBT

### 5. Compute Thermal Image Statistics (RGBT-CC Only)

Thermal images require normalization for consistent training. Compute per-channel mean and standard deviation:

```bash
# Compute stats for training split
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --split train
```

**Expected output**:
```
Computing thermal statistics for .data/RGBT-CC_converted/train/
Mean: [0.492, 0.168, 0.430]
Std: [0.317, 0.174, 0.191]
```

Use `--all-splits` to compute for train/val/test simultaneously:
```bash
python3 tools/compute_thermal_stats.py \
  --data-dir .data/RGBT-CC_converted \
  --all-splits
```

**Configuration**: Add to training config with `--thermal-mean` and `--thermal-std` flags to normalize thermal images during preprocessing.

### 6. Verify Setup (Optional but Recommended)

Run a quick sanity check to validate your training pipeline is working:

#### Quick Training Dry-Run
```bash
python3 tools/quick_train_check.py
```
**Purpose**: 
- Loads trainer with minimal configuration
- Replaces dataset with 2-sample subset
- Validates `trainer.setup()` succeeds without errors
- **Output**: Success/failure message; if successful, you can run full training

#### Verify Detection Head Score Ranges
```bash
python3 tools/verify_score_fix.py
```
**Purpose**:
- Detects if score compression bug exists (Phase 4 issue)
- Validates heatmap scores are properly in [0, 1] range
- **Output**: Score statistics (min/max/mean) and pass/fail status
- Useful after any detection head changes

---

## Training & Inference Guide

### Training Examples

**Recommended: Phase 6.5 Configuration (Best Checkpoint - Current Standard)**

Train detection with Phase 6.5 configuration (balanced confidence calibration, +14.5% ORIG mode AP):
```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --save-dir ./checkpoints_phase6 \
  --nproc 4 --device 0,1,2,3 \
  --batch-size 4 --max-epoch 100 -- \
  --task detection \
  --keypoint-mode --fixed-box-size 16 \
  --use-fpn --use-deconv --head-conv 256 \
  --det-sigma 0.8 --focal-alpha 0.75 --focal-gamma 2.5 \
  --det-pos-weight 1.0 --det-neg-topk-ratio 0.1 \
  --eval-nms-radius 2.0 --head-lr 0.002 \
  --freeze-backbone --freeze-unet --freeze-counter \
  --resume checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth
```
**Note:** Heatmap bias is initialized to -2.0 (Phase 6.5) for better TP confidence distribution. Remove `--resume` for fresh training from scratch.

**Single-GPU Training (Memory-Constrained):**
```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --save-dir ./checkpoints_phase6 \
  --nproc 1 --device 0 \
  --batch-size 1 --max-epoch 100 -- \
  --task detection --keypoint-mode \
  --use-fpn --use-deconv --head-conv 256 \
  --det-sigma 0.8 --focal-alpha 0.75 --focal-gamma 2.5 \
  --freeze-backbone --freeze-unet --freeze-counter
```
**Note:** Batch size 1 is slower but works on single GPU. Consider reducing `--max-epoch` if training time is limited.

**RGBT-CC Training (with Multi-Scale Augmentation):**
```bash
tools/train_entry.sh \
  --data-dir .data/RGBT-CC_converted \
  --save-dir ./checkpoints_phase6_rgbt_cc \
  --nproc 4 --device 0,1,2,3 \
  --batch-size 4 --max-epoch 100 -- \
  --task detection --keypoint-mode \
  --use-fpn --use-deconv --head-conv 256 \
  --det-sigma 0.8 --focal-alpha 0.75 --focal-gamma 2.5 \
  --aug-scale-min 0.5 --aug-scale-max 2.0 \
  --aug-flip 1 --thermal-clahe 1 \
  --det-pos-weight 1.0 --det-neg-topk-ratio 0.1 \
  --freeze-backbone --freeze-unet --freeze-counter
```
**Note:** RGBT-CC has extreme scale variation (5×5 to 100×100 px); augmentation is essential. CLAHE thermal preprocessing improves contrast.

**Counting-Only Training (Baseline):**
```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --save-dir ./checkpoints_counting \
  --nproc 4 --device 0,1,2,3 \
  --batch-size 4 --max-epoch 100 -- \
  --task counting
```
**Note:** Remove `--task detection` flags; only trains density estimation branch.

### Inference & Evaluation

**Visualize Detection Outputs (Phase 6.5 Checkpoint):**
```bash
python3 Fine-tune/test_detection_vis.py \
  --data-dir .data/DroneRGBT_converted \
  --ckpt checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth \
  --out ./visuals_phase6 \
  --num 1806 \
  --num-vis 64 \
  --batch-size 8 \
  --num-workers 4 \
  --keypoint-mode --fixed-box-size 16 \
  --use-fpn --use-deconv --head-conv 256
```
**Parameters:**
- `--num`: Total images to evaluate (1806 for full DroneRGBT test set)
- `--num-vis`: Subset to visualize with overlays (64 for detailed analysis; saves 95%+ preprocessing time)
- `--batch-size`: Larger values faster (8 default); reduce if OOM
- `--keypoint-mode`: Match training config exactly

**Comprehensive Diagnostics (RAW/TILES/ORIG Modes with Grid Search):**
```bash
bash tools/run_posttrain_diagnostics.sh \
  checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth \
  .data/DroneRGBT_converted
```
OR for custom output directory:
```bash
bash tools/run_posttrain_grid.sh \
  checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth \
  .data/DroneRGBT_converted \
  .tmp_posttrain_phase6_results \
  1806 4
```
**Output includes:**
- Three inference modes (RAW: no NMS, TILES: tiled inference, ORIG: full-image)
- Per-mode parameter sweeps (score thresholds, NMS radii, tile sizes)
- Detection overlays (RGB+Thermal with bounding boxes)
- Per-prediction CSVs (image_id, score, is_TP, gt_distance)
- TP/FP histograms for threshold tuning
- Aggregate metrics (AP, F1, precision, recall)

**Verify Counting Stability (Ensure Detection Training Preserves Counting):**
```bash
python3 Fine-tune/test_game.py \
  --data-dir .data/DroneRGBT_converted \
  --ckpt checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth
```
**Expected:** GAME/MAE metrics match Phase 6.5 baseline (counting head frozen, should not degrade).

📖 See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for complete phase timeline and [ARCHITECTURE.md](docs/ARCHITECTURE.md) for comprehensive system design and architecture diagrams.

## Datasets

This project uses two RGBT (RGB + Thermal) datasets for training and evaluation:

### DroneRGBT Dataset
- **Characteristics**: Relatively consistent scale (drone altitude stable)
- **Best Performance**: F1 = 0.608 with current Phase 6.5 configuration
- **Recommended for**: Initial experiments and validation
- **Preparation**: Use `tools/convert_dronergbt.py` (see [Getting Started](#getting-started) step 4)

### RGBT-CC Dataset (CVPR 2021)
- **Characteristics**: Extreme scale variation (5×5 to 100×100 pixels), high object density
- **Challenge**: Lower thermal quality (requires CLAHE preprocessing via `--thermal-clahe 1`)
- **Preparation**: Use `tools/convert_rgbtcc.py` + thermal stats computation (see [Getting Started](#getting-started))
- **Augmentation**: Requires aggressive augmentation (`--aug-scale-min 0.5 --aug-scale-max 2.0`) for generalization

#### DroneRGBT Results (Phase 6.5)

| Mode | Precision | Recall | F1 | AP | Notes |
|---:|---:|---:|---:|---:|---|
| RAW | 0.7542 | 0.5627 | 0.6472 | 0.5622 | No NMS |
| TILES | 0.7387 | 0.5333 | 0.6222 | 0.5281 | Tiled inference, NMS r=2 |
| ORIG | 0.6891 | 0.4801 | 0.5667 | 0.4502 | Full-image, NMS r=2 |

**AP@15px** (8px tolerance): RAW 0.7148, TILES 0.6669, ORIG 0.5590

#### RGBT-CC Results (Phase 6.5)

| Mode | Precision | Recall | F1 | AP | Notes |
|---:|---:|---:|---:|---:|---|
| RAW | 0.7152 | 0.4659 | 0.5642 | 0.4407 | No NMS |
| TILES | 0.9129 | 0.3662 | 0.5227 | 0.3557 | Tiled inference, NMS r=4 |
| ORIG | 0.9511 | 0.2842 | 0.4376 | 0.2792 | Full-image, NMS r=4 |

**Note**: High precision on ORIG/TILES indicates conservative score thresholding; RAW mode shows better recall but lower precision.

---

## Baseline Model Comparison (AP@15px)

To provide fair comparison with external baseline models (Faster RCNN, RetinaNet, YOLOv26s), we evaluate Phase 6.5 at AP@15px (more lenient distance threshold matching standard detection evaluation protocols). For baseline comparison, detections are evaluated as 15px x 15px bounding boxes, and IoU >= 50% is counted as a hit.

### Comparison Chart

![Baseline Comparison Chart](image/compare/baseline_comparison_chart.png)

### Quantitative Results

| Model | Modality | AP@15px | Precision | Recall | F1 | Notes |
|-------|----------|---------|-----------|--------|-----|-------|
| **Phase 6.5 (Ours)** | **RGBT** | **0.7148** | **0.6481** | **0.8035** | **0.7175** | Multi-modal fusion |
| Faster RCNN | RGB | 0.0156 | 0.0897 | 0.2969 | 0.1378 | Detectron2 baseline |
| RetinaNet | RGB | 0.0133 | 0.0899 | 0.2974 | 0.1380 | Detectron2 baseline |
| YOLOv26s | RGB | 0.0012 | 0.0579 | 0.5744 | 0.1056 | High recall, low precision |
| Faster RCNN | Thermal | 0.0275 | 0.1382 | 0.4573 | 0.2123 | Better than RGB |
| RetinaNet | Thermal | 0.0365 | 0.1558 | 0.5155 | 0.2393 | Best single-modal baseline |
| YOLOv26s | Thermal | 0.0034 | 0.0853 | 0.8466 | 0.1551 | Highest recall, lowest precision |

### Key Findings

1. **Multi-modal Superiority**: Phase 6.5 (RGBT fusion) achieves 19.6× higher AP than the best single-modal baseline (RetinaNet Thermal)
2. **Balanced Performance**: Our approach maintains both high precision (0.65) and recall (0.80), while baselines struggle with precision (<0.16)
3. **Thermal > RGB**: All baseline models perform better on thermal imagery than RGB for aerial crowd detection
4. **YOLO Trade-off**: YOLOv26s achieves highest recall (0.85 thermal) but suffers from extremely low precision (0.085), resulting in poor F1

### Visual Comparisons

Below are example detections from baseline models on test images (AP@15px RAW mode evaluation). For Detectron2 models, green boxes are ground truths and other colors are predictions. For YOLOv26s, blue boxes are predictions with no ground truths shown. For our method, blue crosses are ground truths, red circles are false positives, and green circles are true positives.

#### Faster RCNN (Detectron2)
| RGB Inference | Thermal Inference |
|---------------|-------------------|
| ![Faster RCNN RGB - Image 6](image/compare/detectron2-faster-rcnn-fpn-3x/6_comparison.jpg) | ![Faster RCNN Thermal - Image 6](image/compare/detectron2-faster-rcnn-fpn-3x/6R_comparison.jpg) |
| ![Faster RCNN RGB - Image 30](image/compare/detectron2-faster-rcnn-fpn-3x/30_comparison.jpg) | ![Faster RCNN Thermal - Image 30](image/compare/detectron2-faster-rcnn-fpn-3x/30R_comparison.jpg) |
| ![Faster RCNN RGB - Image 117](image/compare/detectron2-faster-rcnn-fpn-3x/117_comparison.jpg) | ![Faster RCNN Thermal - Image 117](image/compare/detectron2-faster-rcnn-fpn-3x/117R_comparison.jpg) |
| ![Faster RCNN RGB - Image 1206](image/compare/detectron2-faster-rcnn-fpn-3x/1206_comparison.jpg) | ![Faster RCNN Thermal - Image 1206](image/compare/detectron2-faster-rcnn-fpn-3x/1206R_comparison.jpg) |

*Note: Faster RCNN shows conservative predictions with low recall (~30% RGB, ~46% Thermal) but very low precision (~9-14%).*

#### RetinaNet (Detectron2)
| RGB Inference | Thermal Inference |
|---------------|-------------------|
| ![RetinaNet RGB - Image 6](image/compare/detectron2-retinanet-fpn-1x/6_comparison.jpg) | ![RetinaNet Thermal - Image 6](image/compare/detectron2-retinanet-fpn-1x/6R_comparison.jpg) |
| ![RetinaNet RGB - Image 30](image/compare/detectron2-retinanet-fpn-1x/30_comparison.jpg) | ![RetinaNet Thermal - Image 30](image/compare/detectron2-retinanet-fpn-1x/30R_comparison.jpg) |
| ![RetinaNet RGB - Image 117](image/compare/detectron2-retinanet-fpn-1x/117_comparison.jpg) | ![RetinaNet Thermal - Image 117](image/compare/detectron2-retinanet-fpn-1x/117R_comparison.jpg) |
| ![RetinaNet RGB - Image 1206](image/compare/detectron2-retinanet-fpn-1x/1206_comparison.jpg) | ![RetinaNet Thermal - Image 1206](image/compare/detectron2-retinanet-fpn-1x/1206R_comparison.jpg) |

*Note: RetinaNet achieves slightly better thermal performance (AP 0.0365) than Faster RCNN but still struggles with precision (~15-16%).*

#### YOLOv26s
| RGB Inference | Thermal Inference |
|---------------|-------------------|
| ![YOLOv2 RGB - Image 6](image/compare/yolov26s/6.jpg) | ![YOLOv2 Thermal - Image 6](image/compare/yolov26s/6R.jpg) |
| ![YOLOv2 RGB - Image 30](image/compare/yolov26s/30.jpg) | ![YOLOv2 Thermal - Image 30](image/compare/yolov26s/30R.jpg) |
| ![YOLOv2 RGB - Image 117](image/compare/yolov26s/117.jpg) | ![YOLOv2 Thermal - Image 117](image/compare/yolov26s/117R.jpg) |
| ![YOLOv2 RGB - Image 1206](image/compare/yolov26s/1206.jpg) | ![YOLOv2 Thermal - Image 1206](image/compare/yolov26s/1206R.jpg) |

*Note: YOLOv26s demonstrates the recall-precision trade-off: highest recall (85% thermal) but lowest precision (8.5%), resulting in massive false positive rates unsuitable for practical deployment.*

#### Phase 6.5 (Ours) - RGBT Fusion at AP@15px
| RGBT Joint Inference |
|----------------------|
| ![Phase 6.5 AP15 - Image 6](image/compare/phase6.5_AP15/6.jpg) |
| ![Phase 6.5 AP15 - Image 30](image/compare/phase6.5_AP15/30.jpg) |
| ![Phase 6.5 AP15 - Image 117](image/compare/phase6.5_AP15/117.jpg) |
| ![Phase 6.5 AP15 - Image 1206](image/compare/phase6.5_AP15/1206.jpg) |

*Note: Phase 6.5 processes RGB and Thermal images simultaneously through multi-modal fusion, achieving balanced precision (0.65) and recall (0.80) with AP 0.7148—significantly outperforming all single-modal baselines.*

#### Phase 6.5 AP@15px Inference Modes (RAW vs TILES vs ORIG)

| Mode | AP@15px | Precision | Recall | F1 | Notes |
|------|---------|-----------|--------|-----|-------|
| **RAW** | **0.7148** | **0.6481** | **0.8035** | **0.7175** | Full image, no tiling |
| **TILES** | **0.6669** | **0.7819** | **0.7382** | **0.7594** | Overlapping tiles + merge |
| **ORIG** | **0.5590** | **0.8587** | **0.6076** | **0.7117** | Legacy thresholds |

| Mode | Image 6 | Image 30 |
|------|---------|----------|
| RAW | ![Phase 6.5 AP15 RAW - Image 6](image/compare/phase6.5_AP15/6.jpg) | ![Phase 6.5 AP15 RAW - Image 30](image/compare/phase6.5_AP15/30.jpg) |
| TILES | ![Phase 6.5 AP15 TILES - Image 6](image/compare/phase6.5_AP15/tiles/6.jpg) | ![Phase 6.5 AP15 TILES - Image 30](image/compare/phase6.5_AP15/tiles/30.jpg) |
| ORIG | ![Phase 6.5 AP15 ORIG - Image 6](image/compare/phase6.5_AP15/orig/6.jpg) | ![Phase 6.5 AP15 ORIG - Image 30](image/compare/phase6.5_AP15/orig/30.jpg) |

| Mode | Image 117 | Image 1206 |
|------|-----------|------------|
| RAW | ![Phase 6.5 AP15 RAW - Image 117](image/compare/phase6.5_AP15/117.jpg) | ![Phase 6.5 AP15 RAW - Image 1206](image/compare/phase6.5_AP15/1206.jpg) |
| TILES | ![Phase 6.5 AP15 TILES - Image 117](image/compare/phase6.5_AP15/tiles/117.jpg) | ![Phase 6.5 AP15 TILES - Image 1206](image/compare/phase6.5_AP15/tiles/1206.jpg) |
| ORIG | ![Phase 6.5 AP15 ORIG - Image 117](image/compare/phase6.5_AP15/orig/117.jpg) | ![Phase 6.5 AP15 ORIG - Image 1206](image/compare/phase6.5_AP15/orig/1206.jpg) |

### Threshold Impact Analysis (Phase 6.5)

Our model's performance at different distance thresholds:

| Threshold | TP | FP | FN | Precision | Recall | AP | Use Case |
|-----------|----|----|----|-----------|---------|----|----------|
| **8px (Strict)** | 38,038 | 29,394 | 16,353 | 0.5641 | 0.6993 | 0.5622 | Internal evaluation, high precision requirements |
| **15px (Lenient)** | 43,705 | 23,727 | 10,686 | 0.6481 | 0.8035 | 0.7148 | Fair baseline comparison, standard detection protocols |
| **Delta** | +5,667 | -5,667 | -5,667 | +0.0840 | +0.1042 | +0.1526 | 27% AP improvement with relaxed matching |

The 15px threshold better reflects practical deployment scenarios where exact pixel-perfect localization is less critical than robust detection, while the 8px threshold provides stringent internal evaluation for model development.

---

## Troubleshooting

### Common Issues

**1. Checkpoint loading errors ("missing/unexpected keys")**
- DDP checkpoints have `module.` prefix; inference strips it automatically
- Use `strict=False` loading for partial weight initialization
- Verify `--keypoint-mode` matches training config

**2. Low precision / high false positives**
- Try `--eval-nms-radius 2.0` (tighter suppression)
- Increase `--det-neg-topk-ratio 0.1` (harder negative mining)
- Enable `--boundary-suppress` and `--adaptive-threshold`
- Check `scores.csv` histograms to tune score threshold

**3. Training instability / NaN loss**
- **See `Training Stability & Reproducibility`** for stability features (gradient clipping, NaN/Inf detection, background suppression).
- Reduce `--head-lr` if gradients explode
- Enable `--det-use-gn` for small-batch training

**4. Out of memory (OOM)**
- Reduce `--batch-size` (try 1 per GPU)
- Keep backbone frozen (`--freeze-backbone`)
- Disable FPN (`--use-fpn 0`) for memory-constrained setups

**5. Results differ from expected**
- Verify `--downsample-ratio 4` (not 8, this is critical!)
- Check all hyperparameters match teammate's config
- Ensure NMS radius matches evaluation settings
- Review gradient clipping and NaN detection are active

## Documentation & Development

- **Implementation notes**: `.plan/extension_progress.md` (recent updates and results)
- **Design rationale**: `.plan/extension_plan.md` (architecture decisions and future work)
- **Reproducibility**: All experiments documented with exact commands and metrics
- **Codebase structure**:
  - `Fine-tune/models/detection/`: CenterHead, FPN, detection wrapper
  - `Fine-tune/datasets/`: Dataset loaders and augmentation
  - `Fine-tune/utils/`: Trainer, evaluation, detection metrics
  - `tools/`: Training launchers, diagnostics, data conversion

## Citation & Acknowledgments

This fork extends the Free-Lunch multimodal counting framework with keypoint detection capabilities for aerial RGBT imagery. Critical stability fixes and hyperparameter tuning contributed by project teammates.

Original Free-Lunch repository: [link if available]

Final year project (FYP) at [institution], 2025.

## Future Work

- [ ] Validate RGBT-CC augmentation and thermal preprocessing
- [ ] Explore backbone unfreezing for higher AP
- [ ] Multi-dataset training and cross-domain evaluation
- [ ] Real-time inference optimization
- [ ] Bounding box prediction (if future datasets provide box annotations)

For feature requests, open an issue with detailed requirements and use case description.