# Free-Lunch-Multimodal-Counting (FYP fork)

This repository is a final-year-project (FYP) fork of the Free-Lunch multi-modal crowd counting codebase. It builds on the original implementation and adds a **CenterNet-style keypoint detection branch** with FPN multi-scale architecture for aerial RGBT imagery.

## Recent Updates (December 2025)

**✅ Inference Acceleration & Grid Search (2025-12-12)** - Performance-optimized diagnostics:
- **Batch processing** (default batch_size=8): ~8× speedup via vectorized inference
- **Parallel data loading** (default num_workers=4): Overlapped I/O and GPU computation
- **Selective visualization** (--num vs --num-vis): Process full test set for AP/F1, visualize subset for analysis (95%+ preprocessing time saved)
- **Raw mode NMS control** (--no-nms flag): Disable all NMS in RAW mode for threshold analysis
- **Three inference modes** (RAW/TILES/ORIG) with automated grid search: Configure parameter sweeps per mode, automatic directory structure
- **Metrics output**: AP (8px distance threshold), F1, per-prediction CSV, TP/FP histograms

**✅ Teammate's Code Integration (2025-12-11)** - Teammate modifications (summary):

- **Boundary Suppression (center_head.py):**
  - Added `_apply_boundary_suppression()` to reduce spurious edge/corner detections
  - Reduced per-pixel confidence near image edges and corners
  - Convolutional boundary filtering to suppress border noise

- **Enhanced Loss Functions (dm_regression_trainer.py):**
  - Added background-suppression loss term
  - Improved negative mining to focus on hard false positives
  - Enforced gradient clipping for stable training

- **Adaptive Thresholding & Density Filtering (detection_eval.py):**
  - Dynamic score threshold per-image based on image statistics
  - Density-based filtering to reduce clustered false positives
  - Better handling of background regions via adaptive rules

- **Evaluation & Inference Improvements (dm_regression_trainer.py):**
  - Filter boundary detections during inference
  - Count-aware filtering to limit excessive detections in dense regions
  - Adaptive score thresholding to improve precision/recall balance

- **Notes:** These teammate changes target FP reduction and inference robustness (boundary filtering, adaptive thresholds, and loss improvements). See Phase 4 results below for empirical impact.

**⚠️ RGBT-CC Dataset Adaptation (2025-12-10)** - Implementation complete, testing pending:
- Multi-scale augmentation (0.5×–2.0× resize, flip, crop) for scale variation handling
- Thermal preprocessing with CLAHE for contrast enhancement
- Dataset-specific normalization stats computation tool

**✅ FPN Multi-Scale Detection (2025-12-09)**:
- SimpleFPN architecture (stride-4 output with multi-level feature fusion)
- 87.8% FP reduction vs single-scale baseline
- Precision jump from 12% → 56% (4.5× improvement)

**✅ Keypoint-Only Mode (2025-12-09)**:
- Removed size head for point-annotation datasets (DroneRGBT, RGBT-CC)
- 12% parameter reduction, cleaner architecture for sparse targets

**✅ CenterNet-Style Head (2025-12-05)**:
- Deconv upsampling (stride-8 → stride-4) for better spatial resolution
- Proper heatmap bias initialization (-2.19 for focal loss)
- Wider head channels (256 vs 128) and max-pooling NMS

## Core Differences & Major Changelog

**Detection Architecture:**
- Added CenterNet-style keypoint detection head with FPN multi-scale support
- Dataset targets: Gaussian heatmap + offset (keypoint-only mode for point annotations)
- CenterHead with deconv upsampling, proper bias initialization, and configurable capacity
- Integrated detection losses (focal/BCE heatmap, L1 offset) and AP-based evaluation

**Training Stability & Reproducibility:**
- ✅ **Gradient clipping** (max_norm=0.5) - prevents gradient explosion
- ✅ **NaN/Inf detection** - skips corrupted batches automatically
- ✅ **Background suppression** - reduces false positives near image boundaries
- Improved checkpoint loading with permissive prefix remapping and `strict=False`
- Disambiguated freeze flags: `--freeze-backbone`, `--freeze-unet`, `--freeze-counter`

**Visualization & Diagnostics:**
- `Fine-tune/test_detection_vis.py`: RGB+Thermal detection overlays with reproducible image selection
- `tools/run_posttrain_diagnostics.sh`: Comprehensive diagnostics (raw/tiled/orig modes)
- Per-prediction scoring CSVs and TP/FP histograms for threshold analysis
- DDP checkpoint prefix handling for seamless inference

**Multi-GPU & Infrastructure:**
- `tools/train_entry.sh`: Safe torchrun wrapper with sensible defaults
- DDP with `find_unused_parameters` handling for frozen components
- Separate param-groups for discriminative learning rates (head vs backbone)

Where to find details
- Implementation notes, tests, quick-run logs and discussion are collected under the `.plan/` directory. Start with `.plan/extension_progress.md` for a short status and next actions.

## Quick Start

We strongly recommend using the convenience launcher `tools/train_entry.sh` for training. It wraps `torchrun` safely, sets sensible defaults, and forwards extra flags to `Fine-tune/train.py`.

### Training Examples

**Recommended: Train with teammate's validated configuration (best F1 0.608 on DroneRGBT):**
```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_counting \
  --save-dir ./checkpoints \
  --nproc 4 --device 0,1,2,3 \
  --batch-size 4 --max-epoch 100 -- \
  --task detection \
  --keypoint-mode --fixed-box-size 16 \
  --use-fpn --use-deconv --head-conv 256 \
  --det-sigma 0.8 --focal-alpha 0.75 --det-neg-topk-ratio 0.1 \
  --eval-nms-radius 2.0 --head-lr 0.002 \
  --freeze-backbone --freeze-unet --freeze-counter \
  --resume .weights/drone_rgbt_best_494_781.pth
```

**Train detection head (single-GPU, simpler config):**
```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_counting \
  --save-dir ./checkpoints \
  --nproc 1 --device 0 \
  --batch-size 1 --max-epoch 50 -- \
  --task detection --keypoint-mode \
  --freeze-backbone --resume .weights/drone_rgbt_best_494_781.pth
```

**Train on RGBT-CC with augmentation (experimental):**
```bash
tools/train_entry.sh \
  --data-dir .data/RGBT-CC_converted \
  --save-dir ./checkpoints_rgbt_cc \
  --nproc 4 --device 0,1,2,3 \
  --batch-size 4 --max-epoch 100 -- \
  --task detection --keypoint-mode \
  --aug-scale-min 0.5 --aug-scale-max 2.0 \
  --aug-flip 1 --thermal-clahe 1 \
  --freeze-backbone --freeze-unet --freeze-counter \
  --resume .weights/drone_rgbt_best_494_781.pth
```

### Inference & Visualization

**Visualize detection outputs from checkpoint (optimized):**
```bash
python3 Fine-tune/test_detection_vis.py \
  --data-dir .data/DroneRGBT_counting \
  --ckpt checkpoints/1211-115847/best_model.pth \
  --out ./visuals_detection \
  --num 1806 \
  --num-vis 64 \
  --batch-size 8 \
  --num-workers 4 \
  --keypoint-mode --fixed-box-size 16 \
  --use-fpn --use-deconv --head-conv 256
```

**Run comprehensive post-training diagnostics with grid search:**
```bash
bash tools/run_posttrain_grid.sh \
  checkpoints/1211-115847/best_model.pth \
  .data/DroneRGBT_counting \
  .tmp_posttrain/1211-115847 \
  1806 4
```

Outputs: Three diagnostic modes (RAW/TILES/ORIG) with detection overlays, per-prediction CSVs, TP/FP histograms, and aggregate metrics. See "Inference Modes" section below for detailed mode descriptions.

#### Inference Performance Enhancements (December 2025)

The inference script has been optimized for speed and flexibility:

**1. Batch Processing**
- Default batch size: 8 images/batch (configurable via `--batch-size`)
- ~8× faster than sequential processing
- Efficient GPU memory utilization with dynamic batching

**2. Parallel Data Loading**
- Default workers: 4 (configurable via `--num-workers`)
- Overlapped I/O and GPU computation
- Custom collate function handles variable-length annotations

**3. Selective Visualization**
- `--num`: Total images to process for AP/F1 computation (e.g., 1806 for full test set)
- `--num-vis`: Subset of processed images to visualize and save overlays (e.g., 64 for detailed analysis)
- Only visualized images are preprocessing (image read, bbox visualization) — saves 95%+ preprocessing time
- Key insight: AP/F1 metrics computed on full `--num` set, visualization subset selected randomly for reproducibility

**4. NMS Control**
- RAW mode automatically disables all NMS (`--no-nms` flag) for raw prediction analysis
- ORIG and TILES modes enable NMS for realistic post-processing
- Per-image AP/F1 computation works correctly regardless of NMS setting

#### AP and F1 Computation

**Average Precision (AP):**
- IoU threshold: 8.0 pixels (AP@8px, configured via `--ap-dist-thresh`)
- Computed by matching predictions to ground-truth detections within distance threshold
- Output: Cumulative TP/FP counts, precision-recall curve, average precision metric
- Saved to: `report.txt` (summary) and `scores.csv` (per-prediction details)

**F1 Score:**
- Computed as harmonic mean of precision and recall: F1 = 2×(Precision×Recall)/(Precision+Recall)
- Derived from aggregate TP, FP, FN counts across all processed images
- Used as primary metric for model selection and hyperparameter tuning

**Metrics Output:**
- `report.txt`: Aggregate statistics (Precision, Recall, F1, TP, FP, FN counts)
- `scores.csv`: Per-prediction details (image_id, detection_score, is_tp, matched_gt_distance)
- `scores.png`: Histogram of detection scores with TP/FP distribution for threshold analysis

#### Inference Modes: RAW vs TILES vs ORIG

| Mode | NMS | Purpose | Use Case |
|------|-----|---------|----------|
| **RAW** | ❌ Disabled | Raw predictions without suppression, score thresholds swept (0.01–0.15) | Threshold tuning, raw score analysis, understanding model confidence |
| **TILES** | ✅ Enabled | Tiled inference with SAHI-style sliding window, multi-parameter optimization (tile_size, tile_overlap, nms_radius) | Large images, memory-constrained inference, parameter optimization |
| **ORIG** | ✅ Enabled | Full-image inference with standard post-processing (NMS radius 4, 150 max detections) | Production evaluation, final metrics, realistic deployment scenario |

**Parameter Sweeps per Mode:**
- **RAW**: score_thresh 0.01→0.15 (find optimal threshold without NMS)
- **TILES**: score_thresh 0.15→0.30, nms_radius 2/4/6, tile_size 512, tile_overlap 0.25 (optimize for tiled inference)
- **ORIG**: score_thresh 0.20→0.30, nms_radius 4, max_dets 150 (realistic full-image evaluation)


## Key Features & Configuration

### Architecture Options
- **Keypoint-only mode** (`--keypoint-mode`): Removes size head for point-annotation datasets
- **Fixed box size** (`--fixed-box-size 16`): Post-processing box size for keypoint detections
- **FPN multi-scale** (`--use-fpn`, `--fpn-levels 3`): Feature pyramid for handling scale variation
- **Deconv upsampling** (`--use-deconv`): Stride-8 → stride-4 for better spatial resolution
- **Head capacity** (`--head-conv 256`): Detection head channel width (default: 256)
- **Downsample ratio** (`--downsample-ratio 4`): Output stride (CRITICAL: must be 4, not 8)

### Loss Functions & Training
- **Focal loss** (`--use-focal-heatmap`, `--focal-alpha 0.75`, `--focal-gamma 1.5`): Addresses class imbalance in sparse heatmaps
- **Hard-negative mining** (`--det-neg-topk-ratio 0.1`): Top-K negative sampling to reduce FPs
- **Positive weighting** (`--det-pos-weight`): Upweight sparse positive pixels
- **Gaussian heatmap** (`--det-sigma 0.8`): Tighter kernel for sharper peaks and better localization
- **Gradient clipping** (automatic, max_norm=0.5): Prevents gradient explosion
- **NaN/Inf detection** (automatic): Skips corrupted batches
- **Background suppression** (`--bg-suppress-weight 0.01`): Reduces boundary false positives
- **Head learning rate** (`--head-lr 0.002`): Separate LR for detection head vs frozen backbone

### Evaluation & Post-Processing
- **NMS radius** (`--eval-nms-radius 2.0`): Tighter suppression for cleaner predictions
- **Soft-NMS** (`--eval-soft-nms-sigma`): Gaussian decay for overlapping detections
- **Max detections** (`--max-dets`): Top-K filtering by score
- **Adaptive threshold** (`--adaptive-threshold`): mean + 1.5×std dynamic thresholding
- **Boundary suppression** (`--boundary-suppress`, `--suppress-margin 4`): Edge/corner suppression
- **Count-aware filtering** (`--count-aware-filtering`): Spatial density filtering via KDTree
- **Tiled inference** (`--tile-size`, `--tile-overlap`): SAHI-style sliding window for large images

### Data Augmentation (RGBT-CC)
- **Multi-scale resize** (`--aug-scale-min 0.5`, `--aug-scale-max 2.0`): Random resize augmentation
- **Horizontal flip** (`--aug-flip 1`): Synchronized RGB+Thermal flipping
- **Random crop** (`--aug-crop 224`): Localized detection simulation
- **Thermal preprocessing** (`--thermal-clahe 1`): CLAHE contrast enhancement for thermal images

### Training Control
- **Freezing** (`--freeze-backbone`, `--freeze-unet`, `--freeze-counter`): Stage-wise training
- **Unfreeze epoch** (`--unfreeze-epoch N`): Automatic unfreezing schedule (-1 = never)
- **Early stopping** (`--det-patience 10`): Validation patience for detection AP
- **GroupNorm** (`--det-use-gn`): Small-batch stability for detection head
- **BCE logits mode** (`--use-bce-logits`): Avoid double-sigmoid mismatches

All detection features are opt-in via CLI flags. Counting-only experiments remain unchanged unless detection flags are enabled.

## Performance Metrics

**DroneRGBT Dataset (Full test set 1806 images, ~54391 targets):**

Grid search results from checkpoint 1130-145629_shallow_centerhead (November 30, 2025):

Train checkpoint (Phase 2): checkpoints/1205-155221_deeper_centerhead/best_model.pth
Train checkpoint (Phase 3): checkpoints/1209-205427_keypoint_mode_fpn/best_model.pth
Train checkpoint (Phase 4): checkpoints/1213-090950_teammate2/best_model.pth

| Phase | Configuration | Mode | Precision | Recall | F1 | TP | FP | FN | Notes |
|-------|--------------|------|-----------|--------|-----|-----|-----|-----|-------|
| **Phase 1** | Shallow CenterHead (no deconv, no FPN, no keypoint mode) | RAW st=0.15 | 0.0804 | 0.2921 | 0.1261 | 15885 | 181594 | 38506 | Baseline, no NMS |
| Phase 1 | Shallow CenterHead | ORIG st=0.20 | 0.0777 | 0.2852 | 0.1221 | 15514 | 184245 | 38877 | With NMS r=4 |
| Phase 1 | Shallow CenterHead | TILES st=0.25 ov=0.20 | 0.1686 | 0.1944 | 0.1806 | 10571 | 52110 | 43820 | Tiled inference |
| **Phase 2** | Deeper CenterHead (+ deconv, no keypoint mode) | RAW st=0.15 | 0.5607 | 0.5923 | 0.5761 | 32218 | 25244 | 22173 | Raw predictions (best at st=0.15); AP≈0.4773 |
| Phase 2 | Deeper CenterHead (+ deconv, no keypoint mode) | ORIG st=0.30 | 0.5693 | 0.5915 | 0.5802 | 32171 | 24339 | 22220 | Deconv added; no keypoint mode; AP≈0.4775 |
| Phase 2 | Deeper CenterHead (+ deconv, no keypoint mode) | TILES st=0.30 ov=0.15 | 0.5693 | 0.5915 | 0.5802 | 32171 | 24339 | 22220 | Tiled inference; AP≈0.4775 |
| **Phase 3** | Keypoint Mode + FPN | RAW st=0.10 | 0.5775 | 0.5362 | 0.5561 | 29163 | 21334 | 25228 | Keypoint-mode + FPN (raw best) AP≈0.4185 |
| Phase 3 | Keypoint Mode + FPN | ORIG st=0.30 | 0.5790 | 0.5360 | 0.5566 | 29152 | 21198 | 25239 | Keypoint-mode + FPN (orig best) AP≈0.4186 |
| Phase 3 | Keypoint Mode + FPN | TILES st=0.30 ov=0.15 | 0.5790 | 0.5360 | 0.5566 | 29152 | 21198 | 25239 | Keypoint-mode + FPN (tiles best) AP≈0.4186 |
| **Phase 4** | Teammate modifications (boundary/adaptive/density) | RAW st=0.15 | 0.4337 | 0.6581 | 0.5228 | 35794 | 46744 | 18597 | Teammate2 (retrain) — raw predictions; AP=0.4575 |
| Phase 4 | Teammate modifications (boundary/adaptive/density) | ORIG st=0.30 | 0.5505 | 0.5971 | 0.5728 | 32475 | 26522 | 21916 | Teammate2 (retrain) — orig inference; AP=0.4314 |
| Phase 4 | Teammate modifications (boundary/adaptive/density) | TILES st=0.30 ov=0.15 | 0.5505 | 0.5971 | 0.5728 | 32475 | 26522 | 21916 | Teammate2 (retrain) — tiled inference; AP=0.4314 |

**Phase 1 Analysis (1130-145629_shallow_centerhead):**
- Very low precision (0.08–0.17): Massive false positive rate across all modes
- Best mode: TILES with overlap=0.20 (F1=0.1806), but still poor absolute performance
- RAW vs ORIG similar F1: Suggests NMS not addressing fundamental detection issues
- Architecture limitations: No deconv upsampling, no FPN, standard detection head
- Critical missing features: Keypoint-mode architecture, deconv stride-4 output, FPN multi-scale

**Evolution Summary (placeholder for future updates):**
- Phase 1→2: Expected improvements from deconv (stride-4 resolution) and keypoint-mode architecture
- Phase 2→3: Expected 87.8% FP reduction from FPN multi-scale feature fusion
- Phase 3→4: Expected precision boost from teammate's hyperparameters (det_sigma=0.8, focal_alpha=0.75, gradient clipping)

**Recommended configuration:** Phase 4 with `det_sigma=0.8`, `focal_alpha=0.75`, `det_neg_topk_ratio=0.1`, `eval_nms_radius=2.0`, `head_lr=0.002`, gradient clipping enabled.

## CLI Flags Reference

### Basic Training
- `--data-dir`: Training data directory
- `--save-dir`: Checkpoint save directory
- `--task`: `counting` | `detection` (select training mode)
- `--lr`: Initial learning rate (default: 1e-5)
- `--head-lr`: Detection head learning rate (default: 0.002)
- `--resume`: Checkpoint path to resume from
- `--batch-size`: Train batch size per GPU
- `--max-epoch`: Maximum training epochs (default: 100)
- `--val-epoch`: Validation frequency in epochs (default: 2)

### Model Architecture
- `--keypoint-mode`: Enable keypoint-only detection (no size head)
- `--fixed-box-size`: Post-processing box size for keypoints (default: 16)
- `--use-fpn`: Enable Feature Pyramid Network multi-scale detection
- `--fpn-levels`: Number of FPN levels (default: 3)
- `--use-deconv`: Enable deconv upsampling (stride-8 → stride-4)
- `--head-conv`: Detection head channel width (default: 256)
- `--downsample-ratio`: Output stride (CRITICAL: use 4, not 8)

### Loss Configuration
- `--use-focal-heatmap`: Enable focal loss for heatmap
- `--focal-alpha`: Focal loss alpha (default: 0.75)
- `--focal-gamma`: Focal loss gamma (default: 1.5)
- `--det-sigma`: Gaussian heatmap sigma (default: 0.8)
- `--det-neg-topk-ratio`: Top-K negative sampling ratio (default: 0.1)
- `--det-pos-weight`: Positive pixel weight (default: 1.0)
- `--bg-suppress-weight`: Background suppression weight (default: 0.01)

### Freezing & Stage Training
- `--freeze-backbone`: Freeze Swin transformer backbone
- `--freeze-unet`: Freeze U-Net feature fusion
- `--freeze-counter`: Freeze counting regression head
- `--unfreeze-epoch`: Epoch to unfreeze backbone (-1 = never)

**✅ Validation**: Counting weights are preserved during detection-only training (verified with `Fine-tune/test_game.py` on checkpoint 1211-115847). Use `Fine-tune/test_game.py` to verify counting metrics (GAME/MAE) remain stable across detection training runs.

### Evaluation & Metrics
- `--eval-nms-radius`: NMS radius for evaluation (default: 2.0)
- `--eval-soft-nms-sigma`: Soft-NMS sigma (0 = disabled)
- `--max-dets`: Maximum detections per image
- `--det-patience`: Early stopping patience for AP (default: 10)

### Data Augmentation (RGBT-CC)
- `--aug-scale-min`: Minimum resize scale (default: 1.0)
- `--aug-scale-max`: Maximum resize scale (default: 1.0)
- `--aug-flip`: Horizontal flip augmentation (default: 1)
- `--aug-crop`: Random crop size (0 = disabled)
- `--thermal-clahe`: Enable CLAHE thermal preprocessing

### Multi-GPU / DDP
- `--nproc`: Number of GPUs (set in `train_entry.sh`)
- `--device`: CUDA device IDs (e.g., `0,1,2,3`)
- `--local_rank`: Set by torchrun automatically

Note: `tools/train_entry.sh` accepts launcher options (`--nproc`, `--device`) before `--`, and forwards training flags after `--` to `Fine-tune/train.py`.

## Datasets

### DroneRGBT (Validated)
- Relatively consistent scale (drone altitude stable)
- **Best F1: 0.608** with teammate's configuration
- Recommended for initial experiments and validation

### RGBT-CC (Experimental)
- Extreme scale variation (5×5 to 100×100 pixels)
- Lower thermal quality (requires CLAHE preprocessing)
- Use augmentation flags for better generalization

#### RGBT-CC Training Results (Teammate)
RGBT-CC training results (full test set) from teammate's run:

| Configuration | Mode | Precision | Recall | F1 | TP | FP | FN | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| orig_st_0.20_r_4_k_150 | ORIG | 0.7215 | 0.3297 | 0.4526 | 19553 | 7546 | 39748 | 0.3083 |
| raw_st_0.15_k_1000 | RAW | 0.6623 | 0.3519 | 0.4596 | 20867 | 10640 | 38434 | 0.3094 |
| tiles_st_0.20_r_4_k_150_ts_512_ov_0.15 | TILES | 0.7215 | 0.3297 | 0.4526 | 19553 | 7546 | 39748 | 0.3083 |

Notes: RAW runs used `--no-nms`; ORIG/TILES used NMS radius 4 and `max_dets=150`. AP computed at 8px distance threshold.

### Dataset Preparation
1. Point annotations stored as `.npy` files (N×2 array of x,y coordinates)
2. RGB and thermal images in separate directories
3. Use `tools/convert_dronergbt.py` or `tools/convert_rgbtcc.py` for format conversion

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
