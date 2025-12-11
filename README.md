# Free-Lunch-Multimodal-Counting (FYP fork)

This repository is a final-year-project (FYP) fork of the Free-Lunch multi-modal crowd counting codebase. It builds on the original implementation and adds a **CenterNet-style keypoint detection branch** with FPN multi-scale architecture for aerial RGBT imagery.

## Recent Updates (December 2025)

**✅ Teammate's Code Integration (2025-12-11)** - Critical stability fixes validated:
- **Gradient clipping** (max_norm=0.5) for stable training
- **NaN/Inf detection** with automatic batch skipping
- **Downsample ratio fix**: Corrected from 8→4 (2× larger heatmaps, major performance impact)
- **Background suppression tracking** for better monitoring
- **7 hyperparameter alignments**: det_sigma=0.8, focal_alpha=0.75, det_neg_topk_ratio=0.1, eval_nms_radius=2.0, det_patience=10, max_epoch=100, val_epoch=2
- **Results**: Precision 0.701, Recall 0.536, F1 0.608 (50.2% FP reduction vs baseline)

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

- Diagnostics & visualization improvements:
  - `Fine-tune/test_detection_vis.py` now supports a reproducible `--indices-file` to force identical image selections across multiple runs, writes `selected_indices.txt` when performing random selection, and deduplicates by dataset-provided image id to avoid repeated visualizations.
  - `tools/run_posttrain_diagnostics.sh` (post-train diagnostics wrapper) has been updated to reuse the `raw/selected_indices.txt` file so the `raw`, `tiles`, and `orig` visualizations process the same images for fair comparison.
  - The visualization tool produces per-prediction CSV (`scores.csv`) and TP/FP histograms (`scores.png`) for easy score-threshold analysis.

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
  --keypoint-mode \
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
  --aug-flip 0.5 --thermal-clahe 1 \
  --freeze-backbone --freeze-unet --freeze-counter \
  --resume .weights/drone_rgbt_best_494_781.pth
```

### Inference & Visualization

**Visualize detection outputs from checkpoint:**
```bash
python3 Fine-tune/test_detection_vis.py \
  --data-dir .data/DroneRGBT_counting \
  --ckpt checkpoints/1211-115847/best_model.pth \
  --out ./visuals_detection \
  --num 64 \
  --keypoint-mode \
  --use-fpn --use-deconv --head-conv 256
```

**Run comprehensive post-training diagnostics:**
```bash
tools/run_posttrain_diagnostics.sh \
  --data-dir .data/DroneRGBT_counting \
  --ckpt checkpoints/1211-115847/best_model.pth \
  --out .tmp_posttrain/1211-115847 \
  --num 64
```
This produces:
- `raw/`, `tiles/`, `orig/`: Detection overlays in three inference modes
- `raw/report.txt`: Per-sample TP/FP/FN metrics and totals
- `raw/scores.csv`: Per-prediction scores for threshold tuning
- `raw/scores.png`: TP/FP score histograms


## Key Features & Configuration

### Architecture Options
- **Keypoint-only mode** (`--keypoint-mode`): Removes size head for point-annotation datasets
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
- **Horizontal flip** (`--aug-flip 0.5`): Synchronized RGB+Thermal flipping
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

**DroneRGBT Dataset (40 test images, ~1982 targets):**

| Configuration | Precision | Recall | F1 | TP | FP | FN | Notes |
|--------------|-----------|--------|-----|-----|-----|-----|-------|
| Phase 1 (Keypoint baseline) | 0.12 | 0.53 | 0.20 | 1047 | 7453 | 935 | Single-scale, many FPs |
| Phase 2 (FPN multi-scale) | 0.56 | 0.58 | 0.57 | 1142 | 910 | 840 | 87.8% FP reduction |
| **Phase 3 (Teammate config)** | **0.70** | **0.54** | **0.61** | **1063** | **453** | **919** | ✅ **Production-ready** |

**Key improvements in Phase 3:**
- 50.2% FP reduction vs Phase 2 (910 → 453)
- 26.1% precision improvement (0.556 → 0.701)
- Sharper localization (det_sigma=0.8) and better class balance (focal_alpha=0.75)
- Stable training with gradient clipping and NaN detection

**Recommended configuration:** `det_sigma=0.8`, `focal_alpha=0.75`, `det_neg_topk_ratio=0.1`, `eval_nms_radius=2.0`, `head_lr=0.002`, gradient clipping enabled.

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
- `--aug-flip`: Horizontal flip probability (default: 0.0)
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
- Gradient clipping is automatic (max_norm=0.5)
- NaN/Inf detection skips bad batches automatically
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
