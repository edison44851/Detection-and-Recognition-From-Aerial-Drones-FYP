# Inference and Evaluation Guide

## Quick Start

### Visualize Predictions

```bash
python3 Fine-tune/test_detection_vis.py \
  --ckpt checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth \
  --data-dir .data/DroneRGBT_converted \
  --num 64 \
  --use-deconv 1 \
  --head-conv 256
```

This generates RGB+Thermal overlays with predicted bounding boxes for visual inspection.

### Comprehensive Diagnostics & Metrics

```bash
bash tools/run_posttrain_diagnostics.sh \
  checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth \
  .data/DroneRGBT_converted
```

This runs complete evaluation:
- Three inference modes (RAW, TILES, ORIG)
- Parameter sweeps (thresholds, NMS radii, tile sizes)
- Detailed metrics (AP, F1, precision, recall)
- Per-prediction CSVs and histograms

### Validate Counting Unaffected

```bash
python3 Fine-tune/test_game.py \
  --ckpt checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth \
  --data-dir .data/DroneRGBT_converted
```

Verifies counting metrics (GAME, MAE) unchanged while training detection

---

## Evaluation Modes Explained

### RAW Mode (Recommended)

**What it does:**
- Single-pass inference on full-resolution image
- Standard CenterNet evaluation
- No tiling or preprocessing tricks

**Command:**
```bash
python3 Fine-tune/test_detection_vis.py \
  --ckpt <checkpoint> \
  --data-dir <dataset> \
  --eval-mode raw
```

**Output Characteristics:**
- Predictions distributed naturally where objects exist
- Score range: 0.01-0.999
- Number of detections: varies by image (50-200 typical)

**When to Use:**
- Primary evaluation metric
- Deployment scenario
- Comparing models fairly

---

### TILES Mode (Scale-Robust)

**What it does:**
- Divide image into overlapping tiles (e.g., 256×256)
- Run inference on each tile separately
- Merge predictions with NMS across tiles

**Benefits:**
- Reduces stride-related quantization error
- Better for scale variation (different object sizes)
- Can improve AP on complex imagery

**Command:**
```bash
python3 Fine-tune/test_detection_vis.py \
  --ckpt <checkpoint> \
  --data-dir <dataset> \
  --eval-mode tiled \
  --tile-size 256 \
  --tile-overlap 0.5
```

**Parameters:**
- `tile-size`: Size of each tile (256 typical)
- `tile-overlap`: Overlap fraction (0.5 = 50% overlap)

**Trade-offs:**
- More stable AP across varied scales
- Slower inference (~2-3× slower than RAW)
- More detections (may increase FP if thresholds not adjusted)

---

### ORIG Mode (Legacy)

**What it does:**
- Historical evaluation mode with calibrated thresholds
- Applied carefully tuned score threshold (0.3) and NMS radius (4.0)
- Maintains backward compatibility with past results

**Command:**
```bash
python3 Fine-tune/test_detection_vis.py \
  --ckpt <checkpoint> \
  --data-dir <dataset> \
  --eval-mode orig
```

**Important Notes:**
- Thresholds hardcoded for Phase 6.3+ checkpoints
- Not recommended for new work (use RAW instead)
- Phase 6.5 achieves +14.5% in this mode (better bias calibration)

---

## Phase 6.5 AP@15px Mode Comparison (RAW vs TILES vs ORIG)

The images below use the AP@15px evaluation rule (15px x 15px boxes, IoU >= 50%) and show how inference mode changes output density and calibration.

| Mode | AP@15px | Precision | Recall | F1 | Notes |
|------|---------|-----------|--------|-----|-------|
| **RAW** | **0.7148** | **0.6481** | **0.8035** | **0.7175** | Full image, no tiling |
| **TILES** | **0.6669** | **0.7819** | **0.7382** | **0.7594** | Overlapping tiles + merge |
| **ORIG** | **0.5590** | **0.8587** | **0.6076** | **0.7117** | Legacy thresholds |

| Mode | Image 6 | Image 30 |
|------|---------|----------|
| RAW | ![Phase 6.5 AP15 RAW - Image 6](../image/compare/phase6.5_AP15/6.jpg) | ![Phase 6.5 AP15 RAW - Image 30](../image/compare/phase6.5_AP15/30.jpg) |
| TILES | ![Phase 6.5 AP15 TILES - Image 6](../image/compare/phase6.5_AP15/tiles/6.jpg) | ![Phase 6.5 AP15 TILES - Image 30](../image/compare/phase6.5_AP15/tiles/30.jpg) |
| ORIG | ![Phase 6.5 AP15 ORIG - Image 6](../image/compare/phase6.5_AP15/orig/6.jpg) | ![Phase 6.5 AP15 ORIG - Image 30](../image/compare/phase6.5_AP15/orig/30.jpg) |

| Mode | Image 117 | Image 1206 |
|------|-----------|------------|
| RAW | ![Phase 6.5 AP15 RAW - Image 117](../image/compare/phase6.5_AP15/117.jpg) | ![Phase 6.5 AP15 RAW - Image 1206](../image/compare/phase6.5_AP15/1206.jpg) |
| TILES | ![Phase 6.5 AP15 TILES - Image 117](../image/compare/phase6.5_AP15/tiles/117.jpg) | ![Phase 6.5 AP15 TILES - Image 1206](../image/compare/phase6.5_AP15/tiles/1206.jpg) |
| ORIG | ![Phase 6.5 AP15 ORIG - Image 117](../image/compare/phase6.5_AP15/orig/117.jpg) | ![Phase 6.5 AP15 ORIG - Image 1206](../image/compare/phase6.5_AP15/orig/1206.jpg) |

---

## Parameter Tuning

### Score Threshold

**Effect:** Higher threshold = fewer, more confident detections

**Typical Range:**
- `0.01`: Very permissive, high FP (~80% FP rate)
- `0.10`: Balanced (recommended starting point)
- `0.20`: Conservative, lower FP (~10% FP rate)
- `0.30`: Very conservative, minimal FP

**How to Tune:**
```bash
# Run grid search across thresholds
bash tools/run_posttrain_grid.sh \
  <checkpoint> \
  <dataset> \
  <num-images> \
  <num-workers>
```

Then examine output CSVs:
```bash
ls .tmp_posttrain/*/raw/TP_FP_histograms/
# Shows precision/recall curves for each threshold
```

**For Your Application:**
- **High precision needed** (fewer false alarms): threshold=0.20-0.30
- **Balanced** (default): threshold=0.10-0.15
- **Coverage important** (catch most targets): threshold=0.01-0.05

---

### NMS (Non-Maximum Suppression) Radius

**Effect:** Larger radius = fewer detections (more duplicate suppression)

**Typical Range:**
- `2.0`: Tight, removes many duplicates (recommended)
- `4.0`: Moderate (Phase 6.3 default)
- `6.0`: Loose, keeps more detections
- `0.0`: Disabled (all peaks kept)

**When to Adjust:**
- Many duplicate detections → increase radius (e.g., 4.0 → 6.0)
- Losing nearby objects → decrease radius (e.g., 4.0 → 2.0)

---

### Tile Parameters (TILES mode only)

**Tile Size:**
- `128`: Smaller, more detailed but slower
- `256`: Recommended, good balance
- `512`: Larger, faster but loses local fine-grained detail

**Tile Overlap:**
- `0.25`: Minimal overlap (25%)
- `0.5`: Recommended (50%)
- `0.75`: Maximum overlap, slower but more stable

---

## Understanding Metrics

### Average Precision (AP)

**Definition:** Area under precision-recall curve

**Scoring:** Detections matched to ground-truth using **spatial distance** (8px threshold)
- TP: Detection within 8 pixels of nearest GT point
- FP: No matching GT within 8 pixels
- FN: GT point without nearby detection

**Interpretation:**
- AP=0.50: 50% area under curve (good)
- AP=0.70: 70% area under curve (very good)
- AP>0.80: Excellent (rarely achieved on this dataset)

### Precision & Recall

**Precision:** Of detections made, what % are correct
- High precision = few false alarms
- Formula: TP / (TP + FP)

**Recall:** Of ground-truth objects, what % did we find
- High recall = catching most targets
- Formula: TP / (TP + FN)

**Trade-off:** Usually must sacrifice one for the other

### F1 Score

**Harmonic mean:** F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Interpretation:**
- F1=0.60: Decent balanced performance
- F1=0.70: Good balanced performance
- F1>0.80: Excellent

---

## Detailed Output Analysis

### Output Structure

```
.tmp_posttrain/<timestamp>_<config>/
├── raw/
│   ├── predictions_score_threshold_*.csv      # Per-prediction data
│   ├── TP_FP_histograms.png                   # Score distribution
│   ├── metrics_summary.json                   # AP, F1, precision, recall
│   └── detections_*.png                       # Visualization overlays
├── tiled/
│   └── (same structure as raw)
└── orig/
    └── (same structure as raw)
```

### CSV Format (predictions_*.csv)

Columns:
- `image_id`: Which image
- `score`: Detection confidence (0-1)
- `is_TP`: 1 if matched to GT, 0 if false positive
- `distance_to_gt`: Pixel distance to nearest GT point
- `location_x`, `location_y`: Detection coordinates
- `size_w`, `size_h`: Predicted bounding box dimensions

**Use case:** Custom threshold analysis
```bash
# Find optimal threshold by analyzing CSVs
python3 -c "
import pandas as pd
df = pd.read_csv('.tmp_posttrain/raw/predictions_*.csv')
thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
for t in thresholds:
    tp = (df[df['score'] > t]['is_TP']).sum()
    fp = (df[df['score'] > t]['is_TP'] == 0).sum()
    print(f'Threshold {t}: TP={tp}, FP={fp}')
"
```

### Histogram Analysis

**TP/FP histogram** shows:
- X-axis: Confidence scores (0-1)
- Y-axis: Count of detections
- Green bars: True positives
- Red bars: False positives

**Interpretation:**
- **Good model:** TPs skewed right (high confidence), FPs skewed left (low confidence)
- **Poor model:** Overlap between TP and FP distributions
- **Over-threshold:** All detections low confidence; raise score_threshold

---

## Advanced: Custom Evaluation

### Evaluate on Custom Metrics

```python
# Fine-tune/test_detection_vis.py - modify evaluation section
import torch
from Fine-tune.utils.detection_eval import DetectionEvaluator

evaluator = DetectionEvaluator(
    distance_threshold=8,  # Change TP/FP matching distance
    max_dets=300,         # Limit predictions
    nms_radius=2.0,       # NMS kernel size
    score_threshold=0.1   # Score cutoff
)

results = evaluator.evaluate(predictions, ground_truth)
print(f"Custom AP: {results['ap']:.4f}")
```

### Evaluate on Subsets

```bash
# Only evaluate on dense regions
python3 -c "
import cv2
import numpy as np

# Load image, detect dense regions
# Then run inference only on those regions
"
```

### Export for External Tools

```bash
# Convert predictions to COCO format
python3 -c "
import json
predictions = [
    {'image_id': i, 'category_id': 1, 'bbox': [x, y, w, h], 'score': s}
    for (i, x, y, w, h, s) in detections
]
with open('detections.json', 'w') as f:
    json.dump(predictions, f)
"

# Now use external tools (e.g., pycocotools) for evaluation
```

---

## Counting Head Validation

### GAME Score (Geometric Accuracy Metric)

Measures density estimation accuracy at multiple scales:

```bash
python3 Fine-tune/test_game.py \
  --ckpt <checkpoint> \
  --data-dir <dataset>
```

**Output:**
```
GAME[0]: 62.3  (full image)
GAME[1]: 45.2  (4 quadrants)
GAME[2]: 38.1  (16 sub-regions)
MAE: 15.3      (mean absolute error in count)
RMSE: 21.4     (root mean squared error)
```

**Interpretation:**
- Lower GAME/MAE = better counting
- Detection training should NOT degrade these (counting head frozen)
- If counting worsens significantly: reduce `det-loss-weight` or unfreeze counting losses

---

## Production Checklist

### Before Deployment

- [ ] Run diagnostics on target dataset
- [ ] Verify counting metrics unchanged
- [ ] Analyze TP/FP histogram, choose appropriate threshold
- [ ] Test on representative edge cases
- [ ] Document chosen threshold and NMS radius
- [ ] Store checkpoint path and config in config file

### Inference Configuration File

```yaml
# inference_config.yaml
model:
  checkpoint: checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth
  use_deconv: true
  use_fpn: false
  head_conv: 256
  
inference:
  eval_mode: raw              # or "tiled" for more robustness
  score_threshold: 0.15       # Tuned on your validation data
  nms_radius: 2.0
  max_detections: 300
  
performance:
  expected_ap: 0.56
  expected_precision: 0.70
  expected_recall: 0.54
```

### Health Checks

Before running on new images:

```python
# Sanity checks
assert model is not None
assert checkpoint_loaded_successfully
assert images.shape[-3:] == (3, 800, 800)  # RGB
assert thermal.shape[-3:] == (3, 800, 800)  # Thermal (3-channel format)
assert 0 <= score_threshold <= 1
assert nms_radius > 0
```

---

## Troubleshooting

### Issue: Score Distribution Compressed (All scores 0.05-0.45)

**Cause:** Boundary suppression or similar features enabled

**Solution:** Verify in [KNOWN_ISSUES.md](KNOWN_ISSUES.md), ensure features are disabled

**Check:**
```bash
python3 tools/verify_score_fix.py
```

### Issue: AP Lower Than Expected

**Causes:**
- Wrong threshold (too high/low)
- Input format wrong (RGB channel order, scale, etc.)
- Different evaluation function (distance vs IoU)

**Solutions:**
1. Run grid search with `run_posttrain_grid.sh`
2. Visualize images with `test_detection_vis.py`
3. Compare metrics before/after preprocessing

### Issue: Many Detections, But Low AP

**Causes:**
- Threshold too low
- NMS radius too small

**Solutions:**
```bash
# Increase threshold
--score-threshold 0.20

# Increase NMS
--nms-radius 4.0
```

### Issue: Missing Detections (High FN)

**Causes:**
- Threshold too high
- Model didn't see those objects during training

**Solutions:**
```bash
# Lower threshold
--score-threshold 0.05

# Use TILES mode (more careful scale handling)
--eval-mode tiled
```

---

## Next Steps

1. Review [TRAINING.md](TRAINING.md) if you plan to fine-tune further
2. Check [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for common pitfalls
3. See [DEVELOPMENT.md](DEVELOPMENT.md) for what hyperparameters worked best
4. Read [ARCHITECTURE.md](ARCHITECTURE.md) if modifying the model
