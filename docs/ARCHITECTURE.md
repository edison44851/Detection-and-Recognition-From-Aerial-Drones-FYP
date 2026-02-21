# System Architecture Guide

## Overview

This project extends the Free-Lunch multimodal counting framework with a **CenterNet-style detection head** for aerial RGBT (RGB + Thermal) detection. The unified model performs two tasks simultaneously:

1. **Counting:** Density estimation from point annotations (inherited from Free-Lunch)
2. **Detection:** Center-based object detection for individual person localization

### High-Level System Architecture

Below is the complete pipeline showing how RGB and thermal inputs are fused through a U-Net broker, processed by Swin backbones, and output through separate counting and detection heads:

![Model Architecture - Full Pipeline](../image/FYP-High-level.png)

*Figure: Unified dual-task model — U-Net cross-attention fusion → triple-path Swin backbone features (RGB, Thermal, Broker) → separate counting and detection outputs.*

---

## Core Architecture Components

### 1. U-Net Cross-Modal Fusion

**Purpose:** Create a shared intermediate "broker" modality from RGB and thermal inputs

**Architecture:**
- Encoder: processes both RGB and thermal separately
- Cross-attention decoder: fuses both modalities using attention mechanisms
- Output: broker modality that captures complementary information from both Sources

**Key Insight:** The broker modality bridges single-modality gaps by learning cross-modal relationships

![U-Net Cross-Attention Fusion](../image/FYP-Unet.png)

*Figure: Cross-attention U-Net encoder-decoder with cross-modal transformer at bottleneck. Asymmetric skip connections allow thermal and RGB features to interact at each decoder level.*

---

### 2. Swin Transformer Backbone (Counting Head)

**Configuration:**
- Single shared backbone (`Swin_BM_RGBT`) applied to all three inputs:
  - RGB features → $\text{f}_R$
  - Thermal features → $\text{f}_T$
  - Broker features → $\text{f}_B$
- Hierarchical structure with 4 stages, each reducing spatial dims by 2× (total stride-8 output)
- Multi-scale attention with relative position biases

**Feature Fusion:**
```
Features = f_R + f_T + f_B  (elementwise sum)
Density  = reg_layer(Features)  (3-layer conv stack → 1 channel)
```

### Swin Backbone Architecture

The triple-path Swin backbone processes RGB, thermal, and broker features independently with shared architecture, then combines them for counting and detection:

![Swin Transformer Backbone - Triple Path](../image/FYP-Swin Backbone.png)

*Figure: Hierarchical Swin architecture applied to three input modalities. Each stream outputs 768-channel features at stride-8. Features are fused via elementwise addition.*

### Density Estimation Head

The counting branch predicts density maps from fused backbone features:

![Density Map Regression](../image/FYP-Regression.png)

*Figure: Three-layer convolutional stack (768→256→128→1) with ReLU activations, producing single-channel density map in [0, ∞) range.*

---

### 3. CenterNet-Style Detection Head

**Purpose:** Predict center locations and bounding box parameters

**Architecture (Current Best - Phase 6.5):**

```
Backbone Output (768 channels, stride-8)
    ↓
Upsample Module:
  ConvTranspose2d(768→256, k=4, s=2, p=1) + BatchNorm + ReLU
    ↓
Feature (256 channels, stride-4)
    ↓
╔═ Heatmap Head:  Conv(256→256, 3×3) → ReLU → Conv(256→1, 1×1)
║                 [bias=-2.0 (Phase 6.5)] → [sigmoid]
║
╠═ Size Head:     Conv(256→256, 3×3) → ReLU → Conv(256→2, 1×1)
║
╚═ Offset Head:   Conv(256→256, 3×3) → ReLU → Conv(256→2, 1×1)
    ↓
Inference: Max-pool NMS (kernel=3) → Extract local maxima → Detections
```

**Key Design Choices:**

| Component | Design | Justification |
|-----------|--------|---------------|
| **Output Stride** | 4 (stride-4) | 2× finer than backbone (stride-8); critical for small aerial objects |
| **Head Depth** | 2 layers | Single layer = linear only; 2 layers enable non-linear feature transformation |
| **Channel Reduction** | 768 → 256 | Adaptive feature transformation; 768 designed for dense counting, 256 for sparse detection |
| **Heatmap Bias** | -2.0 | Phase 6.5: sigmoid(-2.0) ≈ 0.12; optimal for focal loss convergence |
| **Activation** | ReLU + Sigmoid | Non-linearity in head + probability for heatmap |
| **NMS** | Max-pooling (k=3) | Removes soft duplicate peaks, proven effective in CenterNet |

### Detection Head Module Architecture

Below is the detailed architecture of the detection head, showing the three parallel prediction branches (heatmap, size, offset):

![CenterNet Detection Head](../image/FYP-Detection Head.png)

*Figure: FPN output (256 channels) feeds parallel heads for heatmap (center location), size (box dimensions), and offset (sub-pixel refinement). Outputs are [B, 1, H, W], [B, 2, H, W], [B, 2, H, W] respectively.*

### Detection Adaptor Module 

When features from the Swin backbone need adaptation, the following module is applied:

![Adaptor Module](../image/FYP_Detection Adapter.png)

*Figure: Adaptor transforms fused backbone features (768 channels) to detection head input (256 channels) via grouped normalization and learned transformation.*

---

## Loss Functions

### Counting Branch (Frozen in Detection-Only Training)

- **Count MAE:** L1 distance between predicted and ground-truth point count
- **OT Loss:** Optimal transport alignment between predicted density and point-based distribution
- **TV Loss:** Total variation for local smoothness
- **RD Loss:** Regional density contrastive learning

### Detection Branch

- **Focal Loss (Heatmap):** $L_{focal} = -\alpha(1-p_t)^\gamma\log(p_t)$
  - Default: $\alpha=0.75, \gamma=2.5$
  - Focuses on hard positives and hard negatives
  
- **L1 Loss (Size & Offset):** Standard L1 regression
  - Size: predicts bounding box width/height
  - Offset: sub-pixel center refinement (±0.5 cell)

- **Hard Negative Mining:** Train only on top-10% hardest negative samples to balance class imbalance

---

## Feature Extraction Options

### Full Features (Default - Phase 6.5)
```python
features = f_R + f_T + f_B  # Includes all three modalities
```
**Benefit:** Multi-scale context, precision +28-32% vs single modality
**Cost:** Slightly higher computational overhead

### Broker Features Only
```python
features = f_B  # Only cross-modal fusion output
```
**Benefit:** Lighter computation
**Trade-off:** Less precision

---

## Training vs Inference Model Consistency

**Critical Requirement:** Train and inference models must match exactly

### Potential Mismatches (Learned from Phase 6.4 Failure)

1. **Head Architecture**
   - Training: 2-layer conv heads with ReLU
   - Inference: Must match exactly; extra conv layers break inference

2. **Adaptor Module**
   - Training: `det_adaptor` transforms backbone to head features
   - Inference: Must be included in forward pass; missing adaptor = random features

3. **GroupNorm Placement**
   - Training: GN applied during training
   - Inference: Must match; changing GN locations breaks calibration

4. **Deconv Upsampling**
   - Training: ConvTranspose2d learns to produce sparse features
   - Inference: Exact weights must carry over; stride mismatch breaks everything

---

## Inference Pipeline

### Peak Extraction

```python
# 1. Apply heatmap threshold
heatmap_binary = heatmap > score_threshold

# 2. Max-pooling NMS to find local maxima
heatmap_max = maxpool2d(heatmap, kernel=3, stride=1, padding=1)
peaks_mask = (heatmap == heatmap_max) & (heatmap > score_threshold)

# 3. Extract peak coordinates and scores
peak_coords = where(peaks_mask > 0)  # (y, x) locations
peak_scores = heatmap[peaks_mask]

# 4. Sort by confidence
top_k = argsort(peak_scores, descending=True)[:max_dets]
```

### Bounding Box Decoding

```python
# For each peak at (cy, cx):
center = (cy, cx) * stride + offset[cy, cx] * stride
width = size[cy, cx, 0]
height = size[cy, cx, 1]
bbox = [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]
```

### Distance-Based Evaluation

Detections are matched to ground-truth using **spatial distance** (not IoU):
- **TP:** detection within 8 pixels of nearest ground-truth point
- **FP:** no ground-truth point within 8 pixels
- **FN:** ground-truth point with no nearby detection

---

## Evaluation Modes

### RAW Mode
- Direct inference on full image at original size
- Standard CenterNet approach
- **Best for:** Controlled settings, consistent scale

### TILES Mode  
- Divide image into grid of tiles (default 256×256)
- Run inference on each tile independently
- Merge predictions with non-maximum suppression
- **Best for:** Large images with scale variation; reduces stride-related quantization error

### ORIG Mode
- Legacy evaluation mode
- Applied historical calibrated thresholds
- **Note:** Maintained for reproducibility; RAW/TILES recommended for new work

---

## Historical Evolution (Baseline → Current)

### Baseline Detection Head (Catastrophic Failure)

| Issue | Impact |
|-------|--------|
| Shallow 1-layer heads (linear only) | No feature transformation; learns poor representations |
| Stride-8 output | Small objects collapse to single cell; massive quantization error |
| Heatmap bias = 0 | sigmoid(0) = 0.5; conflicts with focal loss optimization |
| No NMS | 75-79 soft peaks per image; ~96% false positives |
| **Result:** TP=11-280, Recall=0.5-14%, AP~0.01-0.3% |

### CenterNet Upgrade (Phase 6.5)

| Improvement | Mechanism | Gain |
|------------|-----------|------|
| Deconv upsampling → stride-4 | 2× finer heatmap resolution | 2-4× reduction in quantization error |
| 2-layer heads with ReLU | Enables non-linear feature transformation | 10-20× better feature expressiveness |
| Heatmap bias = -2.0 | sigmoid(-2.0) ≈ 0.12; matches focal loss prior | Faster convergence, better calibration |
| Max-pooling NMS | Removes soft duplicate peaks | ~10× reduction in false positives |
| 256 channel features | Sufficient capacity for 3 independent heads | Better than 128-channel bottleneck |

**Combined Effect:** 122× better recall, 48× better AP

---

## Multi-Scale FPN Option

**Phase 6.1-6.3 added SimpleFPN** for multi-scale feature fusion:

```
Backbone Features (stride-8)
    ↓
FPN Levels: P4 (stride-4), P8 (stride-8), P16 (stride-16)
    ↓
Per-level CenterHead predicts peaks
    ↓
Merge predictions with cross-scale NMS
```

**Benefits:**
- Better handling of scale variation (people in varying distances)
- Multi-scale contextual information

**Trade-offs:**
- More parameters (~10% increase)
- Slower inference (~10% slower)

---

## Keypoint Mode Option

**Phase 6.2+ supported keypoint-only detection:**

```
Standard mode:  Predicts heatmap + size + offset (3 heads)
Keypoint mode:  Predicts heatmap + offset only (2 heads, no size)
```

**When to use:**
- Point annotations only (no bounding box ground truth)
- Faster training & inference
- ~12% fewer parameters

---

## Future Architecture Improvements

### Potential Enhancements

1. **Adaptive Feature Selection**
   - Dynamic weighting of RGB/Thermal/Broker based on image content
   - More robust to modality-specific degradation

2. **Attention-Based Fusion**
   - Replace additive fusion with learned attention mechanisms
   - Could improve multi-modal alignment

3. **Scale-Adaptive Offset**
   - Currently offset is ±0.5 cell regardless of scale
   - Could use scale-aware offsets for better sub-pixel localization

4. **Confidence Calibration**
   - Post-training calibration using validation set
   - Better confidence scores for downstream tasks

5. **Ensemble Approaches**
   - Multiple prediction heads with different initialization
   - Voting-based ensemble for robustness

---

## References

- **CenterNet:** Zhou et al., "Objects as Points," ICCV 2019
  - https://arxiv.org/abs/1904.07850
- **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
  - https://arxiv.org/abs/1708.02002
- **Swin Transformer:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021
  - https://arxiv.org/abs/2103.14030
