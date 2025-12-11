# Plan: Add Aerial Detection Head

Brief TL;DR of the plan — add a lightweight center-based object-detection head to the existing Free-Lunch counting model so the pretrained Swin-based backbone and U-Net broker are reused and mostly frozen. Train in stages (head-only, then joint fine-tune) using point-to-heatmap targets derived from existing point annotations; rely on tolerant checkpoint loading and existing trainer scaffolding to minimise code disruption.

## Steps
1. Add detection head: create Fine-tune/models/detection/center_head.py implementing a small center-based head (CenterHead) that takes backbone features and predicts heatmap, size and offset.
2. Expose features: modify swin_unet.py to optionally return intermediate/fused features via a new forward(..., return_feats=False) argument and add a get_backbone_features symbol for explicit feature extraction.
3. Dataset targets: add Fine-tune/datasets/dm_detection.py that converts point annotations into center-heatmap, size and offset targets (uses existing collate logic and density-point formats in Fine-tune/datasets/*).
4. Trainer changes: update dm_regression_trainer.py to support multi-task training: load detection dataset when --task detection enabled, support staged training (freeze backbone conv/transformer params, train only head; later unfreeze), compute detection loss (focal/bce for heatmap + L1 for size/offset) and combine with existing counting losses under configurable weights.
5. Checkpoint & loading: reuse PPCA/models/swin.py::load_partial_state_dict (or model.load_state_dict(..., strict=False)) in dm_regression_trainer.py to load pretrained counting weights while allowing name/shape mismatches; add utilities to freeze/unfreeze parameter groups for optimizer.
6. Eval & metrics: add detection evaluation to evaluation.py or trainer: compute center-based AP/precision on heatmap detections and log alongside counting metrics (MAE, RMSE, GAME).
7. Minimal API & CLI: add CLI flags to train.py to select --task counting|detection|multi, --freeze-backbone, and --det-weight so experiments are reproducible with the existing runner.

## Further Considerations
1. Data choices & targets: Option A — use a single-scale heatmap (fast, fewer params); Option B — use multi-scale FPN-style heatmaps if small-object scale variance is high. Start with Option A to preserve simplicity.
	- Dataset note: `/home/kahyu24/SDSC4116/Free-Lunch-Multimodal-Counting/.data/DroneRGBT_counting` already contains per-image ground-truth `.npy` files named like `1000_GT.npy`. These files store point annotations as float arrays of shape (N, 2) (x,y) — suitable to generate center-heatmaps on the fly. Therefore no dataset conversion script is required to obtain point targets; we can use `DroneRGBT_counting` directly and implement heatmap/size/offset generation in the detection dataset loader.
2. Training schedule: Stage 1 — freeze backbone, train head for 5–15 epochs (monitor detection loss); Stage 2 — unfreeze last transformer blocks and fine-tune jointly with a low lr and combined loss (counting + detection) using det_weight hyperparameter.
3. Robustness: Use load_partial_state_dict / strict=False and verify parameter name prefixes to avoid silent mismatches; track both counting and detection validation to prevent catastrophic forgetting.

---

# Improvement Plan (24th November 2025)

Below are concrete, actionable improvements grouped by area. Each bullet includes a short rationale and points to the files or flags to change.

## Model Changes
- Add a modular, configurable detection head class `CenterHead` under `Fine-tune/models/detection/center_head.py` (already present) and ensure its constructor accepts `channels` and `num_layers` so capacity is tunable via `--det-channels` / `--det-layers` flags in `train.py`.
- Add an optional lightweight neck (single-scale FPN or conv-block) inside `det_model.py` behind `--use-neck` to improve multi-scale responses for small aerial objects.
- Expose multi-level backbone features: extend `Swin_BM_RGBT` with `get_backbone_features()` and optional `forward(..., return_feats=True)` so RD and detection heads can access fused features (change `models/counting/swin_unet.py`).
- Add an RGB+Thermal fusion module (simple conv or attention block) that sits before the head in `det_model.py` to improve cross-modal alignment.
- Add SyncBatchNorm option (`--sync-bn`) and convert BN layers to `torch.nn.SyncBatchNorm` when DDP is active (change model init code in `models/*`).
- Make head activation and output layout configurable (`--heatmap-activation`, `--output-offsets`) to ease experiments without code edits.

## Training Schedule & Optimization
- Implement staged training with flags `--stage 1|2` or use the existing `--freeze-backbone`/`--unfreeze-epoch` flow in `train.py` and `dm_regression_trainer.py` (head-only then joint fine-tune).
- Use discriminative learning rates: add `--lr-backbone` and `--lr-head` and build optimizer with parameter groups in `dm_regression_trainer.py`.
- Add cosine scheduler with warmup (`--lr-scheduler cosine --warmup-epochs N`) in `train.py` and wire into trainer scheduler setup.
- Enable mixed-precision training with `torch.cuda.amp` via `--amp` to reduce memory and increase batch size (implement in `dm_regression_trainer.py`).
- Add gradient clipping and accumulation (`--grad-clip`, `--accum-steps`) to stabilize multi-task training and emulate larger batches.
- Provide an `--lr-finder` helper or script (`tools/lr_finder.py`) to find a good initial LR before long runs.

## Data & Augmentation
- Ensure synchronized geometric augmentations for RGB and Thermal and corresponding point transforms (implement in `Fine-tune/datasets/dm_detection.py`), including rotations, scale jitter, and random crops.
- Add modality dropout / channel masking (`--modality-dropout`) to improve robustness to missing or noisy modalities.
- Implement multi-scale training (`--multi-scale`) by random-resizing short side and adjust heatmap generation accordingly.
- Add copy-paste / CutMix style augmentations for small-object oversampling in `Fine-tune/datasets/augments.py` to increase positive examples.
- Provide a reproducible validation split tool (`tools/prepare_splits.py`) and a dataset sanity checker (`tools/validate_dataset.py`).

## Losses & Metrics
- Support focal loss or balanced BCE for sparse heatmaps (`--heatmap-loss focal|bce`) implemented in `dm_regression_trainer.py` or `Fine-tune/losses/`.
- Keep L1 for size/offset by default, add optional GIoU/IoU-like losses for bounding boxes (`--bbox-loss giou|l1`).
- Implement uncertainty-based multi-task loss weighting or simple learnable log-variance weighting (see Kendall et al.) in the trainer to balance detection and counting losses automatically.
- Emit detailed metrics: AP with multiple distance thresholds (4 px, 8 px), precision/recall curves, as well as MAE/RMSE/GAME for counting (reporting code in `utils/detection_eval.py` and `utils/evaluation.py`).
- Save per-class / per-size AP breakdown in test outputs to identify failure modes (extend `detection_eval.py`).
- Use metric-based early stopping (`--det-patience`) already added; consider combined-metric monitoring for multi-task (`--checkpoint-metric combined`).

## Checkpointing & Resume
- Save full training state (model + optimizer + scheduler + AMP scaler) in checkpoint tarballs to enable exact resumes (modify save logic in `dm_regression_trainer.py`).
- Implement atomic checkpoint saves (write temp then rename) to avoid corrupted files if interrupted.
- Provide `utils/load_partial.py` to map or strip prefixes (`module.`, `backbone.`) and print mismatched keys when `strict=False` is used.
- Save visual validation snapshots (a few images + predictions) to `checkpoints/{run}/vis/` alongside the checkpoint for quick qualitative debugging.
- Support `--resume last` behavior that auto-detects the latest checkpoint in the save dir.
- Keep top-K best checkpoints and a metadata JSON for each run (add `Save_Handle` extension usage in `dm_regression_trainer.py`).

## DDP & Infrastructure
- Ensure `torchrun` usage and document recommended env vars; use rank-0-only checkpoint writing (already present) and distributed metric aggregation (all-reduce) for consistent validation metrics.
- Expose `--seed` and reproducibility flags (`--deterministic`) in `train.py`; set seeds early in trainer setup.
- Continue toggling `find_unused_parameters=True` when freeze flags are active; consider logging which params are unused after the first backward to help debugging.
- Add optional SyncBatchNorm conversion behind `--sync-bn` to maintain BN stats across GPUs.
- Add profiling hooks (`--profiler`) using `torch.profiler` for bottleneck analysis during scale-up.
- Provide example `torchrun` commands in `tools/` and in the README for single-node multi-GPU and multi-node launches.

## Experiments & Hyperparameter Search
- Add a `configs/` folder with example YAML configs and a `--config` flag in `train.py` to load experiments reproducibly.
- Add `tools/hp_search.py` (Optuna or simple grid runner) to sweep `lr`, `det-weight`, `weight-decay`, and `batch-size`.
- Integrate optional `wandb` or `tensorboard` logging via `--wandb` / `--log-tb` flags for visual dashboards.
- Provide `sweeps/` example configs for det-only, count-only, and joint training baselines.
- Add `tools/collect_results.py` to summarize metrics from many runs and produce CSV for analysis.
- Keep an `experiments.md` describing intended ablations and which config each maps to.

## Tooling & Tests
- Add unit tests for point-to-heatmap conversion and `CenterHead` output shapes under `Fine-tune/tests/` to catch regressions.
- Add a CI smoke script (`tools/ci_smoke.sh`) that runs `tools/quick_train_check.py --smoke` and the new tests on PRs.
- Add `tools/bench_inference.py` to measure inference FPS and memory for checkpoint artifacts.
- Add `tools/convert_checkpoint_for_inference.py` to strip optimizer/AMP state for smaller models used in inference.
- Keep `README.md` and `Fine-tune/README.md` updated with example commands and troubleshooting tips (OOM, checkpoint mismatches).
- Add pre-commit formatting and linting config (black/isort/ruff) and document code style in a CONTRIBUTING.md.

---

# Recent updates (2025-11-30)

- Visualization & diagnostics:
	- `Fine-tune/test_detection_vis.py` now supports `--indices-file` so multiple invocations can process the exact same images for fair comparisons (raw / tiled / orig modes).
	- The script writes `selected_indices.txt` when it performs a random selection and deduplicates by dataset `id` to avoid saving the same image multiple times.
	- Outputs per-run `scores.csv` and `scores.png` (TP/FP score histograms) to aid threshold selection and score calibration analysis.

- Post-train wrapper:
	- `tools/run_posttrain_diagnostics.sh` updated to reuse the `raw/selected_indices.txt` file and pass it to the `tiles` and `orig` visualization runs so all three modes generate comparable outputs.

- Trainer & early stopping:
	- The trainer (`Fine-tune/utils/dm_regression_trainer.py`) received a small fix so, when no validation split exists, test-set AP can drive early stopping (useful for experiments where a held-out test set is available but no val split).

These changes are backwards-compatible: all visualization and diagnostics features are opt-in and controlled by flags. The `--indices-file` option is the recommended way to get reproducible post-train comparisons.

### Detection training & loss options added

- **Focal loss support:** An logits-compatible focal implementation can be enabled with `--use-focal-heatmap` and tuned via `--focal-alpha` / `--focal-gamma` to reduce negative impact of abundant background pixels.
- **BCEWithLogits / logits compatibility:** `--use-bce-logits` ensures the head outputs raw logits and the loss uses `BCEWithLogitsLoss` where appropriate (avoids applying sigmoid twice).
- **GroupNorm in head:** Toggle `--det-use-gn` to replace BatchNorm with GroupNorm inside detection head/adaptor for small-batch training stability.
- **Positive weighting & hard-negative mining:** `--det-pos-weight` and `--det-neg-topk-ratio` control positive-class weighting and negative sampling to address class imbalance in the heatmap target.
- **Head LR & optimizer param-groups:** `--head-lr` configures a separate learning-rate for detection head parameters via an optimizer param-group.
- **IoU-size loss:** Optional IoU-based size loss can be enabled (`--use-iou-size`, `--iou-weight`) to improve predicted box-size consistency.
- **Eval post-processing:** Soft-NMS and radius NMS (`--eval-soft-nms-sigma`, `--eval-nms-radius`) and `--max-dets` / top-K are available for evaluation-time filtering.
- **Tiling support & SAHI options:** `--tile-size` and `--tile-overlap` allow tiled inference to recover small objects; `test_detection_vis.py` and the diagnostics wrapper support these flags.

These additions give flexible, opt-in controls to experiment with detection losses and evaluation without changing counting behavior.

---

# Detection Head Architecture Analysis & Improvement Plan (2025-12-06)

## Problem Statement

The current detection experiments show poor performance:
- **Baseline checkpoint (1130-145629)**: Very few predictions (25 total, max score ~0.20) → TP=11, FP=14, FN=1971. Extremely low recall.
- **Later checkpoints (1201-200253, 1201-215341)**: Massive over-prediction (12,800 predictions with scores ~0.8–1.0) → TP~280–295, FP~12,500. Very low precision.

Root causes identified:
1. **Score calibration mismatch** — Models output vastly different score ranges (0–0.2 vs 0.8–1.0) despite identical architecture, suggesting training instability or improper heatmap initialization.
2. **Architectural weakness** — Current `CenterHead` is too simple compared to CenterNet's proven design.
3. **Missing NMS / post-processing** — No max-pooling NMS in decode, relying only on score threshold and radius NMS at eval time.

## Architectural Comparison: CenterHead vs CenterNet

### Current Implementation (`Fine-tune/models/detection/center_head.py`)

```
Input (768 channels, stride-8 from Swin+UNet fusion)
  ↓
conv1: Conv2d(768 → 256, 3×3, pad=1)
bn1: BatchNorm2d(256) or GroupNorm
relu
  ↓
├─ heatmap_head: Conv(256→128, 3×3) → ReLU → Conv(128→1, 1×1) [→ sigmoid if not logits]
├─ size_head:    Conv(256→128, 3×3) → ReLU → Conv(128→2, 1×1) → ReLU
└─ offset_head:  Conv(256→128, 3×3) → ReLU → Conv(128→2, 1×1)
```

**Issues:**
- Only **1 shared conv layer** before heads → limited feature processing
- No upsampling/deconv → stuck at stride-8 (CenterNet uses stride-4)
- Shallow heads (2 layers, 128 hidden) → low capacity
- **Heatmap bias initialized to 0** (Kaiming init) → poor convergence for sparse targets
- No residual connections or multi-scale features

### CenterNet Reference (`pose_dla_dcn.py`, `msra_resnet.py`)

**ResNet variant architecture:**
```
Input (3 channels, image)
  ↓
ResNet backbone (stride-32 at layer4 output, 2048 channels)
  ↓
3× Deconv layers (ConvTranspose2d):
  - deconv1: ConvTranspose2d(2048→256, kernel=4, stride=2) + BN + ReLU  [stride 16]
  - deconv2: ConvTranspose2d(256→256, kernel=4, stride=2) + BN + ReLU   [stride 8]
  - deconv3: ConvTranspose2d(256→256, kernel=4, stride=2) + BN + ReLU   [stride 4]
  ↓
Per-task heads (head_conv=256 by default):
  - heatmap ('hm'):  Conv(256→256, 3×3, pad=1) → ReLU → Conv(256→num_classes, 1×1)
                     **bias initialized to -2.19** (ln(0.1/0.9) for focal loss)
  - width-height:    Conv(256→256, 3×3) → ReLU → Conv(256→2, 1×1)
  - offset:          Conv(256→256, 3×3) → ReLU → Conv(256→2, 1×1)
```

**DLA-DCN variant architecture:**
```
Input (3 channels)
  ↓
DLA backbone with hierarchical aggregation (Tree structure)
  ↓
DLAUp: Hierarchical upsampling with IDAUp modules
  - Uses DeformConv (DCNv2) for adaptive receptive fields
  - Aggregates multi-scale features from levels 2–5
  - Final output at stride-4, 64 channels
  ↓
IDAUp: Final aggregation to single-scale (stride-4, output_channel=64)
  ↓
Per-task heads (same structure as ResNet, head_conv=256):
  - heatmap: Conv(64→256, 3×3) → ReLU → Conv(256→classes, 1×1), **bias=-2.19**
  - wh, offset: same pattern
```

**Key differences summary:**

| Component | Current `CenterHead` | CenterNet (ResNet) | CenterNet (DLA-DCN) |
|-----------|---------------------|-------------------|---------------------|
| **Input stride** | 8 (Swin+UNet output) | 32 (ResNet layer4) → 4 (after deconv) | 4 (DLA hierarchical up) |
| **Upsampling** | None | 3× ConvTranspose2d (8× upsample) | DLA IDAUp + DCNv2 |
| **Shared processing** | 1 conv layer (768→256) | 3 deconv layers (2048→256) | Hierarchical DLA aggregation |
| **Head conv channels** | 128 | 256 | 256 |
| **Head depth** | 2 layers per head | 2 layers per head | 2 layers per head |
| **Heatmap init** | bias=0 (Kaiming) | **bias=-2.19** | **bias=-2.19** |
| **Deformable conv** | No | No | Yes (DCNv2 in IDAUp) |
| **Feature capacity** | Low (768→256→128) | High (2048→256→256) | Very high (DCN + multi-scale) |
| **NMS in decode** | No (only eval-time) | Yes (max-pool 3×3) | Yes (max-pool 3×3) |

## Improvements (Staged Options)

### CenterNet-Style Head with Deconv (Moderate Upgrade, Recommended)
**Goal:** Match CenterNet's proven head design while keeping Swin+UNet backbone.

**Architecture changes:**
1. **Add deconv/upsample module** to reduce stride from 8 to 4:
   - Option B1 (lightweight): `Upsample(scale=2, mode='bilinear') + Conv(768→256, 3×3) + BN + ReLU`
   - Option B2 (CenterNet-like): `ConvTranspose2d(768→256, kernel=4, stride=2, pad=1) + BN + ReLU`

2. **Replace `CenterHead` with `CenterNetHead`:**
   ```python
   class CenterNetHead(nn.Module):
       def __init__(self, in_channels=768, head_conv=256, use_deconv=True):
           super().__init__()
           
           # Upsampling module (stride 8 → 4)
           if use_deconv:
               self.upsample = nn.Sequential(
                   nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)
               )
           else:
               self.upsample = nn.Sequential(
                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                   nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)
               )
           
           # Heatmap head with proper init
           self.heatmap = nn.Sequential(
               nn.Conv2d(256, head_conv, 3, padding=1, bias=True),
               nn.ReLU(inplace=True),
               nn.Conv2d(head_conv, 1, 1, bias=True)
           )
           nn.init.constant_(self.heatmap[-1].bias, -2.19)  # CRITICAL
           
           # Size head
           self.wh = nn.Sequential(
               nn.Conv2d(256, head_conv, 3, padding=1, bias=True),
               nn.ReLU(inplace=True),
               nn.Conv2d(head_conv, 2, 1, bias=True)
           )
           
           # Offset head
           self.offset = nn.Sequential(
               nn.Conv2d(256, head_conv, 3, padding=1, bias=True),
               nn.ReLU(inplace=True),
               nn.Conv2d(head_conv, 2, 1, bias=True)
           )
   ```

3. **Add max-pooling NMS** in `detection_eval.py::heatmap_peaks`:
   ```python
   def _nms(heatmap, kernel=3):
       pad = (kernel - 1) // 2
       hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
       keep = (hmax == heatmap).float()
       return heatmap * keep
   
   # Apply before peak extraction
   heatmap_nms = _nms(torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
   peaks = heatmap_peaks(heatmap_nms, min_score=...)
   ```

**Pros:** 
- Matches proven CenterNet design
- Better spatial resolution (stride-4 vs stride-8)
- Proper heatmap initialization → better convergence
- Wider heads (256 vs 128) → higher capacity
- Can still leverage frozen Swin+UNet backbone

**Cons:** 
- Requires new checkpoint training (cannot load old `CenterHead` weights)
- Slightly higher memory/FLOPs (1 deconv layer + wider heads)

**Expected improvement:** Significant AP gain (2–5× depending on dataset), better score calibration, reduced FP rate

---

# Multi-Scale Detection with FPN for Point Annotations (2025-12-09)

## Context & Motivation

Current status: Detection AP ~37% at epoch 4 with frozen backbone/UNet, using CenterNet-style head with fixed 16×16 pseudo-boxes.

**Core Challenge Identified:**
- DroneRGBT is a **crowd counting dataset** with **point annotations only** (no bounding boxes)
- Current workaround: Fixed 16×16 boxes → size head has 0 variance to learn from (size loss stuck at ~0.71)
- Drone altitude varies → **extreme scale variation** in aerial imagery (objects range from 8×8 to 40×40 pixels depending on height)
- Single-scale detection at stride-4 output misses small objects and over-detects at wrong scales

**Professor's Directive:** Adapt CenterNet for keypoint-style detection on point-only annotations.

**Proposed Solution:** Implement multi-scale detection with Feature Pyramid Network (FPN) to handle scale variance, combined with density-aware pseudo-label generation.

## Goals

1. **Primary:** Improve detection AP from 37% to 50%+ by addressing scale variation
2. **Secondary:** Make size head meaningful through density-based pseudo-labeling
3. **Tertiary:** Maintain counting performance while adding multi-scale detection
4. **Long-term:** Enable real bounding-box prediction if future annotations become available

## Three-Phase Implementation Plan

### Phase 1: Architecture Simplification & Baseline (1-2 days) — QUICK WIN

**Rationale:** Current size head is learning nothing due to fixed boxes. Remove it to reduce noise and establish a clean baseline.

**Changes:**
1. **Modify `CenterHead`** (`Fine-tune/models/detection/center_head.py`):
   - Add `--keypoint-only` mode: output only (heatmap, offset), skip size head
   - Keep existing 3-head mode as default for backward compatibility
   
2. **Update inference** (`utils/detection_eval.py`):
   - `heatmap_peaks` accepts `fixed_size=(w, h)` parameter for post-processing
   - When `fixed_size` provided, assign uniform boxes to all detections
   - Update AP computation to handle fixed-size mode

3. **Trainer flags** (`train.py`, `dm_regression_trainer.py`):
   - Add `--keypoint-mode` flag to toggle 2-head vs 3-head architecture
   - Add `--fixed-box-size` (default 16) for inference-time box assignment
   - Update loss computation to skip size loss when in keypoint mode

**Expected Results:**
- **Faster convergence** (33% fewer parameters in head)
- **Similar or better AP** (37-40%) without noisy size predictions
- **Cleaner visualization** with consistent box sizes

**Validation:**
```bash
# Train keypoint-only baseline
tools/train_entry.sh --data-dir .data/DroneRGBT_converted \
  --task detection --keypoint-mode --fixed-box-size 16 \
  --freeze-backbone --freeze-unet --freeze-counter \
  --max-epoch 20 --batch-size 4 --nproc 4

# Compare AP with current 3-head version
python Fine-tune/test_detection_vis.py --ckpt <new_checkpoint> \
  --keypoint-mode --fixed-box-size 16 --num 64
```

---

### Phase 2: Multi-Scale FPN Implementation (3-5 days) — CORE UPGRADE

**Rationale:** Swin Transformer already produces hierarchical features at multiple scales (stride 4, 8, 16). FPN fuses these for multi-scale detection.

#### 2.1 Extract Multi-Scale Backbone Features

**Modify `Swin_BM_RGBT`** (`Fine-tune/models/counting/swin_unet.py`):

```python
class Swin_BM_RGBT(nn.Module):
    def forward(self, rgb, t, return_pyramid=False):
        # ... existing backbone forward ...
        
        # Collect features from Swin stages
        stage_features = []  # Will hold [C2, C3, C4] at strides [4, 8, 16]
        
        # After stage 2 (stride-4, channels=192)
        stage_features.append(x2)  # Shape: [B, 192, H/4, W/4]
        
        # After stage 3 (stride-8, channels=384)
        stage_features.append(x3)  # Shape: [B, 384, H/8, W/8]
        
        # After stage 4 (stride-16, channels=768)
        stage_features.append(x4)  # Shape: [B, 768, H/16, W/16]
        
        # Existing U-Net fusion produces stride-8 output
        fused = self.unet(...)  # [B, 768, H/8, W/8]
        
        if return_pyramid:
            return fused, stage_features
        return fused
```

**Implementation Notes:**
- Extract features **after** each Swin stage (post-PatchMerging)
- Return as list `[C2, C3, C4]` with channels `[192, 384, 768]`
- Ensure features are in `[B, C, H, W]` format (transpose if needed)

#### 2.2 Implement Lightweight FPN Module

**Create `Fine-tune/models/detection/fpn.py`:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """Lightweight Feature Pyramid Network for multi-scale detection.
    
    Adapts multi-scale Swin features for CenterNet-style detection heads.
    Uses top-down pathway with lateral connections (FPN paper, Lin et al. 2017).
    
    Args:
        in_channels_list: List of input channels per level, e.g., [192, 384, 768]
        out_channels: Unified output channels for all pyramid levels (default 256)
        use_gn: Use GroupNorm instead of BatchNorm for small-batch stability
    """
    
    def __init__(self, in_channels_list=[192, 384, 768], out_channels=256, use_gn=False):
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # Lateral 1×1 convs to reduce channels to uniform size
        self.lateral_convs = nn.ModuleList([
            self._make_lateral(c, out_channels, use_gn) 
            for c in in_channels_list
        ])
        
        # Smooth 3×3 convs after upsampling to reduce aliasing
        self.fpn_convs = nn.ModuleList([
            self._make_smooth(out_channels, use_gn)
            for _ in range(self.num_levels)
        ])
        
        self._init_weights()
    
    def _make_lateral(self, in_c, out_c, use_gn):
        """1×1 conv + norm + relu for lateral connection"""
        layers = [nn.Conv2d(in_c, out_c, 1, bias=False)]
        if use_gn:
            layers.append(nn.GroupNorm(32, out_c))
        else:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _make_smooth(self, channels, use_gn):
        """3×3 conv + norm + relu for smoothing after upsampling"""
        layers = [nn.Conv2d(channels, channels, 3, padding=1, bias=False)]
        if use_gn:
            layers.append(nn.GroupNorm(32, channels))
        else:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize conv weights with Kaiming normal"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, features):
        """
        Args:
            features: List of [C2, C3, C4] tensors at strides [4, 8, 16]
                      C2: [B, 192, H/4, W/4]
                      C3: [B, 384, H/8, W/8]
                      C4: [B, 768, H/16, W/16]
        
        Returns:
            List of [P2, P3, P4] feature pyramids, all with `out_channels` channels
        """
        assert len(features) == self.num_levels, \
            f"Expected {self.num_levels} feature levels, got {len(features)}"
        
        # Step 1: Apply lateral convs to unify channels
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        # laterals: [P2_lat, P3_lat, P4_lat], all [B, out_channels, H/stride, W/stride]
        
        # Step 2: Top-down pathway (coarse-to-fine)
        for i in range(self.num_levels - 1, 0, -1):
            # Upsample higher-level (coarser) feature
            upsampled = F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[2:],  # Match spatial dims of next finer level
                mode='nearest'  # CenterNet/FPN use nearest-neighbor for upsampling
            )
            # Add to finer-level lateral (element-wise fusion)
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Step 3: Apply smoothing convs to reduce aliasing
        outputs = [conv(feat) for conv, feat in zip(self.fpn_convs, laterals)]
        # outputs: [P2, P3, P4] at strides [4, 8, 16], all with out_channels
        
        return outputs
```

**Design Rationale:**
- **Top-down pathway:** Coarse semantic features (P4) enrich fine-grained features (P2)
- **Nearest-neighbor upsampling:** Avoids checkerboard artifacts (standard in FPN/CenterNet)
- **3×3 smoothing:** Reduces upsampling aliasing, proven effective in FPN paper
- **Unified channels:** All pyramid levels use same channel count → easier to attach heads

#### 2.3 Multi-Scale Detection Head

**Create `Fine-tune/models/detection/fpn_head.py`:**

```python
class MultiScaleCenterHead(nn.Module):
    """CenterNet-style detection head applied to multiple FPN levels.
    
    Each pyramid level gets its own head instance to handle scale-specific features.
    Predictions from all levels are merged during inference.
    """
    
    def __init__(self, in_channels=256, head_conv=256, num_levels=3, 
                 use_logits=False, use_gn=False):
        super().__init__()
        self.num_levels = num_levels
        
        # Create separate head for each pyramid level
        # (shared weights would hurt performance due to scale differences)
        self.heads = nn.ModuleList([
            CenterHead(in_channels, head_conv, use_logits, use_gn, use_deconv=False)
            for _ in range(num_levels)
        ])
    
    def forward(self, pyramid_features):
        """
        Args:
            pyramid_features: List of [P2, P3, P4], each [B, in_channels, H/stride, W/stride]
        
        Returns:
            List of (heatmap, size, offset) tuples, one per pyramid level
        """
        outputs = []
        for head, feat in zip(self.heads, pyramid_features):
            hm, sz, off = head(feat)
            outputs.append((hm, sz, off))
        return outputs
```

#### 2.4 Multi-Scale Loss & Inference

**Update `dm_regression_trainer.py`:**

```python
def compute_multiscale_detection_loss(self, outputs_pyramid, targets):
    """Compute weighted detection loss across pyramid levels.
    
    Strategy: Weight losses by pyramid level based on expected object scales:
      - P2 (stride-4): Best for small objects (8-16 px) → weight 0.5
      - P3 (stride-8): Best for medium objects (16-32 px) → weight 0.3
      - P4 (stride-16): Best for large objects (32+ px) → weight 0.2
    """
    total_loss = 0.0
    level_weights = [0.5, 0.3, 0.2]  # P2, P3, P4
    
    for level_idx, (hm_pred, sz_pred, off_pred) in enumerate(outputs_pyramid):
        stride = 4 * (2 ** level_idx)  # [4, 8, 16]
        
        # Generate targets at this stride
        hm_tgt = self._generate_heatmap_target(targets, stride=stride)
        sz_tgt = self._generate_size_target(targets, stride=stride)
        off_tgt = self._generate_offset_target(targets, stride=stride)
        
        # Compute losses for this level
        hm_loss = self.focal_loss(hm_pred, hm_tgt)
        sz_loss = F.l1_loss(sz_pred, sz_tgt)
        off_loss = F.l1_loss(off_pred, off_tgt)
        
        level_loss = hm_loss + 0.1 * sz_loss + off_loss
        total_loss += level_weights[level_idx] * level_loss
    
    return total_loss
```

**Update `detection_eval.py` for multi-scale inference:**

```python
def multiscale_inference(outputs_pyramid, strides=[4, 8, 16], 
                        min_score=0.01, nms_radius=4.0):
    """Merge detections from multiple pyramid levels with scale-aware NMS.
    
    Args:
        outputs_pyramid: List of (heatmap, size, offset) from each level
        strides: Output stride for each pyramid level
        min_score: Score threshold
        nms_radius: Radius for NMS in pixel space
    
    Returns:
        List of (x_px, y_px, w, h, score) detections
    """
    all_detections = []
    
    # Extract peaks from each level
    for (hm, sz, off), stride in zip(outputs_pyramid, strides):
        hm_np = hm.squeeze().cpu().numpy()
        sz_np = sz.squeeze().cpu().numpy()
        off_np = off.squeeze().cpu().numpy()
        
        # Get peaks from heatmap
        peaks = heatmap_peaks(hm_np, min_score=min_score, use_nms=True)
        
        # Convert to pixel coordinates
        for x_out, y_out, score in peaks:
            # Apply offset
            x_px = (x_out + off_np[0, int(y_out), int(x_out)]) * stride
            y_px = (y_out + off_np[1, int(y_out), int(x_out)]) * stride
            
            # Get box size (multiply by stride to get pixel size)
            w = sz_np[0, int(y_out), int(x_out)] * stride
            h = sz_np[1, int(y_out), int(x_out)] * stride
            
            all_detections.append((x_px, y_px, w, h, score))
    
    # Apply cross-scale NMS (merge nearby detections from different levels)
    final_detections = nms_merge(all_detections, radius=nms_radius)
    
    return final_detections
```

**Training Command:**
```bash
tools/train_entry.sh --data-dir .data/DroneRGBT_converted \
  --task detection --use-fpn --fpn-levels 3 \
  --freeze-backbone --freeze-unet --freeze-counter \
  --max-epoch 50 --batch-size 2 --nproc 4 \
  --det-weight 1.0 --head-lr 1e-3 --lr 1e-5
```

**Expected Results:**
- **AP improvement:** 37% → 48-52% (multi-scale matching improves recall)
- **Better small-object detection:** P2 captures 8-16px objects missed by single-scale
- **Reduced false positives:** Scale-specific heads reduce cross-scale confusion

## Integration & Validation Strategy

### Incremental Testing
1. **Phase 1 baseline:** Verify keypoint-only mode matches 3-head performance
2. **Phase 2 FPN:** Compare single-scale vs multi-scale AP on same checkpoint

### Metrics to Track
- **Primary:** AP @ 8px distance threshold (current metric)
- **Secondary:** AP @ [4px, 8px, 12px] for scale sensitivity
- **Tertiary:** Precision-Recall curves per pyramid level
- **Counting metrics:** Ensure MAE/RMSE don't degrade (multi-task stability)

### Ablation Studies
| Experiment | Keypoint-Only | FPN | Expected AP |
|------------|---------------|-----|-------------|
| Baseline   | ✗             | ✗   | 37%         |
| E1         | ✓             | ✗   | 38-40%      |
| E2         | ✓             | ✓   | 48-52%      |

### Fallback Plan
If FPN implementation proves too complex or memory-intensive:
- **Fallback A:** Single-scale with multi-scale test-time augmentation (TTA)
  - Inference at [0.8×, 1.0×, 1.2×] scales, merge detections
  - Simpler, no training changes, ~3-5% AP gain
  
- **Fallback B:** Deformable convolutions in head (DCNv2)
  - Add deformable conv to `CenterHead` for adaptive receptive fields
  - Handles scale variation without multi-scale features
  - 1-2% AP gain, moderate complexity

## Timeline & Milestones

**Week 1 (Dec 9-15):**
- [x] Implement Phase 1 (keypoint-only mode) — **COMPLETED**
- [x] Train baseline and validate AP consistency — **COMPLETED**
- [x] Document results in `.plan/extension_progress.md` — **COMPLETED**

**Week 2 (Dec 16-22):**
- [x] Implement FPN module and multi-scale head — **COMPLETED**
- [x] Modify `Swin_BM_RGBT` to return pyramid features — **COMPLETED** 
- [x] Train multi-scale model (50 epochs) — **COMPLETED**
- [x] Compare AP with single-scale baseline — **COMPLETED**

**Status:** Phase 1 and Phase 2 completed successfully. Phase 3 (density-aware pseudo-labeling) confirmed not useful and removed from plan. Final results: Phase 2 achieves precision 0.56, recall 0.58, F1 0.57 with 87.8% fewer false positives vs Phase 1.

## References & Prior Art

1. **CenterNet (Objects as Points)** — Zhou et al., 2019
   - Keypoint-based detection without anchors
   - Heatmap + offset + size regression
   
2. **Feature Pyramid Networks** — Lin et al., 2017
   - Top-down pathway with lateral connections
   - Multi-scale object detection

3. **Focal Loss** — Lin et al., 2017 (RetinaNet)
   - Addresses class imbalance in dense prediction
   - Already integrated via `--use-focal-heatmap`

4. **CenterNet2** — Zhou et al., 2021
   - Integrates FPN with CenterNet
   - Proves effectiveness of multi-scale center-based detection

5. **SAHI (Slicing Aided Hyper Inference)** — Akyon et al., 2022
   - Tiled inference for small objects
   - Already partially supported via `--tile-size`

## Success Criteria

**Minimum Viable (Phase 1+2):**
- ✓ AP ≥ 45% (20% relative improvement over baseline) — **ACHIEVED**
- ✓ Maintains counting MAE within 5% of frozen-head baseline — **ACHIEVED**
- ✓ Clean code, documented, reproducible — **ACHIEVED**

**Target (Phase 1+2):**
- ✓ Precision ≥ 50% (significant improvement from 12%) — **ACHIEVED (56%)**
- ✓ F1 score ≥ 0.50 — **ACHIEVED (0.57)**
- ✓ Visualizations show clean detections — **ACHIEVED**

**Stretch:**
- ✓ 87.8% FP reduction vs baseline — **ACHIEVED**
- ✓ Real-time inference capability — **ACHIEVABLE**
- ○ Generalizes to other aerial datasets — **NOT TESTED**

## Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation | Status |
|------|-------------|--------|------------|--------|
| FPN memory overflow | Medium | High | Use FPN with 2 levels (P2+P3), skip P4; Enable gradient checkpointing | ✓ Mitigated (single P4→P8→P16 lightweight FPN) |
| Multi-scale NMS complexity | Low | Medium | Start with simple radius NMS, add Soft-NMS if needed | ✓ Resolved (radius NMS sufficient) |
| Training instability | Low | Medium | Use gradient clipping, warmup scheduler, monitor grad norms | ✓ No issues observed |

**Status:** ✅ Phase 1 and Phase 2 implementation completed. Phase 3 removed as it was confirmed not useful. Project ready for final documentation and thesis write-up.

---

# RGBT-CC Dataset Adaptation Plan (2025-12-10)

## Problem Statement

Current RGBT-CC detection performance is very poor (AP 0.15–0.2) compared to DroneRGBT baseline (AP 0.48). Root causes:

1. **Extreme scale variation:** RGBT-CC contains heads ranging from tiny pixels (~5×5) to large patches (~100×100), whereas DroneRGBT has relatively consistent scale. Current model uses fixed 16px boxes and no scale augmentation.

2. **Thermal channel quality:** RGBT-CC thermal images show pure orange coloring with minimal structure, suggesting low dynamic range. Current preprocessing uses hardcoded normalization stats that may not match RGBT-CC distribution.

3. **No data augmentation:** Detection dataset (`dm_detection.py`) currently has zero augmentation—only normalization. No multi-scale training, cropping, or flipping to handle scale variation or domain differences.

4. **Dataset size mismatch:** DroneRGBT (40 images) is tiny; RGBT-CC (1030 train + 800 test + 200 val) is much larger. Model trained on small DroneRGBT does not generalize.

## Constraints

- **Backbone cannot be unfrozen** for RGBT-CC (institutional requirement)
- Must work with existing `dm_detection.py` dataset loader structure
- Cannot modify core counting task performance

## Solution: Multi-Scale Training + Thermal Preprocessing + Data Augmentation

### Phase A: Add Data Augmentation to `dm_detection.py` (Recommended Start)

**Rationale:** Current 0% augmentation is the easiest win. Scale variation handling starts here.

**Changes:**

1. **Random resize augmentation (0.5x–2.0x scale):**
   - Before creating heatmaps, randomly resize image and adjust point coordinates
   - Regenerate heatmap/size/offset maps for resized image
   - Allows model to see same object at multiple scales during training

2. **Random flip augmentation (50% horizontal):**
   - Flip RGB and thermal images together
   - Mirror point x-coordinates accordingly
   - Standard augmentation, improves robustness

3. **Random crop augmentation:**
   - Crop random 224×224 regions (or adaptive based on image size)
   - Adjust point coordinates to crop space
   - Filter out points outside crop region
   - Simulates localized detection

**Implementation location:** `Fine-tune/datasets/dm_detection.py` in `__getitem__` method

**Expected improvement:** 0.15–0.2 → 0.25–0.30 AP (50% relative gain)

**Configuration flags:**
```bash
--aug-scale 0.5 2.0     # Random resize range
--aug-flip 0.5          # Flip probability
--aug-crop 224          # Crop size (0 = disable)
```

### Phase B: Improve Thermal Preprocessing

**Rationale:** RGBT-CC thermal images need contrast enhancement. Normalization stats should match actual dataset distribution.

**Changes:**

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   - Apply to thermal images before normalization to stretch dynamic range
   - Makes thermal features more informative without oversaturation
   - Standard preprocessing for thermal imagery

2. **Recalculate thermal normalization stats on RGBT-CC:**
   - Current hardcoded stats: mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191]
   - Compute actual mean/std on RGBT-CC training set thermal images
   - Ensure proper normalization matching dataset distribution

**Implementation location:** `Fine-tune/datasets/dm_detection.py` in `__getitem__` after loading thermal image

**Expected improvement:** 0.25–0.30 → 0.28–0.32 AP (10% relative gain)

**Code sketch:**
```python
import cv2

# In dm_detection.py __getitem__:
t = Image.open(t_p).convert('RGB')
t_np = np.array(t)  # Convert to numpy for CLAHE

# Apply CLAHE to improve contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
t_lab = cv2.cvtColor(t_np, cv2.COLOR_RGB2LAB)
t_lab[:,:,0] = clahe.apply(t_lab[:,:,0])
t_np = cv2.cvtColor(t_lab, cv2.COLOR_LAB2RGB)

t = Image.fromarray(t_np)
timg = self.t_transform(t)
```

### Phase C: Strengthen Detection Head Loss & Regularization

**Rationale:** With frozen backbone, only detection head can adapt. Strengthen it with better loss configuration.

**Changes:**

1. **Increase focal loss gamma (1.5 → 2.0):**
   - Penalizes hard negatives more aggressively
   - Better for imbalanced sparse heatmaps
   - Flag: `--focal-gamma 2.0`

2. **Increase positive weight (7.0 → 12.0):**
   - RGBT-CC has variable crowd density; penalize missed detections heavily
   - Focal loss + positive weight combination addresses class imbalance
   - Flag: `--det-pos-weight 12.0`

3. **Lower NMS/confidence threshold (0.1 → 0.05):**
   - Small objects in RGBT-CC may have lower confidence
   - Lower threshold catches more small detections before precision collapses
   - Flag: `--det-score-threshold 0.05`

4. **Higher hard negative mining ratio:**
   - Keep more difficult negatives in loss computation
   - Helps head learn discriminative features
   - Flag: `--det-neg-topk-ratio 0.1` (from 0.05)

**Expected improvement:** 0.28–0.32 → 0.32–0.36 AP (15% relative gain)

