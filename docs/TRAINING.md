# Training Guide

## Quick Start

We strongly recommend using the convenience launcher `tools/train_entry.sh` for all training. It wraps `torchrun` safely, sets sensible defaults, and forwards extra flags to `Fine-tune/train.py`.

```bash
# Basic training
tools/train_entry.sh --data-dir <path-to-dataset>

# Resume from checkpoint
tools/train_entry.sh --data-dir <path-to-dataset> \
  --resume <path-to-checkpoint>

# See all options
tools/train_entry.sh --help
```

---

## Recommended Configurations

### Phase 6.5 (Current Best - Multi-Modal Detection)

**Ideal for:** Best performance with multi-modal RGB+Thermal data

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --use-deconv 1 \
  --use-fpn 0 \
  --keypoint-mode 0 \
  --head-conv 256 \
  --det-sigma 0.8 \
  --focal-alpha 0.75 \
  --focal-gamma 2.5 \
  --det-pos-weight 1.0 \
  --det-neg-topk-ratio 0.1 \
  --eval-nms-radius 2.0 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 100 \
  --batch-size 4 \
  --resume checkpoints_phase6/phase6.5_better_bias/best_model_epoch_68.pth
```

**Expected Results:**
- Detection AP: 0.56 (RAW), 0.53 (TILES), 0.45 (ORIG)
- Training time: ~4-6 hours on single GPU
- Convergence: ~60-80 epochs

**Key Parameters:**
- `det-sigma=0.8`: Sharp Gaussian targets for better localization
- `focal-alpha=0.75`: Strong focus on hard examples
- `det-neg-topk-ratio=0.1`: Aggressive negative sampling
- `eval-nms-radius=2.0`: Tight NMS to reduce false positives

---

### PhaseSlim (Lightweight - Keypoint Mode)

**Ideal for:** Fast training without bounding box supervision

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --use-deconv 1 \
  --use-fpn 0 \
  --keypoint-mode 1 \
  --head-conv 256 \
  --det-sigma 0.8 \
  --focal-alpha 0.75 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 50 \
  --batch-size 8 \
  --lr-det 0.001
```

**Benefits:**
- ~12% fewer parameters (no size head)
- 10-15% faster training
- Better for point-only annotations

---

### Multi-Scale (FPN)

**Ideal for:** Datasets with extreme scale variation

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --use-deconv 1 \
  --use-fpn 1 \
  --fpn-levels 3 \
  --keypoint-mode 1 \
  --fixed-box-size 16 \
  --head-conv 256 \
  --det-sigma 0.8 \
  --focal-alpha 0.75 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 60 \
  --batch-size 4
```

**Notes:**
- FPN adds ~10% parameters and training time
- Useful for datasets with 5-10× scale variation
- Standard Phase 6.3 configuration

---

### Single-GPU Training (Memory-Constrained)

**Ideal for:** Training on limited GPU memory

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --use-deconv 1 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --batch-size 1 \
  --lr-det 0.001 \
  --max-epoch 100 \
  --num-workers 0
```

**Trade-offs:**
- Batch size 1 = slower convergence
- No gradient accumulation = smaller effective batch size
- Training time: ~12-18 hours (vs 4-6 hours for batch-size 4)

---

## Advanced Training Options

### Staged Training (Backbone Unfreezing)

**Strategy:** Train head-only first, then fine-tune backbone

```bash
# Stage 1: Head-only training (5-10 epochs)
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 10 \
  --lr-det 0.001 \
  --save-interval 1

# Stage 2: Fine-tune everything (20-40 epochs)
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --freeze-counter \
  --unfreeze-epoch 0 \
  --lr-backbone 1e-5 \
  --lr-det 5e-5 \
  --max-epoch 50 \
  --resume <checkpoint-from-stage1>
```

**Benefits:**
- Better stability when unfreezing
- Gradual adaptation to detection task
- Reduced risk of catastrophic forgetting

---

### Mixed Precision Training

**Ideal for:** Reducing memory and increasing batch size

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --amp \
  --batch-size 8 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 80
```

**Benefits:**
- ~50% memory reduction
- ~20% faster training
- Maintained accuracy with proper loss scaling

---

### Gradient Accumulation (Larger Effective Batch)

**Ideal for:** Simulating larger batch on limited GPU memory

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --batch-size 2 \
  --accum-steps 4 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 60
```

**Effective batch size:** 2 × 4 = 8 (for gradient computation)

---

## Dataset-Specific Configurations

### DroneRGBT (Primary Dataset)

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --num-train 906 \
  --num-val 151 \
  --num-test 1806 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter
```

**Expected:**
- Training converges in 60-80 epochs
- No need for heavy augmentation (clear, well-lit imagery)
- Dropout/regularization: conservative

---

### RGBT-CC (Challenging Dataset)

RGBT-CC has extreme scale variation, small objects, and crowded scenes. Use aggressive augmentation:

```bash
tools/train_entry.sh \
  --data-dir .data/RGBT-CC_converted \
  --aug-scale-min 0.5 \
  --aug-scale-max 2.0 \
  --aug-flip 0.5 \
  --aug-crop 256 \
  --thermal-clahe 1 \
  --freeze-backbone \
  --freeze-unet \
  --freeze-counter \
  --max-epoch 120 \
  --batch-size 4
```

**Expected:**
- Longer training needed due to scale complexity
- CLAHE thermal preprocessing improves contrast
- Multi-scale augmentation essential
- AP may be lower (harder dataset) but more robust

**Augmentation Flags:**
- `aug-scale-min/max`: Scale range (0.5-2.0 = 50%-200%)
- `aug-flip`: Horizontal flip probability
- `aug-crop`: Random crop size (0 = disabled)
- `thermal-clahe`: CLAHE preprocessing

---

## Hyperparameter Tuning

### Learning Rate Tuning

**Conservative (Safe):**
```bash
--lr-backbone 1e-6
--lr-unet 1e-6
--lr-det 5e-4
```

**Moderate (Standard, Phase 6.5):**
```bash
--lr-backbone 1e-5
--lr-unet 1e-5
--lr-det 1e-3
```

**Aggressive (Risk of Instability):**
```bash
--lr-backbone 1e-4
--lr-unet 1e-4
--lr-det 5e-3
```

### Loss Weighting

**More aggressive detection training** (focus on detection):
```bash
--wot 0.0 \
--wtv 0.0 \
--wrd 0.0 \
--det-loss-weight 1.0
```

**Balanced detection + counting** (preserve counting ability):
```bash
--wot 0.5 \
--wtv 0.5 \
--wrd 0.5 \
--det-loss-weight 1.0
```

### Focal Loss Tuning

**Conservative (Less focus on hard examples):**
```bash
--focal-alpha 0.25
--focal-gamma 1.5
```

**Moderate (Standard, Phase 6.5):**
```bash
--focal-alpha 0.75
--focal-gamma 2.5
```

**Aggressive (Strong focus on hard examples):**
```bash
--focal-alpha 0.9
--focal-gamma 3.0
```

---

## Monitoring Training

### Key Metrics to Watch

**Detection Loss:**
- Should decay smoothly from ~3.0 to <0.5
- If loss plateaus high (~1.0+): LR may be too low
- If loss oscillates wildly: LR may be too high or gradient instability

**Validation AP:**
- Best metric for detection performance
- Should increase monotonically until plateau
- If AP decreases after epoch 30: possibly overfitting

**Counting Metrics (if training joint model):**
- MAE/RMSE should remain stable
- If counting degrades significantly: reduce `det-loss-weight`

### TensorBoard Monitoring

```bash
tensorboard --logdir=runs/ --port=6006
```

Then open http://localhost:6006

**Key plots to monitor:**
- `loss/det_heatmap`: Should decay (ideally exponential)
- `loss/det_size`: Should decay smoothly
- `metric/ap_val`: Should increase, then plateau
- `metric/mae`: Should remain constant (unless intentionally fine-tuning)

---

## Checkpoint Management

### Saving Checkpoints

```bash
--save-interval 5          # Save every 5 epochs
--save-top-k 3             # Keep best 3 checkpoints
--checkpoint-metric ap_val # Track AP metric
```

### Resume Training

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --resume <checkpoint-path> \
  --max-epoch 150           # Continue to epoch 150
```

### Load Pretrained (Different Architecture)

```bash
# Tolerant loading with strict=False
python3 -c "
import torch
ckpt = torch.load('some_checkpoint.pth')
model.load_state_dict(ckpt['model'], strict=False)
"
```

---

## Common Issues and Solutions

### Issue: Loss NaN or Inf

**Causes:** Learning rate too high, gradient explosion, bad data

**Solutions:**
```bash
# Reduce learning rate
--lr-det 5e-4

# Enable gradient clipping
--grad-clip 0.5

# Add debug logging
--log-level debug
```

### Issue: AP Plateaus at Low Value (<0.1)

**Causes:** Learning rate too low, frozen features, initialization issue

**Solutions:**
```bash
# Increase learning rate
--lr-det 5e-3

# Check if backbone is frozen (should be for detection-only)
--freeze-backbone 0        # Unfreeze if needed

# Verify data loading
python3 -c "from Fine-tune.datasets.dm_detection import *; ..."
```

### Issue: Training Slow (Many hours for 1 epoch)

**Causes:** Dataloader bottleneck, small batch size, excessive augmentation

**Solutions:**
```bash
# Increase batch size (if memory allows)
--batch-size 8

# Increase workers
--num-workers 4

# Disable expensive augmentations
--aug-crop 0
--thermal-clahe 0
```

### Issue: Overfitting (High val AP early, then drops)

**Causes:** Model too large, data too small, LR too high

**Solutions:**
```bash
# Reduce learning rate
--lr-det 1e-4

# Add early stopping
--min-patience 10

# Increase data augmentation
--aug-scale-min 0.5 --aug-scale-max 2.0
```

---

## Reproducibility

### Seed Setting

```bash
tools/train_entry.sh \
  --data-dir .data/DroneRGBT_converted \
  --seed 42 \
  --deterministic 1
```

**Note:** DDP and `deterministic=True` may reduce training speed slightly

### Exact Configuration Logging

All configurations are logged to `logs/<timestamp>/config.yaml`:

```bash
cat runs/*/config.yaml  # View all training configs
```

### Checkpoint Reproducibility

```bash
# Get the exact training command from checkpoint metadata
python3 -c "
import torch
ckpt = torch.load('path/to/checkpoint.pth')
print(ckpt['config'])  # Training hyperparameters
"
```

---

## Advanced: Custom Loss Functions

To modify loss functions, edit `Fine-tune/losses/`:

1. **Focal Loss parameters:** `Fine-tune/losses/`
2. **Loss weighting:** `Fine-tune/utils/loss_manager.py`
3. **Multi-task weighting:** `Fine-tune/utils/dm_regression_trainer.py`

Example: Custom focal loss weight:

```python
# In loss_manager.py
def compute_detection_loss(self, ...):
    if self.focal_alpha != 0.75:
        # Custom focal loss
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        ...
```

Then use:
```bash
--focal-alpha 0.9 --focal-gamma 3.0
```

---

## Next Steps

1. **Review [INFERENCE.md](INFERENCE.md)** for evaluation and post-training diagnostics
2. **Check [KNOWN_ISSUES.md](KNOWN_ISSUES.md)** to understand what not to do
3. **Read [ARCHITECTURE.md](ARCHITECTURE.md)** if you need to modify the model
4. **See [DEVELOPMENT.md](DEVELOPMENT.md)** for what worked/didn't in past phases
