# Frequently Asked Questions (FAQ)

## Getting Started

### Q: How long does training take?
**A:** On a single RTX 4090 GPU:
- Phase 6.5 resuming from checkpoint: 4-6 hours for 100 epochs
- Full training from scratch: 8-12 hours
- Multi-GPU (4×RTX 4090): ~2-3 hours for 100 epochs

Training time scales roughly inversely with number of GPUs.

---

### Q: Can I train on CPU?
**A:** Yes, but impractical:
- **Single epoch time**: ~5-10 minutes (vs 2-3 minutes on GPU)
- **100 epochs**: ~8-16 hours of continuous compute
- **Memory**: CPU usually sufficient (~16GB)

Not recommended for development due to iteration speed.

---

### Q: What GPU do I need?
**A:** Minimum specs:
- **VRAM**: 8GB for single-GPU batch size 1
- **Compute Capability**: SM 7.0+ (RTX 2060 or newer)
- **Recommended**: 12GB+ for batch size 4

Cards: RTX 3090, RTX 4090, A100, H100 all work well.

---

## Installation & Environment

### Q: I'm getting `ModuleNotFoundError: No module named 'torch'`
**A:** Your Python environment isn't activated:

```bash
# Check which environment is active
which python3

# Activate .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\activate   # Windows

# Verify
python3 -c "import torch; print(torch.__version__)"
```

---

### Q: CUDA out of memory - what should I do?
**A:** Try in order:

1. **Reduce batch size**:
   ```bash
   # Lower BATCH_SIZE in tools/train_entry.sh, then run:
   bash tools/train_entry.sh
   ```

2. **Freeze more components** (reduces memory for gradients):
   ```bash
   # Enable the freeze variables in tools/train_entry.sh, then run:
   bash tools/train_entry.sh
   ```

3. **Use a smaller crop size**:
   ```bash
   # Lower CROP_SIZE in tools/train_entry.sh, then run:
   bash tools/train_entry.sh
   ```

4. **Disable FPN** (edit `USE_FPN=0` in `tools/train_entry.sh` defaults)

---

### Q: My environment has conflicting packages. How do I start fresh?
**A:** Remove and recreate the virtual environment:

```bash
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Datasets

### Q: Where do I download DroneRGBT?
**A:** 
1. Visit: https://github.com/VisDrone/DroneRGBT
2. Follow their download instructions (may require form submission)
3. Extract to `.data/DroneRGBT/`
4. Convert: `python3 tools/convert_dronergbt.py --src .data/DroneRGBT --out .data/DroneRGBT_converted`

---

### Q: How much disk space do I need?
**A:**
- DroneRGBT (raw): ~30GB
- DroneRGBT (converted): ~15GB
- RGBT-CC (raw): ~20GB
- RGBT-CC (converted): ~10GB
- Checkpoints: 200MB-1GB per checkpoint
- **Total recommendation**: 100GB

---

### Q: My thermal images look very dark. Is this normal?
**A:** Yes, this is expected. Thermal images have different value distributions than RGB:
- Raw thermal values: Often dark (concentrated in low 8-bit range)
- Solution: CLAHE preprocessing improves contrast by adding `--thermal-clahe` to your training command

---

### Q: What's the difference between DroneRGBT and RGBT-CC?
**A:**

| Aspect | DroneRGBT | RGBT-CC |
|--------|-----------|---------|
| **Scale variation** | Low (stable drone altitude) | Extreme (5×5 to 100×100 px) |
| **Object density** | Sparse-moderate | High |
| **Thermal quality** | Better | Lower (requires CLAHE) |
| **Difficulty** | Easier (good for learning) | Harder (requires augmentation) |
| **Use case** | Initial experiments | Stress testing |

**Recommendation**: Start with DroneRGBT, then validate on RGBT-CC.

---

## Training & Validation

### Q: What does Phase 6.5 mean?
**A:** Phases 1-6 are experimental iterations:
- **Phase 1-2**: Architecture validation
- **Phase 3**: FPN exploration
- **Phase 4**: Failed experiments (documented in KNOWN_ISSUES.md)
- **Phase 6**: Systematic improvements (6.1→6.5)
- **Phase 6.5**: Current best checkpoint with balanced confidence calibration

See [docs/DEVELOPMENT.md](DEVELOPMENT.md) for full timeline.

---

### Q: Should I resume from Phase 6.5 or train from scratch?
**A:** 

**Resume from Phase 6.5** (recommended):
```bash
# Update RESUME in tools/train_entry.sh, then run:
bash tools/train_entry.sh
```
- Faster convergence (4-6 hours vs 12 hours)
- Backbone and U-Net frozen (recommended for stability)
- Good for fine-tuning on similar data

**Train from scratch**:
```bash
# Clear RESUME or point it to an empty value in tools/train_entry.sh, then run:
bash tools/train_entry.sh
```
- Required for new datasets
- Longer training time
- May need hyperparameter adjustment

---

### Q: My AP is much lower than expected. What should I check?
**A:** Verification checklist:

1. **Verify dataset conversion**:
   ```bash
   ls .data/DroneRGBT_converted/train/ | head -20
   # Should see: 1_RGB.jpg, 1_T.jpg, 1_GT.npy, ...
   ```

2. **Check model config matches documentation**:
   - `--head-conv 256` (not 128)
   - `--use-deconv` enabled (stride-4 upsample)
   - `--keypoint-mode` enabled

3. **Verify checkpoint loading**:
   ```bash
   python3 tools/verify_score_fix.py
   ```

4. **Compare with Phase 6.5 baseline**:
   - Phase 6.5 AP@8px should be ~0.56 on DroneRGBT
   - If lower, check hyperparameters

---

### Q: What's the difference between RAW, TILES, and ORIG modes?
**A:**

| Mode | What | When | Trade-off |
|------|------|------|-----------|
| **RAW** | Single-pass full image | Always (default) | Standard evaluation, most fair |
| **TILES** | Overlapping tile inference + merge | Scale variation | Better AP but slower |
| **ORIG** | Full-image with legacy thresholds | Legacy comparison | Not recommended for new work |

**Recommendation**: Use RAW mode for standard evaluation. TILES useful if you see scale sensitivity.

---

### Q: Why does my confidence distribution look weird?
**A:** Common issues:

1. **All scores near 0 or 1**: Indicates sigmoid saturation
   - Check checkpoint was loaded correctly
   - Verify `--det-pos-weight` is not too extreme

2. **Bimodal distribution** (two peaks): Possible score compression bug
   - Check: `python3 tools/verify_score_fix.py`

3. **Narrow range** (e.g., 0.4-0.6): Indicates suppression artifacts
   - Review: [docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md)

---

## Inference & Evaluation

### Q: How do I run inference on a new image?
**A:** See `Fine-tune/test_detection_vis.py` for a complete inference example.

Quick checkpoint loading reference:

```python
import torch

# Load checkpoint
ckpt = torch.load('checkpoints_phase6/phase6.5_better_bias/phase6_5_best_model_epoch_68.pth')
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt

# Use test_detection_vis.py for full inference pipeline
```

See `Fine-tune/test_detection_vis.py` for the complete example.

---

### Q: What's a "true positive" in this project?
**A:** A detection is TP if:
- Prediction center is within **8 pixels** (Euclidean distance) of ground truth point
- Prediction score ≥ threshold (default 0.1 in RAW, variable in ORIG)

See [INFERENCE.md](INFERENCE.md) for detailed metrics explanation.

---

### Q: How do I compute AP on my own predictions?
**A:** Using the AP calculation tool:

```bash
python3 tools/calculate_ap_pr_curve.py \
  --csv my_predictions.csv \
  --gt-count 54391 \
  --distance-threshold 8
```

CSV format required:
```
image_id,score,pred_x,pred_y,gt_x,gt_y
1,0.95,100,200,102,198
1,0.87,150,250,152,248
...
```

---

## Model Architecture

### Q: What's a "keypoint" in this context?
**A:** In this project, "keypoint" refers to person center points (not body joints):
- Single point per person at their center
- Used for both counting and detection
- Also called "point annotations"

---

### Q: What's the difference between U-Net and Swin backbone?
**A:**

| Component | Purpose | Input | Output |
|-----------|---------|--------|--------|
| **U-Net** | Cross-modal fusion | RGB + Thermal | "Broker" modality |
| **Swin Backbone** | Feature extraction | RGB + Thermal + Broker | 768-channel stride-8 features |

The U-Net creates a shared "broker" representation that captures complementary information from both modalities, then Swin extracts features from all three inputs.

---

### Q: Why stride-4 instead of stride-8?
**A:** 
- **Stride-8**: Each cell represents 8×8 pixels (quantization error ±4px)
- **Stride-4**: Each cell represents 4×4 pixels (quantization error ±2px)
- For small objects (16-64px typical in aerial), stride-4 provides **4-8× better precision**

See [docs/LESSONS_LEARNED.md](LESSONS_LEARNED.md) for detailed justification.

---

### Q: Can I use this model for other detection tasks?
**A:** Partially:
- **RGB-only data**: Yes, remove thermal branch
- **Different object types**: Maybe (architecture is generic)
- **Large objects**: Consider stride-8 (faster, less VRAM)
- **Different modalities** (e.g., depth, nightvision): Requires retraining

The architecture is flexible but performs best for small-object aerial detection.

---

## Benchmarking & Comparison

### Q: How does this compare to YOLO?
**A:** On DroneRGBT with AP@8px:
- **Phase 6.3 (Our method, RGBT)**: 0.7018 AP
- **YOLO26s (Thermal)**: 0.6210 AP (best single-modal)
- **YOLO26s (RGB)**: 0.3259 AP

Our method achieves **13% higher AP** through multi-modal fusion.

See [README.md](../README.md) "Baseline Model Comparison" for full details.

---

### Q: Why not use Faster R-CNN or RetinaNet?
**A:** Detectron2 baselines performed poorly on this task:
- **Faster R-CNN (Thermal)**: 0.0992 AP (7× lower)
- **RetinaNet (Thermal)**: 0.1380 AP (5× lower)

Likely because:
1. Anchor-based methods struggle with varied object scales
2. Aerial drone objects are very small
3. Our anchor-free CenterNet design works better

See [README.md](../README.md) baseline comparisons for visualizations.

---

## Troubleshooting Common Errors

### Q: `RuntimeError: CUDA device not found`
**A:** 
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

---

### Q: `KeyError: 'model_state_dict'` when loading checkpoint
**A:** Your checkpoint format is incompatible:

```python
# Try loading as direct state dict
ckpt = torch.load('checkpoint.pth')

# If it's wrapped, try:
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=False)
```

---

### Q: `ValueError: not enough values to unpack` during data loading
**A:** Dataset conversion may have failed. Verify:

```bash
# Check converted dataset structure
python3 -c "
import numpy as np
from pathlib import Path

data_dir = Path('.data/DroneRGBT_converted/train')
for f in sorted(data_dir.glob('*_GT.npy'))[:5]:
    gt = np.load(f)
    print(f'{f.name}: shape {gt.shape}, dtype {gt.dtype}')
"
```

Expected: Shape `(N, 2)` (N points), dtype `float32` or `int32`.

---

### Q: Training loss is NaN from the start
**A:** 
```bash
# 1. Lower learning rate
# Lower HEAD_LR in tools/train_entry.sh, then run:
bash tools/train_entry.sh

# 2. Verify dataset
python3 tools/quick_train_check.py
```

---

## Getting Help

### No Answer Above?

1. Check **[docs/KNOWN_ISSUES.md](KNOWN_ISSUES.md)** for documented problems
2. Review **[docs/DEVELOPMENT.md](DEVELOPMENT.md)** for phase-specific insights
3. Check **[docs/SETUP.md](SETUP.md)** for installation help
4. Examine error logs in `.tmp_posttrain_phase6/` directory

### Providing Context

When asking for help via email/issues, include:
- Error message (full traceback)
- Your command (without API keys)
- Output of: `nvidia-smi`, `python3 -c "import torch; print(torch.__version__)"`
- Dataset size and structure
- Expected vs actual result

This helps troubleshoot quickly.
