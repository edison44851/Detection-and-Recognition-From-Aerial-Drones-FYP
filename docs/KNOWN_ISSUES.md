# Known Issues & Disabled Features

**Status:** ✅ All problematic features disabled by default (Feb 2026)

This document catalogs issues discovered during development and explains why certain features are disabled.

---

## Executive Summary

### Phase 4 Incident (December 2025)

Several well-intentioned features were added to reduce false positives in Phase 4. However, detailed analysis revealed they were **fundamentally flawed and have been disabled by default**.

**Key Issues Found:**
1. **Boundary Suppression** — Harmful, compresses scores
2. **Background Suppression Loss** — Trains model to be overly conservative
3. **Adaptive Threshold** — Falls apart under different score distributions
4. **Count-Aware Filtering** — Data leakage in evaluation

**All problematic features have been reverted and documented below.**

---

## Issue 1: Boundary Suppression ❌ HARMFUL

**Location:** `Fine-tune/models/detection/center_head.py:105-120`

**What it did:**
- Suppressed detections near image boundaries (< 4 pixels from edge)
- Reduced false positives by filtering edge artifacts
- Applied as output scaling: `heatmap *= (1 - boundary_mask)`

**Why it's Harmful:**

```python
# During forward pass: scores multiplied by suppression mask
heatmap_suppressed = heatmap * (1 - boundary_mask)
# Result: All boundary pixels clamped to ~0
```

**Observed Effect:**
- **Score compression:** Entire score range compressed to 0.1-0.45 (vs normal 0.1-0.999)
- **Artificial improvement:** Model appeared better due to pruning FPs
- **Masked poor performance:** Hid generalization issues
- **Non-standard:** Not used in CenterNet, FCOS, or other anchor-free detectors

**Why It Failed:**

1. **Applied during inference** (should be post-processing only)
2. **Non-differentiable side effect** on training
3. **Unfair comparison** against methods without suppression
4. **Unrealistic** — production models can't suppress edges

**Example Impact:**

```
Phase 4 with suppression:
  Score range: 0.10 - 0.457
  Detections: Limited by suppression
  
Phase 6.5 without suppression:
  Score range: 0.10 - 0.999
  Detections: Natural distribution
```

**Current Status:** ✅ **DISABLED BY DEFAULT**
```python
self.boundary_suppress = False  # Never apply
```

---

## Issue 2: Background Suppression Loss ❌ HARMFUL

**Location:** `Fine-tune/utils/loss_manager.py:250-270`

**What it did:**
- Added auxiliary loss term: encourage model to output lower scores on background
- Aimed to improve class imbalance handling

**Why it's Harmful:**

```python
# Background suppression loss
bg_loss = -(1 - heatmap_pred[background_mask]).log().mean()
# Pushes all background pixels toward 0
```

**Problems:**

1. **Redundant** — Focal loss already focuses on hard negatives
2. **Over-regularization** — Forces background to be artificially suppressed
3. **Conflicts with class balance** — Makes model too conservative
4. **Unmotivated** — Heatmap naturally tends toward sparse patterns through focal loss

**Observed Effect:**
- Model trained to be overly conservative
- Lower recall due to suppressed backgrounds everywhere
- False sense of precision improvement

**Why Focal Loss is Better:**

Focal loss naturally handles class imbalance without explicit background suppression:
```python
# Focal loss: α(1-p)^γ log(p)
# When p is low (background): loss is large γ factor upweights hard negatives
# This is MORE principled than explicit suppression
```

**Current Status:** ✅ **DISABLED BY DEFAULT**
```python
self.use_bg_suppress = False  # Never apply
```

---

## Issue 3: Adaptive Threshold ⚠️ PROBLEMATIC

**Location:** `Fine-tune/utils/detection_eval.py:71-83`

**What it did:**
- Dynamically computed threshold per image: `threshold = mean + 1.5 × std`
- Capped at 0.3 maximum
- Applied during peak extraction

**Why it Failed:**

### Problem A: Design Assumption Violated

```python
# Designed for unbounded logit space
threshold = mean_logits + 1.5 * std_logits  # Makes sense for unbounded logits

# Applied to bounded probability space [0, 1]
threshold = mean_probs + 1.5 * std_probs    # BREAKS for bounded space
```

With bounded scores, threshold frequently exceeds 1.0:

```
Phase 2/3 Score Distribution:
  Mean: 0.75, Std: 0.21
  Adaptive threshold: 0.75 + 1.5×0.21 = 1.06 → capped at 0.3
  Result: Just uses cap, doesn't adapt
```

### Problem B: Score Compression in Phase 4

```
Phase 4 Score Distribution:
  Mean: 0.33, Std: 0.06 (compressed due to boundary suppression)
  Adaptive threshold: 0.33 + 1.5×0.06 = 0.42
  Result: Filters out 93% of detections!
```

### Problem C: Non-Interpretable

```python
# What does threshold = mean + 1.5*std even mean for probabilities?
# Nothing standard — ad-hoc and unexplained
```

**Better Alternatives:**

If you want adaptive thresholding, use principled approaches:

1. **Percentile-based:**
   ```python
   threshold = np.percentile(scores, 95)  # Keep top 5%
   ```

2. **Expected value minus margin:**
   ```python
   threshold = (mean - std) * 0.8  # Conservative, principled
   ```

3. **Fixed thresholds per phase:**
   - Phase 2/3: 0.15
   - Phase 6.x: 0.10-0.20

**Current Status:** ✅ **DISABLED BY DEFAULT**
```python
use_adaptive_threshold = False  # Never apply
```

---

## Issue 4: Count-Aware Filtering ❌ DATA LEAKAGE

**Location:** `Fine-tune/utils/evaluation_manager.py:233-238`

**What it did:**
- During evaluation, limited predictions to 1.5× ground-truth count
- Removed lowest-scoring detections beyond the cap

```python
gt_count = len(sample['points'][idx])
max_dets = max(1, int(gt_count * 1.5))
preds_px_sorted = sorted(preds_px, key=lambda x: x[2], reverse=True)[:max_dets]
```

**Why it's Harmful:**

### Problem A: Evaluation Cheating (Data Leakage)

```
Standard evaluation:
  1. Predict bboxes
  2. Match to GT
  3. Compute TP/FP
  
Count-aware evaluation:
  1. Predict bboxes
  2. **LIMIT PREDICTIONS USING GT COUNT** ← Cheating!
  3. Match to GT
  4. Compute TP/FP (biased)
```

**This uses ground-truth information that wouldn't be available in deployment!**

### Problem B: Unrealistic Improvement

```
Evidence of data leakage:
  Image 1000: GT=25, Predictions=23 (within 1.5×=37.5 cap)
  Image 1001: GT=24, Predictions=23 (within 1.5×=36 cap)
  Image 1002: GT=28, Predictions=24 (within 1.5×=42 cap)
  
Prediction counts suspiciously close to GT
→ Strong signal of count-aware filtering active
```

### Problem C: Unrealistic for Deployment

In production, you don't know how many objects are in an image! This feature makes the model artificially good.

**Why This Matters:**

Precision/recall are only meaningful if evaluated fairly:
- True precision = TP / (TP + FP)
- With count-aware filtering: artificially high (many FPs pruned)
- Misleading for model comparison

**Current Status:** ✅ **DISABLED BY DEFAULT**
```python
use_count_aware_filtering = False  # Never apply
```

**Correct Approach:**

Use fixed `max_detections` instead:
```python
--max-detections 300  # Fixed limit, no GT leakage
```

---

## Issue 5: Spatial Distribution Filtering ⚠️ PROBLEMATIC

**Location:** `Fine-tune/utils/detection_eval.py:91-106`

**What it did:**
- Used KDTree to find neighbors within 10% of image dimension
- Weighted scores by spatial density: `score *= min(neighbors/5, 2.0)`
- Assumed clustered detections = TPs, isolated = FPs

**Why it Failed:**

### Problem A: Invalid Assumption for Sparse Imagery

```
Assumption: Clustered detections = correct
Reality in aerial imagery: Often opposite!
  - Crowd of detections at fence pattern → Many FPs
  - Single person in field → Correct, but isolated
```

**Example Failure:**

```
Scenario: Single person correctly detected in open field
  - Detection score: 0.8 (high confidence)
  - Neighbors within radius: 0 (isolated)
  - Density weight: 0.2 (harsh penalty)
  - Weighted score: 0.8 × 0.2 = 0.16
  
  Then: Clustered FP at fence
  - Detection scores: 0.5 each × 20 detections
  - Neighbors: 5-10 (clustered)
  - Density weight: 1.0-2.0
  - Weighted scores: 0.5-1.0
  
Result: Correct isolated detection ranked lower than FP cluster!
```

### Problem B: Arbitrary Parameters

```python
radius = 0.10 * image_dimension  # Why 10%? Not justified
max_weight = 2.0                 # Why 2x? Pulled from thin air
density_threshold = 5            # Why 5 neighbors?
```

No theoretical justification for parameters.

### Problem C: Doesn't Generalize

Works only on datasets where objects actually cluster:
- ✅ Dense crowd images (original paper probably tested this)
- ❌ Sparse aerial imagery (our use case)
- ❌ Mixed density scenarios

**Current Status:** ✅ **DISABLED BY DEFAULT**
```python
USE_SPATIAL_FILTERING = False
```

---

## Issue 6: Filter Boundary Dets ❌ NOT IMPLEMENTED

**Location:** CLI flag `--filter-boundary-dets` in `Fine-tune/train.py:114`

**What it claims to do:**
- Filter detections near image boundaries
- Reduce edge artifacts

**Why it's Useless:**
- **Flag exists but is never checked in code**
- No actual implementation
- Likely leftover from development
- Boundary suppression (Issue #1) was the attempted fix instead

**Current Status:** ⚠️ **NOT FUNCTIONAL**
- Flag parsing exists but has no effect
- If you want boundary filtering, use:
  ```bash
  # Post-processing: filter manually
  remaining_dets = [d for d in dets if d['x'] > 10 and d['x'] < w-10]
  ```

---

## Verification

### Health Check Script

Run this to verify all issues are fixed:

```bash
python3 tools/verify_score_fix.py
```

**Expected Output:**
```
✅ boundary_suppress defaults to False
✅ use_bg_suppress defaults to False
✅ adaptive_threshold disabled
✅ count_aware_filtering disabled
✅ spatial_filtering disabled
✅ Max score reaches normal range (>0.9)
```

**If any failures:** Your version may not have the latest fixes. Check git status.

### Manual Check

```python
# Verify in code
from Fine-tune.models.detection.center_head import CenterHead

ch = CenterHead()
assert ch.boundary_suppress == False  # MUST be False
assert ch.use_bg_suppress == False    # MUST be False

# Verify during inference
heatmap_max = heatmap.max().item()
assert heatmap_max > 0.9  # Should reach near 1.0, not capped at 0.45
```

---

## What NOT to Do

### ❌ Never Enable These

```bash
# WRONG - will harm performance
--boundary-suppress 1
--use-bg-suppress 1
--adaptive-threshold 1
--count-aware-filtering 1
```

### ❌ Never Use These Approaches

1. **Real-time suppression in forward pass** — Use post-processing instead
2. **Ground-truth information in evaluation** — Unfair comparison
3. **Soft suppression masks** — Non-differentiable hacks
4. **Dataset-specific thresholds** — Generalization suffers

---

## What TO Do Instead

### For Reducing False Positives

**During Training:**
- ✅ Focal loss (already used)
- ✅ Hard negative mining (already used)
- ✅ Proper loss weighting
- ✅ Data augmentation

**During Post-Processing:**
- ✅ Higher score thresholds (e.g., 0.20 instead of 0.10)
- ✅ Stricter NMS (e.g., radius 2.0 instead of 4.0)
- ✅ Soft-NMS for overlapping detections
- ✅ Size filtering (min/max object size)
- ✅ Fixed `max_detections` limit

**Example:**
```bash
# Good post-processing
--score-threshold 0.20       # Conservative thresholding
--nms-radius 2.0             # Tight NMS
--min-box-size 4             # Filter tiny detections
--max-detections 300         # Fixed cap
```

### For Calibration & Reliability

- ✅ Confidence calibration on validation set
- ✅ Temperature scaling for better probabilities
- ✅ Ensemble predictions
- ✅ Uncertainty estimation

---

## Checkpoint Status

### Phase 6.5 (Current)

```
✅ SAFE — All problematic features disabled
✅ No boundary suppression
✅ No background suppression loss
✅ No adaptive thresholds
✅ No count-aware filtering
✅ Normal score range (0.1-0.999)
```

### Phase 6.3 (Previous Best)

```
✅ SAFE — Released before these issues
✅ Higher AP (0.59 vs 0.56)
✅ If you prefer: Use phase6.3_full_features/best_model_epoch_40.pth
```

### Phase 6.4 (Failed)

```
❌ NOT RECOMMENDED — Different issues (adaptor failure)
```

### Phase 4 and Earlier

```
⚠️ USE WITH CAUTION
❌ May have problematic features enabled
— If using: Manually verify --boundary-suppress 0, etc.
```

---

## Summary: What Changed

| Feature | Phase 4 | Phase 6+ | Reason |
|---------|---------|---------|--------|
| Boundary Suppress | Enabled | **Disabled** | Score compression harmful |
| BG Suppress Loss | Enabled | **Disabled** | Redundant with focal loss |
| Adaptive Thresh | Enabled | **Disabled** | Invalid for probability space |
| Count-Aware Filter | Enabled | **Disabled** | Data leakage in evaluation |
| Spatial Filtering | Enabled | **Disabled** | Invalid assumptions for sparse imagery |

---

## References & Further Reading

- **CenterNet Paper:** Zhou et al., "Objects as Points" (2019)
  - No suppression tricks; clean, simple design
- **Focal Loss Paper:** Lin et al., "Focal Loss for Dense Object Detection" (2017)
  - Focal loss alone sufficient for class imbalance
- **Evaluation Standards:** COCO detection benchmark
  - Uses fixed thresholds, no data leakage

---

## Need to Report an Issue?

If you discover:
- Score compression or unusual distributions
- Disabled features being re-enabled
- Count-aware filtering suspected

**Check:**
1. Verify with `tools/verify_score_fix.py`
2. Compare against Phase 6.5 checkpoint
3. Review git history: `git log --oneline | grep -i suppress`
