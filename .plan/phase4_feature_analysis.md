# Phase 4 Feature Analysis: Evaluation of Teammate Modifications

**Date:** February 6, 2026  
**Analysis:** Review of Phase 4 detection features for usefulness and correctness  
**Status:** 4 features reviewed, 3 disabled by default, 1 unimplemented

---

## Executive Summary

Phase 4 introduced 4 features aimed at reducing false positives:
1. **Boundary Suppression** - ❌ HARMFUL (causes 55% score compression)
2. **Background Suppression Loss** - ❌ HARMFUL (trains model to be overly conservative)
3. **Adaptive Threshold** - ⚠️ PROBLEMATIC (filters 93% of Phase 4 detections, useless for Phase 2/3)
4. **Count-Aware Filtering** - ❌ HARMFUL (data leakage during evaluation)

**Bonus findings:**
- **Spatial Distribution Filtering** - ⚠️ QUESTIONABLE (assumptions don't hold for sparse imagery)
- **Filter Boundary Dets** - ❌ NOT IMPLEMENTED (flag exists but no code uses it)

**All problematic features have been disabled by default.**

---

## Detailed Analysis

### 1. Adaptive Threshold ⚠️ PROBLEMATIC

**Location:** `Fine-tune/utils/detection_eval.py:71-83`, `evaluation_manager.py:208-213`

**What it does:**
- Calculates dynamic threshold per image: `threshold = mean + 1.5*std`
- Caps at 0.3 maximum
- Applied during heatmap peak extraction

**Problems:**

#### A. Catastrophic for Phase 4 (Compressed Scores)
```
Phase 4 Score Distribution:
  Mean: 0.33, Std: 0.06
  Adaptive threshold: 0.33 + 1.5*0.06 = 0.42
  Result: Filters out 93% of detections!
```

#### B. Useless for Phase 2/3 (Normal Scores)
```
Phase 2/3 Score Distribution:
  Mean: 0.75, Std: 0.21
  Adaptive threshold: 0.75 + 1.5*0.21 = 1.06 → capped at 0.3
  Result: Just uses cap (0.3), doesn't adapt
```

**Why it fails:**
- Designed for **logit space** (unbounded) but applied in **probability space** (0-1)
- Assumes score distribution is consistent across images, but Phase 4 compression breaks this
- For normal distributions, threshold exceeds 1.0 immediately, making it just a cap

**Verdict:** ❌ **DISABLED BY DEFAULT**
- Harmful for compressed scores (over-filters)
- Useless for normal scores (just uses cap)
- If you want adaptive thresholds, use percentile-based (e.g., 95th percentile) instead of mean+std

---

### 2. Count-Aware Filtering ❌ DATA LEAKAGE

**Location:** `Fine-tune/utils/evaluation_manager.py:233-238`

**What it does:**
```python
# During EVALUATION, limits detections to 1.5x ground truth count
gt_count = len(sample['points'][idx])
max_dets = max(1, int(gt_count * 1.5))
preds_px = sorted(preds_px, key=lambda x: x[2], reverse=True)[:max_dets]
```

**Problems:**

#### A. Uses Ground Truth During Evaluation
- **This is data leakage!** Evaluation should not use any ground truth information except for scoring matches
- Artificially inflates precision by preventing model from making too many false positives
- Masks true model behavior - we don't know actual FP rate

#### B. Not Realistic for Deployment
- In real-world deployment, you don't know how many objects are in the image
- This feature makes the model look better than it actually is

**Evidence from results:**
```
Image 1000: GT=25, Preds=23 (within 1.5x=37.5 cap)
Image 1001: GT=24, Preds=23 (within 1.5x=36 cap)
Image 1002: GT=28, Preds=24 (within 1.5x=42 cap)
```
Predictions are suspiciously close to GT counts, suggesting the filter is active.

**Verdict:** ❌ **DISABLED BY DEFAULT**
- This is evaluation cheating (data leakage)
- Use fixed `max_detections` (e.g., 300) instead
- If you need count estimation, train a separate count head

---

### 3. Spatial Distribution Filtering ⚠️ QUESTIONABLE

**Location:** `Fine-tune/utils/detection_eval.py:91-106`

**What it does:**
- Uses KDTree to count neighbors within 10% of image dimension
- Weights scores by density: `weighted_score = score * min(neighbors/5, 2.0)`
- Assumes clustered detections = true positives, isolated = false positives

**Problems:**

#### A. Invalid Assumption for Sparse Imagery
- **Drone images have sparse, distributed objects** (people in open areas)
- Isolated detections are often correct (single person in field)
- Clustered detections can be false positives (repetitive patterns)

#### B. Arbitrary Parameters
- 10% radius - why not 5% or 20%?
- Density weight capped at 2x - why?
- No theoretical justification

**Example failure case:**
```
Scenario: Single person in open field (correct detection, score=0.8)
Problem: No neighbors within radius → density weight = 0.2
Result: Weighted score = 0.8 * 0.2 = 0.16 (ranked lower than clustered FPs)
```

**Verdict:** ⚠️ **DISABLED BY DEFAULT**
- Assumption doesn't hold for sparse aerial imagery
- Might work for dense crowds, but not general-purpose
- Hardcoded as `USE_SPATIAL_FILTERING = False` in code

---

### 4. Filter Boundary Dets ❌ NOT IMPLEMENTED

**Location:** CLI flag in `Fine-tune/train.py:114`

**What it does:**
- Flag exists: `--filter-boundary-dets`
- **NO CODE IMPLEMENTS IT** - searched entire codebase, flag is never checked

**Verdict:** ❌ **NOT IMPLEMENTED**
- This is a ghost feature - exists in CLI but does nothing
- If you want boundary filtering, use higher NMS radius instead

---

## Summary of Changes Applied

### Code Changes
1. ✅ Disabled `boundary_suppress` by default (was causing score compression)
2. ✅ Disabled `use_bg_suppress` by default (was training model to be conservative)
3. ✅ Disabled `adaptive_threshold` by default (filters 93% of Phase 4, useless for Phase 2/3)
4. ✅ Disabled `count_aware_filtering` by default (data leakage during evaluation)
5. ✅ Disabled spatial distribution filtering (invalid assumptions for sparse imagery)
6. ✅ Added warnings to all CLI flags explaining issues

### Documentation Updates
- Updated README with bugfix notice
- Added verification script: `tools/verify_score_fix.py`
- Created this analysis document

---

## Recommendations

### Immediate Actions
1. **Use Phase 3 checkpoint** (`1209-205427_keypoint_mode_fpn`) - clean implementation without these issues
2. **Retrain Phase 4** without boundary/background suppression to recover normal scores
3. **Re-evaluate all phases** without count-aware filtering to see true model performance

### For False Positive Reduction
Instead of these features, use proven approaches:

#### During Training:
- ✅ Focal loss with proper alpha/gamma tuning
- ✅ Hard negative mining (already implemented)
- ✅ Higher learning rate for detection head
- ⚠️ DO NOT apply score suppression during forward pass

#### During Inference:
- ✅ Higher score thresholds (e.g., 0.30 instead of 0.15)
- ✅ Stricter NMS (radius=4 or 6 instead of 2)
- ✅ Soft-NMS for overlapping detections
- ✅ Post-processing filters (e.g., minimum detection size)
- ⚠️ DO NOT use ground truth counts

#### For Deployment:
- ✅ Ensemble multiple thresholds and NMS configs
- ✅ Calibrate scores using validation set
- ✅ Use confidence histograms to set thresholds
- ⚠️ DO NOT rely on adaptive/learned filters that use GT

---

## Testing & Verification

Run the verification script:
```bash
python3 tools/verify_score_fix.py
```

Expected output:
```
✅ CenterHead.boundary_suppress defaults to False
✅ DetectionHeadWrapper.head.boundary_suppress defaults to False
✅ Max score reaches normal range (>0.9)
```

---

## Phase Comparison After Fixes

| Phase | Config | Score Range | Issues |
|-------|--------|-------------|--------|
| Phase 2 | Deeper CenterHead | 0.10 - 0.999 (mean 0.75) | ✅ Clean |
| Phase 3 | + FPN + Keypoint | 0.10 - 0.999 (mean 0.75) | ✅ Clean |
| Phase 4 (original) | + Teammate features | 0.10 - 0.457 (mean 0.33) | ❌ Compressed |
| **Phase 4 (fixed)** | **Without suppression** | **Expected: 0.10 - 0.999** | **✅ Should be clean** |

**Note:** Phase 4 checkpoint still has compressed weights from training with suppression. Need to retrain to fully recover.

---

## Conclusion

The Phase 4 teammate modifications were **well-intentioned but fundamentally flawed**:
- Boundary/background suppression applied incorrectly (during forward pass)
- Adaptive threshold assumes wrong score distribution
- Count-aware filtering is evaluation cheating
- Spatial filtering has invalid assumptions

**All have been disabled by default. Use Phase 3 checkpoint or retrain Phase 4.**

For legitimate FP reduction, use proven post-processing techniques (higher thresholds, stricter NMS) instead of these experimental features.
