# Lessons Learned: Design Decisions & Key Insights

**Final Status:** Project complete, all findings documented (Feb 2026)

---

## Part 1: Detection Architecture

### Lesson 1: Spatial Resolution is Non-Negotiable

**Finding:** Stride (output downsampling) is the single most impactful factor

| Metric | Stride-8 | Stride-4 (2× finer) | Improvement |
|--------|----------|-------------------|-------------|
| Quantization error | ±4 pixels | ±2 pixels | 2× accuracy |
| Small object (16×16px) | 1-2 cells | 4-8 cells | 4-8× precision |
| Heatmap resolution | 100×100 | 200×200 | 4× detail |
| **Recall** | ~0.5% | **61%** | **122× improvement** |

**Why:** Aerial objects are small (16-64 pixels typical). At stride-8, multiple objects collapse into single heatmap cell, making them indistinguishable.

**Design Decision:** ✅ Always use stride-4 output for small object detection. Deconv upsampling is worth the cost.

**Generalization:** This applies to any small-object detection task:
- Traffic signs, small animals, faces
- Pedestrians in wide-angle drone footage
- QR codes, license plates

---

### Lesson 2: Non-Linearity in Heads is Essential

**Finding:** Single-layer linear heads cannot capture object detection patterns

| Design | Layers | Activation | Feature Expression | Performance |
|--------|--------|-----------|-------------------|-------------|
| Linear bottleneck | 1 | None | Rank-1 constraint | TP=11-280, AP~0.01% |
| ReLU bottleneck | 2 | ReLU ↔ gates feature flow | Rank-unlimited | TP=1211, AP~48% |

**Why:** Object detection requires non-linear feature transformations:
- Learning edge patterns (non-linear boundaries)
- Object-specific attributes (not linearly separable from background)
- Multi-modal interaction patterns (RGB + Thermal fusion)

**Design Decision:** ✅ Always use 2-layer detection heads with ReLU bottleneck

```python
# Good: Non-linear feature transformation
h = conv_3x3(features, 256)  # Large receptive field
h = ReLU(h)                  # Non-linearity
output = conv_1x1(h)         # Fine-grained prediction

# Bad: Linear transform only
output = conv_1x1(features)  # No non-linearity
```

**Generalization:** Applies beyond detection:
- Segmentation, pose estimation, any dense prediction task
- 2-layer minimum; deeper helps but with diminishing returns

---

### Lesson 3: Heatmap Bias Initialization Matters Enormously

**Finding:** Initial prediction bias directly affects focal loss convergence

| Bias | sigmoid(bias) | Interpretation | Focal Loss Fit | Training Stability | AP Achieved |
|------|---------------|--------------------------------------------------------|---|---|---|
| +2.0 | 0.88 | Most pixels "object" | ❌ Poor | ❌ Unstable | ~0.01% |
| 0.0 | 0.50 | Ambiguous | ❌ Poor | ❌ Fights focal loss | ~0.01% |
| **-2.0** | **0.12** | **Most pixels "background"** | ✅ **Excellent** | ✅ **Stable** | **0.56** |
| -4.6 | 0.01 | Extreme background prior | ⚠️ Conservative | ⚠️ More FNs | 0.45 |

**Why:** Focal loss assumes most pixels are background

$$L_{focal} = \begin{cases}
-\alpha(1-p)^\gamma \log(p) & \text{if } y=1 \text{ (positive)} \\
-(1-\alpha)p^\gamma \log(1-p) & \text{if } y=0 \text{ (negative)}
\end{cases}$$

With $p_0 = 0.12$ (bias = -2.0):
- Negative pixels start easily satisfied (loss small)
- Positive pixels start unsatisfied (loss large)
- Natural gradient flow toward sparse peaks

**Design Decision:** ✅ Always initialize heatmap bias to $-\ln(1/0.12 - 1) \approx -2.0$

**Mathematical Insight:**
```python
# Optimal initialization for focal loss
sigmoid_target = 0.12  # Start with 12% as "object"
bias = -np.log(1/sigmoid_target - 1)
```

**Generalization:** Principle applies broadly:
- Any classification with class imbalance
- Initialization should match expected prior
- For 99% negatives: bias ≈ -4.6
- For 90% negatives: bias ≈ -2.2

---

### Lesson 4: NMS is Mandatory, Not Optional

**Finding:** Soft heatmap peaks cluster without suppression, creating massive FP rate

| Configuration | Detections/Image | TP | FP | FP Rate |
|---|---|---|---|---|
| Without NMS | 75-79 | 11-280 | 95%+ of predictions | ~96% FP |
| With NMS (k=3) | 180-200 | 1211 | ~40% of predictions | ~40% FP |

**Why:** Heatmap smoothness creates soft peaks around objects:
- Gaussian blur when rendering targets
- ReLU in head creates smooth gradients
- Multiple nearby pixels above threshold

**Design Decision:** ✅ Always apply max-pooling NMS during inference

```python
# CenterNet-style NMS
heatmap_max = max_pool2d(heatmap, kernel=3, stride=1, padding=1)
peaks = heatmap[heatmap == heatmap_max]  # Local maxima only
```

**Generalization:** Essential for any dense prediction:
- Semantic segmentation (suppress adjacent pixels)
- Instance segmentation (NMS between instances)
- Keypoint detection (suppress soft peaks)

---

## Part 2: Multi-Modal Fusion

### Lesson 5: Don't Discard Any Modality

**Finding:** Multi-modal fusion significantly outperforms single modalities

```
Single modality (f_B broker only):
  AP: 0.50
  Precision: 0.55
  Recall: 0.54
  
Multi-modality (f_R + f_T + f_B):
  AP: 0.59    (+18% improvement)
  Precision: 0.77
  Recall: 0.58
  
Improvement breakdown:
  - Precision +28-32%  (fewer confusions)
  - FP reduction: 38.5%
```

**Why:** Each modality captures unique information:
- **RGB:** Color, texture, fine details
- **Thermal:** Heat signature, material properties, night visibility
- **Broker:** Cross-modal synthesis (learned fusion)

Together: Complementary coverage of failure modes

**Design Decision:** ✅ Always use all available modalities

```python
# Good: Leverage all information
features = f_rgb + f_thermal + f_broker

# Bad: Throw away information
features = f_broker_only  # 18% AP loss
```

**Challenge:** Over-reliance on single modality during degradation
- If thermal fails: Model can't compensate
- **Counter:** Modality dropout during training (not implemented but valuable)

**Generalization:**
- Multi-task learning benefits from multiple inputs
- Fuse early (features) rather than late (detections)
- Symmetric treatment of modalities (equal weight)

---

### Lesson 6: Simple Fusion is Better Than Complex

**Finding:** Element-wise sum outperforms learned spatial transformations

| Method | Implementation | Parameters | Performance | Stability |
|--------|---|---|---|---|
| Element-wise sum | `r + t + b` | 0 | Good (AP 0.59) | ✅ Stable |
| Attention fusion | Conv chains | ~500K | ❌ -97% AP | Catastrophic |
| Learnable scaling | Per-channel weights | 3 | Similar | ✅ Stable |

**Why:** Simple fusion works because:
1. Features already well-aligned (same backbone)
2. Order-invariant sum is forgiving
3. Over-parameterization → overfitting on small dataset

**Design Decision:** ✅ Stick with element-wise sum for fusion

```python
# Good: Simple and effective
fused = f_rgb + f_thermal + f_broker

# Risky: Learned spatial transformations without careful regularization
fused = learned_fusion_network(f_rgb, f_thermal, f_broker)
```

**Lesson from Phase 6.4:** Better Adaptor (3×3+3×3+1×1 chain) failed catastrophically:
- 6.6M parameters for adaptor alone
- Over-parameterization without regularization
- Overfitting on training, collapse at inference

**Generalization:** Occam's Razor in deep learning:
- Simpler designs generalize better
- Add complexity only when proven necessary
- Validate on held-out test set before claiming improvement

---

## Part 3: Training & Optimization

### Lesson 7: Gradient Clipping and Numerical Stability Matter

**Finding:** Without proper numerical checks, training silently fails

| Configuration | Loss Pattern | Detectable | Consequence |
|---|---|---|---|
| No clipping, no checks | NaN/Inf appears | ❌ Silent | Complete failure, maskable by bad eval |
| Gradient clipping only | Stable loss | ⚠️ Manual check | May still diverge, hard to debug |
| Clipping + NaN detection | Stable loss, logged | ✅ Immediate | Safe, can skip corrupted batches |
| Clipping + LR schedule + early-stop | Smooth convergence | ✅ Best | Reliable, reproducible training |

**Why:** Gradient explosions happen subtly:
- Focal loss can blow up on hard examples
- Optimizer can produce invalid states
- Without detection: training looks "fine" but model fails

**Design Decision:** ✅ Always use:
1. Gradient clipping (`max_norm=0.5-1.0`)
2. NaN/Inf detection and batch skipping
3. Learning rate scheduling (cosine annealing)
4. Early stopping on validation metric

```python
# Good practice
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
if torch.isnan(loss) or torch.isinf(loss):
    continue  # Skip corrupted batch
```

**Generalization:** Principles for any loss function:
- Detection-specific (focal loss) or general
- Always include numerical guards
- Monitor gradient norms during training

---

### Lesson 8: Loss Weighting is Delicate

**Finding:** Multi-task loss balance directly affects performance

| Scenario | Det Loss Weight | Counting Loss Weight | Result |
|---|---|---|---|
| Detection-only (freeze counting) | 1.0 | 0.0 | ✅ Pure detection focus |
| Balanced multi-task | 1.0 | 1.0 | ⚠️ Task competition |
| Count-aware | 0.1 | 1.0 | ❌ Detection ignored |
| Heavy detection | 10.0 | 0.1 | ❌ Counting degraded |

**Why:** Tasks can interfere:
- Counting wants dense features
- Detection wants sparse peaks
- Competing objectives on same features

**Design Decision:** ✅ For detection-only training:
- Freeze counting branch entirely
- Use pure detection loss
- Validate counting metrics unchanged

```python
# Good: Isolate detection training
--freeze-counter
--det-loss-weight 1.0
# No need for --wot, --wtv, --wrd flags
```

**If joint training needed:**
- Careful hyperparameter tuning required
- Uncertainty-weighted losses (Kendall et al.) help
- Monitor both tasks' metrics separately

**Generalization:** Multi-task learning in general:
- Isolate easier subtasks
- Use uncertainty weighting for automatic balancing
- Never rely on fixed weights across datasets

---

## Part 4: Evaluation & Flawed Approaches

### Lesson 9: Never Use Ground Truth Information During Evaluation

**Finding:** Count-aware filtering (using GT during eval) invalidates comparisons

```
Standard Evaluation (Fair):
  Model predicts → Match to GT → Compute metrics
  Metrics reflect true performance
  
Count-Aware Evaluation (Unfair):
  Model predicts → Limit using GT count → Match to GT → Compute metrics
  Metrics artificially high (FPs pruned)
  Model appears better than it is
```

**Why:** Invalidates fair comparison
- Cannot compare fairly against methods without access to GT
- Misleads about actual deployment performance
- Standard in academic papers is: no GT info during evaluation

**Design Decision:** ✅ Always use fixed limits, never GT-dependent ones

```python
# Good: Fixed, GT-independent
--max-detections 300

# Bad: Uses GT information
max_dets = int(gt_count * 1.5)  # Data leakage!
```

**Generalization:** Extends to all evaluation:
- Hyperparameter tuning: validate on separate hold-out set
- Threshold selection: use validation set, not test set
- Confidence calibration: never use test set for calibration

---

### Lesson 10: Adaptive Thresholds are Risky

**Finding:** Adapting thresholds without sound theory breaks reproducibility

| Threshold Method | Principled | Reproducible | Generalizable | Recommendable |
|---|---|---|---|---|
| Fixed (e.g., 0.15) | ✅ | ✅ | ✅ | ✅ YES |
| Adaptive (mean+std) | ❌ | ⚠️ | ❌ | ❌ NO |
| Percentile-based | ⚠️ | ✅ | ⚠️ | ⚠️ Maybe |
| Learned (Platt scaling) | ✅ | ✅ | ✅ | ✅ YES |

**Why:** Adaptive thresholds fail:
- Mean+std assumes unbounded distribution
- Different distributions have different relationships
- No theoretical justification

**Design Decision:** ✅ Use one of:
1. Fixed thresholds per model (simplest)
2. Percentile-based (if must adapt)
3. Learned calibration from validation set (best)

```python
# Good: Fixed threshold
score_threshold = 0.15

# Acceptable: Percentile-based
score_threshold = np.percentile(val_scores, 95)

# Bad: mean + std (no theory)
score_threshold = np.mean(scores) + 1.5 * np.std(scores)
```

**Generalization:** Threshold selection broadly:
- Medical diagnosis: use sensitivity/specificity curves (not adaptive)
- Anomaly detection: ROC curves on validation
- Always validate on separate data

---

## Part 5: What to Avoid

### Anti-Pattern 1: Soft Suppression During Forward Pass

**Bad:**
```python
# Suppress boundaries during inference
heatmap *= (1 - boundary_mask)
```

**Why Harmful:**
- Non-differentiable, stops learning
- Hides poor model behavior
- Hard to reproduce what network learned
- Not standard in literature

**Good Alternative:**
```python
# Filter as post-processing
remaining = [d for d in dets if is_interior(d, image)]
```

---

### Anti-Pattern 2: Too Many Hyperparameters

**Bad:**
```bash
--boundary-suppress 1 \
--bg-suppress-weight 0.5 \
--adaptive-threshold 1 \
--spatial-filtering 1 \
--count-aware-filtering 1 \
... (10+ more flags)
```

**Why Harmful:**
- Overfitting to dataset
- Hard to debug which helps
- Irreproducible
- Each flag increases risk

**Good:**
```bash
--score-threshold 0.15 \
--nms-radius 2.0 \
--max-detections 300
```

Just 3 hyperparameters, highly interpretable.

---

### Anti-Pattern 3: Celebrating Training Metrics Without Test Validation

**Bad:** Phase 6.4 showed:
- Training validation AP: 0.605 ✅
- Inference AP: 0.015 ❌
- **97% discrepancy!**

**Why:** Overfitting can hide in training validation:
- Complex model + small dataset = easy to memorize training distribution
- Must test on truly held-out data

**Good:**
1. Train with validation monitoring
2. Final test on completely held-out set
3. If large discrepancy: investigate overfitting

---

## Part 6: Principles for Future Work

### Principle 1: Simplicity Beats Complexity

> "Make it simple, then add complexity only when proven necessary"

Evidence:
- ✅ Element-wise sum fusion (0 parameters) beats learned fusion
- ✅ 2-layer heads (simple) beat spatial conv chains
- ✅ Fixed thresholds beat adaptive algorithms
- ❌ Phase 6.4's complex adaptor failed catastrophically

**Application:** Before adding a feature, ask:
1. What problem does it solve?
2. Is there a simpler alternative?
3. Can we validate on test set first?

---

### Principle 2: Theory Guides Implementation

> "Understand why before implementing"

Evidence:
- ✅ Bias = -2.0 motivated by focal loss theory
- ✅ Stride-4 motivated by small object coverage
- ✅ NMS motivated by heatmap smoothness
- ❌ Adaptive threshold had NO theoretical foundation
- ❌ Count-aware filtering ignored evaluation principles

**Application:** For each design choice:
1. Understand the problem theoretically
2. Design based on theory
3. Validate empirically
4. Document the chain of reasoning

---

### Principle 3: Validate on Test Set, Not Training Set

> "Training performance is not deployment performance"

Evidence:
- Phase 6.4: Training AP 0.605 → Test AP 0.015
- Phase 6.5: Chose slightly lower AP (0.56) for better calibration

**Application:**
- Always reserve truly held-out test data
- Pick best model based on test metrics
- Never tune hyperparameters on test set
- Use validation set for early stopping

---

### Principle 4: Reproducibility First

> "If you can't reproduce it, it doesn't exist"

What we did right:
- ✅ Fixed random seeds
- ✅ Documented every phase with checkpoint
- ✅ Logged all hyperparameters
- ✅ Archived failed experiments

What to do:
- Save all training configurations
- Store checkpoints with git hashes
- Document what worked/failed
- Enable others to reproduce

---

## Summary: Key Takeaways

### Architecture
1. ✅ Stride-4 output is critical for small objects
2. ✅ 2-layer ReLU heads essential for non-linearity
3. ✅ Heatmap bias initialization (-2.0) matters enormously
4. ✅ NMS mandatory for sparse detection
5. ✅ Multi-modal fusion provides significant gains
6. ✅ Simple fusion (sum) beats complex learned fusion

### Training
7. ✅ Gradient clipping + NaN detection + LR schedule
8. ✅ Loss weights between tasks are delicate
9. ❌ Never use GT information in evaluation
10. ❌ Adaptive thresholds without theory are risky

### Philosophy
11. ✅ Simplicity > Complexity
12. ✅ Theory > Empiricism alone
13. ✅ Test set > Training set
14. ✅ Reproducibility > Performance

---

## Recommended Reading

For anyone continuing or learning from this project:

1. **CenterNet paper** (Zhou et al., ICCV 2019)
   - Best-in-class architecture principles
   - No suppression tricks; elegant design

2. **Focal Loss paper** (Lin et al., ICCV 2017)
   - Understanding why bias initialization matters
   - Class imbalance without ad-hoc tricks

3. **Swin Transformer paper** (Liu et al., ICCV 2021)
   - Hierarchical attention basis of our backbone
   - Multi-scale feature extraction

4. **Mixed Precision Training** (NVIDIA documentation)
   - Numerical stability techniques
   - Gradient clipping strategies

---

## Final Reflection

The journey from baseline (AP 0.001%) to Phase 6.5 (AP 0.56%) taught us:

- **Big wins come from fundamentals:** Stride, non-linearity, proper initialization
- **Small tweaks rarely help unless grounded in theory**
- **Simplicity is a feature, not a limitation**
- **Test on truly held-out data, always**

The project demonstrates that systematic development with clear understanding of principles yields results far better than ad-hoc feature addition.

---

**Final Status:** Project complete, all lessons documented, code locked (Feb 20, 2026)
