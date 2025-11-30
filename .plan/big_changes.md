## High Priority

- Dataset Normalization Mismatch: DetectionDataset uses transform = T.ToTensor() only, while the counting pipeline (dm_crowd.py) normalizes RGB/T with specific means/stds.
    - Impact: backbone (pretrained/fine-tuned on normalized inputs) produces inconsistent features when fed un-normalized images; head gets poor signals.
    - Fix: apply the same per-channel Normalize transforms as dm_crowd (separate transforms for RGB and thermal). Update DetectionDataset.__init__ to use two transforms or accept rgb_transform, t_transform.

- Heatmap Loss Dominated by Background: trainer uses self.heatmap_loss = nn.BCELoss(reduction='sum'). Summing over all pixels lets negatives dominate loss and push predictions to zero.
    - Impact: head learns trivial zero heatmaps, precision & recall remain poor.
    Fix: use reduction='mean', or weighted positive loss (BCEWithLogitsLoss(pos_weight=...) or element-wise weighting on positives), or focal loss. Expose --det-pos-weight hyperparameter.

- Very Small Learning Rate for Head: default --lr is 1e-5 and many run with --freeze-backbone. For a randomly-initialized head this LR is too small.
    - Impact: negligible head updates; slow or no improvement.
    Fix: use head-only lr 1e-3 → 1e-4. Implement optimizer param-groups or pass --lr 1e-3 for head-only runs.

- Input / Label Distribution Mismatch from Converter: convert_dronergbt.py parses XML points and writes .npy with (x,y). If the original dataset uses different coordinate order or cropping, coordinates can be off by one or swapped.
    - Impact: GT centers don't align with image pixels → low matching AP.
    - Fix: validate a handful of .npy files visually (overlay points on images) and confirm (x,y) convention matches training code (the dataset assumes x=column, y=row).

---

## Medium Priority

- Output Stride / Downsample Ratio Confusion: datasets and trainer use output_stride / downsample_ratio in several places; inconsistent defaults (4 vs 8) exist. Trainer passes args.downsample_ratio into dataset but other places may assume different values.
    - Impact: detections mapped to wrong pixel coordinates or AP matching threshold inappropriate.
    Fix: centralize one CLI flag (we added --output-stride alias); ensure DetectionDataset, trainer and inference use the same value.

- AP Matching Threshold Too Strict for Small Aerial Objects: compute_ap uses dist_thresh=4.0 px. For small or downsampled outputs, 4 px may be too strict.
    - Impact: correct predictions counted as false negatives.
    - Fix: raise threshold to 8–16 px or evaluate AP at multiple thresholds; ensure threshold interprets pixels (not output-grid coords).

- Offset & Size Targets Quantized to Floor Cell: DetectionDataset stores offsets and sizes at floor(fx),floor(fy). If peak extraction lands on adjacent cell, offset won't correct large quantization error.
    - Impact: localization error for detections that peak in neighboring cells.
    Fix: use sub-pixel-aware target formation (e.g., place size/offset at the nearest grid cell to the continuous center or encode \Delta relative to center via soft assignment).

- Head Input Adaptor / Capacity Mismatch: Swin_BM_RGBT.det_adaptor is Identity by default. If backbone fused features have different channel range/scale the head may underperform.
    - Impact: head receives features on unexpected scale or distribution.
    - Fix: add a small det_adaptor (1x1 conv + BN + ReLU) to map channels and scale; optionally increase CenterHead capacity.

---

## Low / Implementation Issues

- Checkpoint Loading + Freezing Order: trainer _load_checkpoint loads checkpoint before _attach_detection_head — good. But freezing occurs after DDP wrapping which may fail to freeze attached head if not careful.
    - Impact: accidental freezing/unfreezing of the head or missing parameters in optimizer.
    Fix: verify freeze order and ensure optimizer is created after freezing (trainer does this already); confirm find_unused_parameters in DDP is set appropriately when freezing.

- Heatmap Target Shape & Clipping: heatmap is clipped to [0,1] after summing Gaussians; for very dense clusters peaks might be affected.
    - Impact: reduced peak contrast for overlapping objects.
    Fix: use normalized per-object Gaussians (max 1 per object) or cap at 1 but ensure sigma appropriate for output stride.

- Evaluation & Logging Blind Spots: limited diagnostics (no per-class/per-image loss breakdown, no head grad norms).
    - Impact: harder to debug why head fails to learn.
    - Fix: add logging of det_loss, pos/neg BCE breakdown, and gradient norms for head/adaptor.

---

## Data / Label Quality Checks (must-run)

- Visual check overlay: open several samples from DroneRGBT_converted and overlay *_GT.npy points on *_RGB.jpg and *_T.jpg to confirm coordinates and scale.
- Heatmap sanity: for single-sample run DetectionDataset and visualize heatmap to verify peaks align with input points.
- Coordinate order: ensure converter's (x,y) order corresponds to dataset assumption (x = column, y = row).