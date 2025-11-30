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

## Proposed Architecture Improvements (2025-11-30)

- Lightweight neck + multi-scale head outputs
	- What: Insert a tiny neck (depthwise separable convs + GroupNorm + SiLU) to produce features at strides 4 and 8, feed both to the head, and merge outputs during evaluation.
	- Files: `Fine-tune/models/detection/center_head.py`, `Fine-tune/models/detection/det_model.py`, `Fine-tune/models/counting/swin_unet.py` (expose P4/P8 features).
	- Flags: `--use-neck dwsep_gn`, `--multi-scale-head true`, `--strides 4,8`, `--merge-scales true`.
	- Potential impact: Precision increases for tiny aerial targets; slight FP growth if thresholds not tuned; small parameter/runtime overhead.
	- References: FPN (Lin et al. 2017), PANet (Liu et al. 2018), BiFPN / EfficientDet (Tan et al. 2020), MobileNet depthwise separable convs (Howard et al. 2017), SiLU/Swish (Ramachandran et al. 2017).

- Centerness branch (FCOS-like score calibration)
	- What: Add a 1-channel centerness branch; compute final score as `sigmoid(heatmap) * sigmoid(centerness)`; train centerness with BCE/focal targets from geometry.
	- Files: `center_head.py` (add `centerness_head`), trainer loss wiring, `det_model.py` decode.
	- Flags: `--use-centerness true`, `--centerness-weight 1.0`, `--score-mode heatmap_centerness`.
	- Potential impact: False positives reduced by penalizing off-center peaks; recall may dip if targets are noisy for very small boxes.
	- References: FCOS (Tian et al. 2019) centerness formulation; CenterNet (Zhou et al. 2019) heatmap peak concepts.

- Quality/IoU branch + Varifocal/QFL
	- What: Predict an IoU/quality score per detection; train with Varifocal/QFL and use quality-calibrated scores at inference (`score = heatmap * quality`).
	- Files: `center_head.py` (add `quality_head`), `Fine-tune/losses/varifocal.py` (new), trainer IoU target computation, `det_model.py` post-process.
	- Flags: `--use-quality true`, `--quality-loss varifocal`, `--quality-weight 0.5`, `--enable-quality-after-epochs 5`.
	- Potential impact: Better score calibration and fewer FPs at same recall; extra compute to form IoU targets; enable after warmup for stability.
	- References: Varifocal Loss (Zhang et al. 2021), GFocal (Li et al. 2020), PP-YOLOE (Xu et al. 2022), Task-Aligned Label Assignment (Wang et al. 2021).

- Deformable conv in head (optional)
	- What: Replace the first head conv with `DeformConv2d` to improve localization of small, off-grid centers.
	- Files: `center_head.py` (conditional block).
	- Flags: `--deform-head true`.
	- Potential impact: Localization improves; minor runtime increase; requires CUDA kernels.
	- References: Deformable ConvNets v1 (Dai et al. 2017), Deformable ConvNets v2 (Zhu et al. 2019).

- Cross-attention RGB–Thermal fusion (lightweight)
	- What: Insert a small cross-attention block between RGB and Thermal features at stride 4 before the head to improve modality alignment.
	- Files: `swin_unet.py` (fusion block), `det_model.py` (fusion routing).
	- Flags: `--fusion cross_att`, `--fusion_depth 1`.
	- Potential impact: Precision gain via reduced modality mismatch; increased complexity; keep block tiny for stability.
	- References: Transformer attention (Vaswani et al. 2017), RGB-T object detection survey (e.g., Wu et al. 2020), MFNet (Choi et al. 2018) multimodal fusion concepts.

- Post-process/NMS refinements
	- What: Per-level top-K, global NMS merge across scales, tuned radius-NMS/soft-NMS, and threshold scheduling for better precision.
	- Files: `Fine-tune/utils/detection_eval.py`, `Fine-tune/test_detection_vis.py` (options).
	- Flags: `--topk-per-level 300`, `--eval-nms-radius 2`, `--eval-soft-nms-sigma 0.5`, `--max-dets 200`.
	- Potential impact: FPs decrease with calibrated thresholds/NMS; aggressive settings may reduce recall.
	- References: Soft-NMS (Bodla et al. 2017), SAHI tiling (Ulku et al. 2022), CenterNet radius suppression (Zhou et al. 2019).