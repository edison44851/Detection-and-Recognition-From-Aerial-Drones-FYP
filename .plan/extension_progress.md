# Extension Progress — 2025-11-25

**Summary**
- **Scope:** Extended Free-Lunch (crowd counting) with a center-based detection branch, detection dataset targets, detection evaluation, and trainer improvements. Recently completed major code refactoring for maintainability and simplified task separation.
- **Status:** Detection branch fully implemented and tested; trainer refactored into modular components; multi-task support removed for simplicity. All changes validated with multi-GPU distributed training. Core changes live under the `Fine-tune/` subtree.

**Recent changes (2025-11-25)**
- **Major refactoring:** Reorganized `dm_regression_trainer.py` from monolithic 1000+ line code into clean, focused helper methods (~20 new methods) for better maintainability:
  - Setup phase: `_setup_distributed()`, `_setup_datasets()`, `_create_model()`, `_load_checkpoint()`, `_freeze_model_components()`, `_create_optimizer()`, etc.
  - Training phase: `_prepare_batch()`, `_compute_counting_losses()`, `_compute_detection_losses()`
  - Evaluation phase: `_evaluate_counting_sample()`, `_evaluate_detection_sample()`, `_compute_game_metrics()`, `_should_save_best_model()`
- **Simplified task model:** Removed "multi" task type to eliminate complexity:
  - **Detection task:** Uses ONLY detection losses (heatmap BCE, size/offset L1). No counting losses computed.
  - **Counting task:** Uses ONLY counting losses (OT, count, TV, RD). No detection losses computed.
  - Forward pass simplified from 3 branches to 2 clean branches.
- **Removed complexity:**
  - Eliminated `'multi'` task type (previously mixed counting + detection)
  - Removed `'combined'` saving strategy (α·AP - β·GAME0)
  - Kept only `'det'` (save by AP) and `'count'` (save by GAME0) strategies
- **Verification:** Multi-GPU (4 GPUs) distributed training tested successfully with refactored code:
  - Checkpoint loading: 0 missing/unexpected keys ✓
  - Freeze flags working correctly (16 trainable tensors for detection head only) ✓
  - Detection AP improving over epochs (0.0017 → 0.0071, 4x improvement) ✓
  - Detection loss decreasing (31227 → 12657, 59% reduction) ✓
  - Counting GAME0 stable across epochs (~549, frozen weights working) ✓

# Extension Progress — 2025-11-23

**Summary**
- **Scope:** Extended Free-Lunch (crowd counting) with a center-based detection branch, detection dataset targets, detection evaluation, trainer/CLI improvements for multi-task workflows, and tooling for visualization and distributed runs.
- **Status:** Detection branch implemented and integrated; smoke tests and short GPU dry-runs completed. Core changes live under the `Fine-tune/` subtree. Several trainer fixes and CLI flags were added to make multi-task experiments robust.

**What I implemented (high level)**
- **Detection model & head:** `Fine-tune/models/detection/center_head.py` and `Fine-tune/models/detection/det_model.py` (wrapper combining the existing backbone with a center head).
- **Dataset & targets:** `Fine-tune/datasets/dm_detection.py` — returns `rgb`, `t`, `heatmap`, `size`, `offset`, and `points`. Gaussian heatmap targets implemented and tested.
- **Trainer integration:** `Fine-tune/utils/dm_regression_trainer.py` — supports `--task detection|multi`, detection losses (BCE + L1), separated loss meters (count / det / total), detection AP evaluation, permssive checkpoint remapping, and robust DDP behavior.
- **CLI:** `Fine-tune/train.py` — flags added/extended: `--task`, `--freeze-backbone`, `--freeze-counter`, `--freeze-unet`, `--unfreeze-epoch`, `--det-weight`, `--det-patience`, `--save-by` (save strategy options available).
- **Evaluation & visualization:** `Fine-tune/utils/detection_eval.py` (peak extraction + AP), plus `Fine-tune/test_detection_vis.py` (visualize predicted centers on RGB+Thermal side-by-side).
- **Tools:** `tools/run_torchrun_train_detector.sh` (torchrun helper) and `tools/quick_train_check.py` (small local dry-run harness) added to make quick experiments reproducible.

**Files changed / added (selected)**
- Modified: `Fine-tune/models/counting/swin_unet.py` (exposed backbone features via `get_backbone_features`).
- Added: `Fine-tune/models/detection/center_head.py`, `Fine-tune/models/detection/det_model.py`.
- Modified: `Fine-tune/datasets/dm_detection.py` (Gaussian heatmaps, size/offset targets).
- Modified: `Fine-tune/utils/dm_regression_trainer.py` (loss separation, detection AP eval, checkpoint remapping, freeze-flag disambiguation, early-stop for detection AP, DDP-safety, save-by strategies).
- Added: `Fine-tune/utils/detection_eval.py`, `Fine-tune/test_detection_vis.py`, `tools/run_torchrun_train_detector.sh`, `tools/quick_train_check.py`.

**Smoke tests & quick runs**
- Unit/smoke tests (CPU): head, dataset, backbone feature access, detection integration — passed.
- Quick dry-run (subset batches) — validated:
  - Separate logging for Count Loss, Det Loss, Total Loss.
  - Detection AP reported per-val and per-test; detection loss diagnostics added.
  - Checkpoint loading: many pretrained checkpoints contain `backbone.`, `unet.` and `reg_layer.` keys but no `head.` keys — the detection head will be randomly initialized when absent.

**Trainer / DDP / checkpoint fixes**
- Disambiguated freeze flags so they act on explicit submodules:
  - `--freeze-backbone` freezes the internal Swin transformer (the `backbone` submodule).
  - `--freeze-unet` freezes only the U-Net parameters.
  - `--freeze-counter` freezes only `reg_layer` (the regression head).
  This eliminates the previous overlap where `--freeze-backbone` implicitly froze unet/reg_layer.
- DDP: `find_unused_parameters=True` enabled when any of the freeze flags are active to avoid NCCL reduction errors.
- Checkpoint loading: permissive remapping (prefix insert/strip heuristics) + `strict=False` used so resumes work across small architecture changes (e.g., added `head`).

**Recent fixes & clarifications**
- RD loss (regional density) appeared as 0 in detection-only runs — this is expected because RD is a counting loss. For `--task multi` RD is computed; the trainer now explicitly requests backbone features in multi-task mode so RD is non-zero when applicable.
- `test_game.py` was hardened to accept models returning either `density` or `(density, features)` so older checkpoint forms can be tested.

**Added helpers / visualization**
- `Fine-tune/test_detection_vis.py` — randomly selects test samples, runs detection inference, and writes side-by-side RGB+Thermal visualizations with predicted center points overlayed (saved to `visuals_detection/` by default).
- `tools/run_torchrun_train_detector.sh` — helper to call `torchrun` with the detection config (freeze backbone + resume specified checkpoint).

**Known issues & recommended mitigations**
- GPU OOM when unfreezing the full Swin transformer: mitigations include keeping `--unfreeze-epoch -1`, reducing `--batch-size`, or enabling gradient checkpointing in the Swin implementation.
- If detection AP does not improve, consider unfreezing backbone earlier (`--unfreeze-epoch`), tuning learning rate, or increasing `--det-weight` for stronger supervision.

**Decisions requested / next actions**
1. Add additional checkpoint key remapping rules (e.g., `heads.` -> `head.`) for more robust resume behavior.
2. Add selective unfreeze (last-N transformer blocks) and/or gradient checkpointing to reduce memory pressure when unfreezing.
3. Extend save strategies (combined multi-task scoring) or tune the `--save-by` default for your experiments.

For implementation details, tests, and command examples see the markdowns in the `.plan/` directory.

---

**Log**: updated 2025-11-23 to reflect detection integration, trainer fixes (freeze flag disambiguation, early-stop), visualization tooling, and recent quick runs.


# Extension Progress — 2025-11-22

**Summary**
- **Scope:** Extended the Free-Lunch counting code with a center-based detection branch, dataset targets, detection evaluation, multi‑GPU readiness, and checkpoint/resume improvements.
- **Status:** Implementation and smoke tests completed (CPU); short GPU dry‑runs and distributed experiments executed and debugged. Core changes are in the `Fine-tune` folder.

**What I implemented**
- **Detection model & head:** `Fine-tune/models/detection/center_head.py` (CenterHead) and `Fine-tune/models/detection/det_model.py` (wrapper).
- **Dataset & targets:** `Fine-tune/datasets/dm_detection.py` — returns `rgb`, `t`, `heatmap`, `size`, `offset`, and `points`. Implemented stable Gaussian heatmap targets (configurable `sigma` in output-space).
- **Trainer integration:** `Fine-tune/utils/dm_regression_trainer.py` — supports `--task detection|multi`, detection losses (BCE for heatmap, L1 for size/offset), combined multi-task loss, DDP-safe behavior, checkpoint remapping, and `--freeze-counter` support.
- **CLI:** `Fine-tune/train.py` — new flags: `--task`, `--freeze-backbone`, `--freeze-counter`, `--unfreeze-epoch`, `--det-weight`.
- **Evaluation:** Implemented center-based AP computation and heatmap peak extraction (used in validation/test flows and unit tests).
- **Tests:** Unit/smoke tests added under `.unit_test` (head, dataset, backbone feature access, detection integration). Detection-eval unit tests were added and executed.

**Files changed / added (high level)**
- Modified: `Fine-tune/models/counting/swin_unet.py` (expose backbone features via `get_backbone_features`).
- Added: `Fine-tune/models/detection/center_head.py`, `Fine-tune/models/detection/det_model.py`.
- Modified: `Fine-tune/datasets/dm_detection.py` (Gaussian heatmaps, size/offset targets).
- Modified: `Fine-tune/utils/dm_regression_trainer.py` (loss separation, checkpoint remap, freeze logic, DDP fixes, detection logging, saving by AP for detection task).
- Added: `Fine-tune/utils/detection_eval.py` (peak extraction + AP), `.unit_test/*` tests, and `tools/quick_train_check.py` (small dry‑run harness).

**Smoke tests & quick runs**
- Unit/smoke tests (CPU): head, dataset, swin feature extraction, detection integration — **passed**.
- Quick dry-run (subset batches) using `tools/quick_train_check.py` — demonstrated:
	- Separate logging for Count Loss, Det Loss, Total Loss.
	- Detection AP and detection loss (val/test) diagnostics.
	- Checkpoint loading: backbone + `unet` + `reg_layer` keys present in provided checkpoint; `head.*` keys were missing (so head initialized from scratch).

**GPU / DDP experiments**
- Ran short GPU dry-runs and a 4‑GPU DDP dry-run.
- Observed/solved issues:
	- OOM when unfreezing the entire backbone (Swin transformer) — mitigations discussed below.
	- DDP reduction error when many params were frozen -> fixed by enabling `find_unused_parameters=True` when freezing is active.
	- Checkpoint key-prefix mismatches (e.g., `unet.` vs `backbone.unet.`) handled via light remapping and `strict=False` loads.

**Current known issues & clarifications**
- The checkpoint `.weights/drone_rgbt_best_494_781.pth` contains `backbone.*`, `unet.*`, and `reg_layer.*` keys but no `head.*` keys. As a result, when `--freeze-backbone` is used the trainer froze the whole `Swin_BM_RGBT` object (which contains `unet` and `reg_layer`), leaving only the detection `head` trainable. This led to a large detection loss dominating the previously reported "counting" loss metric. I separated meters and logging so Count/Det/Total losses are now reported independently.

**Recommended mitigations and next steps**
- Short-term (fast):
	- If you want to keep `unet`/`reg_layer` trainable while freezing only the Swin transformer, I can patch the trainer so `--freeze-backbone` only freezes the internal Swin transformer (not the enclosing `Swin_BM_RGBT` object). This will allow the counter to train while keeping backbone weights frozen.
	- For GPU OOMs: use `--unfreeze-epoch -1` (never unfreeze), smaller `--batch-size` (1), or enable Swin gradient checkpointing.
- Medium-term:
	- Add a configurable save strategy: currently `detection` saves by AP, `counting` saves by GAME0; I can add a `--save-by` flag or implement a combined multi-task score.
	- Optionally map alternate checkpoint prefixes (e.g., `heads.`) into `head.` automatically when loading.

**Commands / quick checks**
- Run the training entrypoint (single-GPU):
	```bash
	python3 Fine-tune/train.py --data-dir ./.data/DroneRGBT_counting --save-dir ./checkpoints --task detection --batch-size 1 --freeze-backbone --resume .weights/drone_rgbt_best_494_781.pth --max-epoch 1 --num-workers 2 --device 0
	```
- Run the small dry-run harness (uses subset of dataset):
	```bash
	python3 tools/quick_train_check.py
	```

**Decisions requested / next action choices**
1. Patch trainer so `--freeze-backbone` only freezes the Swin transformer (not `unet`/`reg_layer`).
2. Add mapping rules for alternate checkpoint prefixes (e.g., `heads.` -> `head.`).
3. Add a `--save-by` option or a combined multi-task scoring function for model selection.
4. Implement selective unfreeze (last N blocks) or enable gradient checkpointing to reduce GPU memory when unfreezing.

Tell me which option(s) you prefer and I will implement and re-run a short verification.

---

**Log**: this file was reformatted and updated on 2025-11-22 to reflect implemented detection integration, tests, DDP fixes, checkpoint mapping, and the recent quick dry-run results.