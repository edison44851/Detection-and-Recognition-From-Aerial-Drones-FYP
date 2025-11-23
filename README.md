# Free-Lunch-Multimodal-Counting (FYP fork)

This repository is a final-year-project (FYP) fork of the Free-Lunch multi-modal crowd counting codebase. It builds on the original implementation and adds a small center-based detection branch, multi-task training and evaluation utilities, improved checkpoint/resume handling, and several experiment/visualization helpers.

Core differences and major changelog
- Added a center-based detection head and dataset targets (heatmap/size/offset) to support detection on aerial RGBT imagery.
- Integrated detection losses and AP-based evaluation into the trainer; separated count/detection/total loss logging.
- Improved checkpoint loading with permissive prefix remapping and `strict=False` loads so you can resume older Free-Lunch checkpoints even when adding new head modules.
- Disambiguated freeze flags: `--freeze-backbone`, `--freeze-unet`, and `--freeze-counter` now target separate submodules (Swin backbone, U-Net, and reg_layer respectively).
- Added a detection visualization tool `Fine-tune/test_detection_vis.py` to generate RGB+Thermal overlays with predicted centers.
- Added helper scripts for launching distributed runs (`tools/run_torchrun_train_detector.sh`) and quick dry-run checks (`tools/quick_train_check.py`).
- DDP improvements: when large parameter subsets are frozen, `find_unused_parameters=True` is toggled to avoid NCCL reduction errors.

Where to find details
- Implementation notes, tests, quick-run logs and discussion are collected under the `.plan/` directory. Start with `.plan/extension_progress.md` for a short status and next actions.

Quick examples
- Train detection head (single-GPU):
  ```bash
  python3 Fine-tune/train.py --data-dir ./.data/DroneRGBT_counting --save-dir ./checkpoints --task detection --batch-size 1 --freeze-backbone --resume .weights/drone_rgbt_best_494_781.pth --max-epoch 50 --device 0
  ```

- Launch via `torchrun` helper (example):
  ```bash
  ./tools/run_torchrun_train_detector.sh --data-dir .data/DroneRGBT_counting --save-dir ./ckpt_verify --nproc 1 --device 0
  ```

- Visualize detection outputs from `best_model.pth`:
  ```bash
  python3 Fine-tune/test_detection_vis.py --data-dir .data/DroneRGBT_counting --ckpt checkpoints/1122-222336/best_model.pth --out ./visuals_detection --num 12
  ```

Notes
- This fork is intended for experimentation and reproducibility for a final year project. Refer to `.plan/` for design rationale, test results, and reproducibility notes.

If you want any of the above features refined (e.g., selective unfreeze, prefix remapping rules, or expanded save strategies), open an issue or request and I will implement it.
