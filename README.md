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
We strongly recommend using the convenience launcher `tools/train_entry.sh` for training. It wraps `torchrun` safely, sets sensible defaults, and forwards extra flags to `Fine-tune/train.py`.

- Train detection head (single-GPU):
  ```bash
  tools/train_entry.sh --data-dir .data/DroneRGBT_counting --save-dir ./checkpoints --nproc 1 --device 0 --batch-size 1 --max-epoch 50 -- --freeze-backbone --resume .weights/drone_rgbt_best_494_781.pth
  ```

- Train detection head (multi-GPU on one node):
  ```bash
  tools/train_entry.sh --data-dir .data/DroneRGBT_counting --save-dir ./checkpoints --nproc 4 --device 0,1,2,3 --batch-size 4 --max-epoch 100 -- --freeze-backbone --freeze-unet --freeze-counter --resume .weights/drone_rgbt_best_494_781.pth
  ```

- Visualize detection outputs from `best_model.pth`:
  ```bash
  python3 Fine-tune/test_detection_vis.py --data-dir .data/DroneRGBT_counting --ckpt checkpoints/1122-222336/best_model.pth --out ./visuals_detection --num 12
  ```

Flags (forwarded to `Fine-tune/train.py`)
- `--data-dir`: training data directory
- `--save-dir`: directory to save models
- `--lr`: initial learning rate
- `--resume`: checkpoint path to resume
- `--device`: CUDA devices (single value or comma-separated)
- `--crop-size`: input crop size (default 224)
- `--task`: `counting` | `detection`
- `--freeze-backbone`: freeze backbone at start
- `--freeze-counter`: freeze counting/regression head at start
- `--freeze-unet`: freeze U-Net at start
- `--unfreeze-epoch`: epoch to unfreeze backbone (-1 = never)
- `--det-weight`: detection loss weight
- `--local_rank`: set by torchrun; no need to pass manually
- `--det-patience`: validation patience for detection AP early stopping
- `--weight-decay`: optimizer weight decay
- `--max-model-num`: maximum number of checkpoints to keep
- `--max-epoch`: max training epochs
- `--val-epoch`: validation frequency (in epochs)
- `--val-start`: epoch to start validation
- `--save-all-best`: keep multiple best checkpoints
- `--batch-size`: train batch size
- `--num-workers`: dataloader workers
- `--downsample-ratio`: model output stride for targets
- `--wot`: OT loss weight
- `--wtv`: TV loss weight
- `--reg`: Sinkhorn entropy regularization
- `--num-of-iter-in-ot`: Sinkhorn iterations
- `--norm-cood`: normalize coordinates in distance
- `--wrd`: regional density loss weight

Note: `tools/train_entry.sh` accepts the same flags after `--` and safely forwards them to `Fine-tune/train.py`. Before `--`, you can set launcher options like `--nproc` and `--device`.

Notes
- This fork is intended for experimentation and reproducibility for a final year project. Refer to `.plan/` for design rationale, test results, and reproducibility notes.

If you want any of the above features refined (e.g., selective unfreeze, prefix remapping rules, or expanded save strategies), open an issue or request and I will implement it.
