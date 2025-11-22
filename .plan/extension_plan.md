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