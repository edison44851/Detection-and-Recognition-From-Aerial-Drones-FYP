#!/usr/bin/env bash
set -euo pipefail

# Post-training evaluation and visualization script.
#
# Runs inference on test split using same settings as training evaluation,
# then optionally applies experimental NMS settings to explore precision/recall
# trade-offs.
#
# By default (with EVAL_NMS_RADIUS and EVAL_SOFT_NMS_SIGMA empty), produces
# IDENTICAL results to training evaluation. Set these variables to experiment.
#
# Output structure:
#   raw/     - Standard inference (matches training evaluation)
#   tiles/   - Optional tiled inference (SAHI-style)
#   orig/    - Standard inference with optional experimental NMS
#   masked/  - Full-image inference followed by mask filtering on GT/predictions

CKPT=${1:-checkpoints_phase6/phase6.3_full_features/phase6_3_best_model_epoch_40.pth}
DATA_DIR=${2:-.data/DroneRGBT_masked}
OUT_DIR=${3:-./.tmp_posttrain_phase6/masked}
NUM_VIS=${4:-3}
DOWNSAMPLE=${5:-4}

# Inference settings
NUM=10000  # Process all images (or large number)
BATCH_SIZE=4
NUM_WORKERS=4

MIN_SCORE_RAW=0.1    # Raw: Low threshold to capture all detections for analysis
MIN_SCORE_TILE=0.3   # SAHI-style: Phase 6.3 adjusted for full features 
MIN_SCORE_ORIG=0.4   # Production: Phase 6.3 baseline (note: may be too high, consider 0.15-0.20)
AP_DIST_THRESH=15.0  # AP distance threshold (pixels) - increased to 15px to account for slight localization offsets
MAX_DETS=300
EVAL_NMS_RADIUS=11.07  # Match training configuration (radius NMS, 11.07 px)
EVAL_SOFT_NMS_SIGMA=0.5  # Match training configuration
NMS_KERNEL=3
HEAD_CONV=256
USE_DECONV=1
KEYPOINT_MODE=1
USE_FPN=1
USE_BCE_LOGITS=1
DET_USE_GN=1
TILE_SIZE=512
TILE_OVER=0.25

mkdir -p "$OUT_DIR"/raw "$OUT_DIR"/tiles "$OUT_DIR"/orig "$OUT_DIR"/masked

echo "Post-training evaluation"
echo "  ckpt: $CKPT"
echo "  data: $DATA_DIR"
echo "  out:  $OUT_DIR"
echo "  AP distance threshold: ${AP_DIST_THRESH}px (increased from 8px to account for slight localization offsets)"
echo ""
echo "Inference defaults match training evaluation exactly."
echo "To enable post-extraction NMS, set EVAL_NMS_RADIUS or EVAL_SOFT_NMS_SIGMA"
echo ""

echo "1) Standard inference (training-like, no post-extraction NMS)"
source .venv/bin/activate && uv run Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/raw" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$MIN_SCORE_RAW" --ap-dist-thresh "$AP_DIST_THRESH" \
  --max-dets "$MAX_DETS" --nms-kernel "$NMS_KERNEL" \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" ) \
  $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" ) \
  --scores-csv "scores.csv" --scores-hist "scores.png"
INDICES_FILE="$OUT_DIR/raw/selected_indices.txt"

echo ""
echo "2) Tiled inference (SAHI-style, same training parameters)"
source .venv/bin/activate && uv run Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/tiles" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$MIN_SCORE_TILE" --ap-dist-thresh "$AP_DIST_THRESH" \
  --max-dets "$MAX_DETS" --nms-kernel "$NMS_KERNEL" \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" ) \
  $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" ) \
  ${EVAL_NMS_RADIUS:+--eval-nms-radius "$EVAL_NMS_RADIUS"} \
  ${EVAL_SOFT_NMS_SIGMA:+--eval-soft-nms-sigma "$EVAL_SOFT_NMS_SIGMA"} \
  --tile-size "$TILE_SIZE" --tile-overlap "$TILE_OVER" \
  --indices-file "$INDICES_FILE" \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo ""
echo "3) Experimental NMS test (same training parameters + optional post-NMS settings)"
source .venv/bin/activate && uv run Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/orig" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$MIN_SCORE_ORIG" --ap-dist-thresh "$AP_DIST_THRESH" \
  --max-dets "$MAX_DETS" --nms-kernel "$NMS_KERNEL" \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" ) \
  $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" ) \
  --indices-file "$INDICES_FILE" \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo ""
echo "4) Mask-filtered evaluation (full-image inference, then filter GT/predictions by mask)"
python3 Fine-tune/test_detection_vis_masked.py \
  --data-dir .data/DroneRGBT_converted \
  --ckpt "$CKPT" \
  --mask-dir .data/masked_image \
  --split test \
  --out "$OUT_DIR/masked" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$MIN_SCORE_ORIG" --ap-dist-thresh "$AP_DIST_THRESH" \
  --max-dets "$MAX_DETS" --nms-kernel "$NMS_KERNEL" \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" ) \
  $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" ) \
  ${EVAL_NMS_RADIUS:+--eval-nms-radius "$EVAL_NMS_RADIUS"} \
  ${EVAL_SOFT_NMS_SIGMA:+--eval-soft-nms-sigma "$EVAL_SOFT_NMS_SIGMA"} \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo ""
echo "=== Evaluation Complete ==="
echo "Results written to: $OUT_DIR"
echo "  AP distance threshold: ${AP_DIST_THRESH}px (for TP/FP matching)"
echo "  raw/     - Standard inference (matches training evaluation)"
echo "  tiles/   - Tiled inference (SAHI-style)"
echo "  orig/    - Inference with optional post-NMS (EVAL_NMS_RADIUS=$EVAL_NMS_RADIUS, EVAL_SOFT_NMS_SIGMA=$EVAL_SOFT_NMS_SIGMA)"
echo "  masked/  - Full-image inference, then mask-filter GT/predictions"
echo ""
echo "Each output directory contains:"
echo "  - report.txt      : TP/FP/FN summary, AP/Precision/Recall/F1 metrics (using ${AP_DIST_THRESH}px threshold)"
echo "  - scores.csv      : Per-prediction scores and TP/FP labels"
echo "  - scores.png      : Histogram comparing TP vs FP score distribution"
echo "  - *.jpg           : Sample visualizations with predictions overlay"
echo ""

exit 0
