#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run post-training diagnostics:
# 1) raw (no NMS, low threshold)
# 2) tiled (SAHI-style)
# 3) default visualization (moderate threshold + NMS)

CKPT=${1:-checkpoints/1211-171944_teammate/best_model.pth}
DATA_DIR=${2:-.data/DroneRGBT_converted}
OUT_DIR=${3:-./.tmp_posttrain/1211-171944_teammate}
NUM_VIS=${4:-64}
DOWNSAMPLE=${5:-4}

# Inference settings
NUM=10000  # Process all images (or large number)
BATCH_SIZE=8
NUM_WORKERS=4

MIN_SCORE_RAW=0.1
MIN_SCORE_TILE=0.1
MAX_DETS=150
NMS_RADIUS=3.0
NMS_KERNEL=3
HEAD_CONV=256
USE_DECONV=1
KEYPOINT_MODE=1
USE_FPN=1
SCORE_THRESH_RAW=0.1
SCORE_THRESH_TILE=0.1
SCORE_THRESH_ORIG=0.1
SOFT_NMS_SIGMA_RAW=
SOFT_NMS_SIGMA_TILE=
SOFT_NMS_SIGMA_ORIG=
TILE_SIZE=512
TILE_OVER=0.25

mkdir -p "$OUT_DIR"/raw "$OUT_DIR"/tiles "$OUT_DIR"/orig

echo "Post-train diagnostics"
echo "  ckpt: $CKPT"
echo "  data: $DATA_DIR"
echo "  out:  $OUT_DIR"

echo "\n1) Raw (no NMS, low threshold)"
python3 Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/raw" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$MIN_SCORE_RAW" --ap-dist-thresh 8.0 \
  --max-dets "$MAX_DETS" --no-nms \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  ${SCORE_THRESH_RAW:+--score-thresh "$SCORE_THRESH_RAW"} \
  ${SOFT_NMS_SIGMA_RAW:+--soft-nms-sigma "$SOFT_NMS_SIGMA_RAW"} \
  --scores-csv "scores.csv" --scores-hist "scores.png"
INDICES_FILE="$OUT_DIR/raw/selected_indices.txt"

echo "\n2) Tiled inference (SAHI-style)"
python3 Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/tiles" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$MIN_SCORE_TILE" --ap-dist-thresh 8.0 \
  --max-dets "$MAX_DETS" --nms-radius "$NMS_RADIUS" --nms-kernel "$NMS_KERNEL" \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  ${SCORE_THRESH_TILE:+--score-thresh "$SCORE_THRESH_TILE"} \
  ${SOFT_NMS_SIGMA_TILE:+--soft-nms-sigma "$SOFT_NMS_SIGMA_TILE"} \
  --tile-size "$TILE_SIZE" --tile-overlap "$TILE_OVER" \
  --indices-file "$INDICES_FILE" \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo "\n3) Default visualization (moderate threshold + NMS)"
python3 Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/orig" \
  --num "$NUM" --num-vis "$NUM_VIS" --downsample-ratio "$DOWNSAMPLE" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --min-score "$SCORE_THRESH_ORIG" --ap-dist-thresh 8.0 \
  --max-dets 200 --nms-radius "$NMS_RADIUS" --nms-kernel "$NMS_KERNEL" \
  --head-conv "$HEAD_CONV" $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" ) \
  $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" ) \
  $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" ) \
  --fixed-box-size "$FIXED_BOX_SIZE" \
  ${SCORE_THRESH_ORIG:+--score-thresh "$SCORE_THRESH_ORIG"} \
  ${SOFT_NMS_SIGMA_ORIG:+--soft-nms-sigma "$SOFT_NMS_SIGMA_ORIG"} \
  --indices-file "$INDICES_FILE" \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo "\nDiagnostics finished. Outputs in: $OUT_DIR"
echo "  raw:   $OUT_DIR/raw"
echo "  tiles: $OUT_DIR/tiles"
echo "  orig:  $OUT_DIR/orig"

exit 0
