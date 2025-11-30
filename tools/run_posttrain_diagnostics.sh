#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run post-training diagnostics:
# 1) raw (no NMS, low threshold)
# 2) tiled (SAHI-style)
# 3) default visualization (moderate threshold + NMS)

CKPT=${1:-checkpoints/1130-171113/best_model.pth}
DATA_DIR=${2:-.data/DroneRGBT_converted}
OUT_DIR=${3:-./tmp_posttrain_1130-171113}
NUM=${4:-64}
DOWNSAMPLE=${5:-8}

MIN_SCORE_RAW=0.01
MIN_SCORE_TILE=0.01
MAX_DETS=1000
NMS_RADIUS_RAW=0
NMS_RADIUS_DEF=4
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
  --num "$NUM" --downsample-ratio "$DOWNSAMPLE" \
  --min-score "$MIN_SCORE_RAW" --ap-dist-thresh 8.0 \
  --max-dets "$MAX_DETS" --nms-radius "$NMS_RADIUS_RAW" \
  --scores-csv "scores.csv" --scores-hist "scores.png"
INDICES_FILE="$OUT_DIR/raw/selected_indices.txt"

echo "\n2) Tiled inference (SAHI-style)"
python3 Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/tiles" \
  --num "$NUM" --downsample-ratio "$DOWNSAMPLE" \
  --min-score "$MIN_SCORE_TILE" --ap-dist-thresh 8.0 \
  --max-dets "$MAX_DETS" --nms-radius "$NMS_RADIUS_DEF" \
  --tile-size "$TILE_SIZE" --tile-overlap "$TILE_OVER" \
  --indices-file "$INDICES_FILE" \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo "\n3) Default visualization (moderate threshold + NMS)"
python3 Fine-tune/test_detection_vis.py \
  --data-dir "$DATA_DIR" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/orig" \
  --num "$NUM" --downsample-ratio "$DOWNSAMPLE" \
  --min-score 0.05 --ap-dist-thresh 8.0 \
  --max-dets 200 --nms-radius "$NMS_RADIUS_DEF" \
  --indices-file "$INDICES_FILE" \
  --scores-csv "scores.csv" --scores-hist "scores.png"

echo "\nDiagnostics finished. Outputs in: $OUT_DIR"
echo "  raw:   $OUT_DIR/raw"
echo "  tiles: $OUT_DIR/tiles"
echo "  orig:  $OUT_DIR/orig"

exit 0
