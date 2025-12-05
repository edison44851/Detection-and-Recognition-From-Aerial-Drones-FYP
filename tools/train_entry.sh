#!/usr/bin/env bash
# Convenience script to launch detector-head training with torchrun.
# It calls Fine-tune/train.py with `--task detection`, `--freeze-backbone`
# and resumes from .weights/drone_rgbt_best_494_781.pth by default.
#
# Usage:
#   ./tools/run_torchrun_train_detector.sh --data-dir /path/to/data --save-dir ./ckpts --nproc 1 --device 0
# For multi-GPU on a single node:
#   ./tools/run_torchrun_train_detector.sh --data-dir /path --save-dir ./ckpts --nproc 2 --device 0,1

set -euo pipefail

DATA_DIR=".data/DroneRGBT_converted"
SAVE_DIR="./checkpoints"
NPROC=4
DEVICE="0,1,2,3"
BATCH_SIZE=1
MAX_EPOCH=100
DET_WEIGHT=1.0
EXTRA_ARGS=""
DET_POS_WEIGHT=7.0
HEAD_LR=0.001
AP_DIST_THRESH=8.0
USE_DET_ADAPTOR=1
USE_BCE_LOGITS=1
DET_USE_GN=1
DET_SIGMA=2.0
USE_FOCAL_HEATMAP=1
FOCAL_ALPHA=0.25
FOCAL_GAMMA=1.5
DET_NEG_TOPK_RATIO=0.1
USE_IOU_SIZE=1
IOU_WEIGHT=0.5
EVAL_NMS="radius"
EVAL_NMS_RADIUS=4.0
EVAL_SOFT_NMS_SIGMA=0.5
HEAD_CONV=256
USE_DECONV=1
NMS_KERNEL=3
OUTPUT_STRIDE=4

print_usage(){
  echo "Usage: $0 --data-dir PATH [--save-dir PATH] [--nproc N] [--device DEVICES] [--batch-size N] [--max-epoch N] [--det-weight W] -- [extra args passed to train.py]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --save-dir)
      SAVE_DIR="$2"; shift 2;;
    --nproc)
      NPROC="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --max-epoch)
      MAX_EPOCH="$2"; shift 2;;
    --det-weight)
      DET_WEIGHT="$2"; shift 2;;
    --det-pos-weight)
      DET_POS_WEIGHT="$2"; shift 2;;
    --head-lr)
      HEAD_LR="$2"; shift 2;;
    --ap-dist-thresh)
      AP_DIST_THRESH="$2"; shift 2;;
    --use-det-adaptor)
      USE_DET_ADAPTOR=1; shift 1;;
    --use-bce-logits)
      USE_BCE_LOGITS=1; shift 1;;
    --det-use-gn)
      DET_USE_GN=1; shift 1;;
    --det-sigma)
      DET_SIGMA="$2"; shift 2;;
    --use-focal-heatmap)
      USE_FOCAL_HEATMAP=1; shift 1;;
    --focal-alpha)
      FOCAL_ALPHA="$2"; shift 2;;
    --focal-gamma)
      FOCAL_GAMMA="$2"; shift 2;;
    --det-neg-topk-ratio)
      DET_NEG_TOPK_RATIO="$2"; shift 2;;
    --use-iou-size)
      USE_IOU_SIZE=1; shift 1;;
    --iou-weight)
      IOU_WEIGHT="$2"; shift 2;;
    --eval-nms)
      EVAL_NMS="$2"; shift 2;;
    --eval-nms-radius)
      EVAL_NMS_RADIUS="$2"; shift 2;;
    --eval-soft-nms-sigma)
      EVAL_SOFT_NMS_SIGMA="$2"; shift 2;;
    --head-conv)
      HEAD_CONV="$2"; shift 2;;
    --use-deconv)
      USE_DECONV=1; shift 1;;
    --nms-kernel)
      NMS_KERNEL="$2"; shift 2;;
    --output-stride)
      OUTPUT_STRIDE="$2"; shift 2;;
    --)
      shift
      EXTRA_ARGS="$*"
      break;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

if [[ -z "$DATA_DIR" ]]; then
  echo "--data-dir is required"
  print_usage
  exit 1
fi

echo "Running detector-head training with torchrun"
echo "  data-dir: $DATA_DIR"
echo "  save-dir: $SAVE_DIR"
echo "  nproc:    $NPROC"
echo "  device:   $DEVICE"
echo "  batch-size: $BATCH_SIZE"
echo "  max-epoch:  $MAX_EPOCH"
echo "  det-pos-weight: $DET_POS_WEIGHT"
echo "  head-lr: $HEAD_LR"
echo "  ap-dist-thresh: $AP_DIST_THRESH"
echo "  use-det-adaptor: $USE_DET_ADAPTOR"
echo "  det-use-gn: $DET_USE_GN"
echo "  det-sigma: $DET_SIGMA"
echo "  use-focal-heatmap: $USE_FOCAL_HEATMAP (alpha=$FOCAL_ALPHA gamma=$FOCAL_GAMMA)"
echo "  det-neg-topk-ratio: $DET_NEG_TOPK_RATIO"
echo "  use-iou-size: $USE_IOU_SIZE (iou-weight=$IOU_WEIGHT)"
echo "  eval-nms: $EVAL_NMS (radius=$EVAL_NMS_RADIUS soft-sigma=$EVAL_SOFT_NMS_SIGMA)"
echo "  head-conv: $HEAD_CONV"
echo "  use-deconv: $USE_DECONV"
echo "  nms-kernel: $NMS_KERNEL"
echo "  output-stride: $OUTPUT_STRIDE"

# Build torchrun command. We don't pass --local_rank (torchrun sets it in env).
if [[ "$NPROC" -eq 1 ]]; then
  # Run single-process directly with python3 to avoid torchrun rendezvous port issues
  CMD=(python3 Fine-tune/train.py
    --task detection
    --freeze-backbone
    --freeze-unet
    --freeze-counter
    --resume .weights/drone_rgbt_best_494_781.pth
    --output-stride "${OUTPUT_STRIDE}"
    --data-dir "${DATA_DIR}"
    --save-dir "${SAVE_DIR}"
    --batch-size "${BATCH_SIZE}"
    --max-epoch "${MAX_EPOCH}"
    --det-weight "${DET_WEIGHT}"
    --det-pos-weight "${DET_POS_WEIGHT}"
    --head-lr "${HEAD_LR}"
    --ap-dist-thresh "${AP_DIST_THRESH}"
    $( [[ "$USE_DET_ADAPTOR" -eq 1 ]] && echo "--use-det-adaptor" )
    $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" )
    $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" )
    $( [[ -n "$DET_SIGMA" ]] && echo "--det-sigma $DET_SIGMA" )
    $( [[ "$USE_FOCAL_HEATMAP" -eq 1 ]] && echo "--use-focal-heatmap --focal-alpha $FOCAL_ALPHA --focal-gamma $FOCAL_GAMMA" )
    $( [[ -n "$DET_NEG_TOPK_RATIO" ]] && echo "--det-neg-topk-ratio $DET_NEG_TOPK_RATIO" )
    $( [[ "$USE_IOU_SIZE" -eq 1 ]] && echo "--use-iou-size --iou-weight $IOU_WEIGHT" )
    $( [[ -n "$EVAL_NMS" ]] && echo "--eval-nms $EVAL_NMS" )
    $( [[ -n "$EVAL_NMS_RADIUS" ]] && echo "--eval-nms-radius $EVAL_NMS_RADIUS" )
    $( [[ -n "$EVAL_SOFT_NMS_SIGMA" ]] && echo "--eval-soft-nms-sigma $EVAL_SOFT_NMS_SIGMA" )
    $( [[ -n "$HEAD_CONV" ]] && echo "--head-conv $HEAD_CONV" )
    $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" )
    $( [[ -n "$NMS_KERNEL" ]] && echo "--nms-kernel $NMS_KERNEL" )
    --device "${DEVICE}" --local_rank 0)
else
  CMD=(torchrun --nproc_per_node=${NPROC} Fine-tune/train.py
    --task detection
    --freeze-backbone
    --freeze-unet
    --freeze-counter
    --resume .weights/drone_rgbt_best_494_781.pth
    --output-stride "${OUTPUT_STRIDE}"
    --data-dir "${DATA_DIR}"
    --save-dir "${SAVE_DIR}"
    --batch-size "${BATCH_SIZE}"
    --max-epoch "${MAX_EPOCH}"
    --det-weight "${DET_WEIGHT}"
    --det-pos-weight "${DET_POS_WEIGHT}"
    --head-lr "${HEAD_LR}"
    --ap-dist-thresh "${AP_DIST_THRESH}"
    $( [[ "$USE_DET_ADAPTOR" -eq 1 ]] && echo "--use-det-adaptor" )
    $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" )
    $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" )
    $( [[ -n "$DET_SIGMA" ]] && echo "--det-sigma $DET_SIGMA" )
    $( [[ "$USE_FOCAL_HEATMAP" -eq 1 ]] && echo "--use-focal-heatmap --focal-alpha $FOCAL_ALPHA --focal-gamma $FOCAL_GAMMA" )
    $( [[ -n "$DET_NEG_TOPK_RATIO" ]] && echo "--det-neg-topk-ratio $DET_NEG_TOPK_RATIO" )
    $( [[ "$USE_IOU_SIZE" -eq 1 ]] && echo "--use-iou-size --iou-weight $IOU_WEIGHT" )
    $( [[ -n "$EVAL_NMS" ]] && echo "--eval-nms $EVAL_NMS" )
    $( [[ -n "$EVAL_NMS_RADIUS" ]] && echo "--eval-nms-radius $EVAL_NMS_RADIUS" )
    $( [[ -n "$EVAL_SOFT_NMS_SIGMA" ]] && echo "--eval-soft-nms-sigma $EVAL_SOFT_NMS_SIGMA" )
    $( [[ -n "$HEAD_CONV" ]] && echo "--head-conv $HEAD_CONV" )
    $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" )
    $( [[ -n "$NMS_KERNEL" ]] && echo "--nms-kernel $NMS_KERNEL" )
    --device "${DEVICE}")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  CMD+=( $EXTRA_ARGS )
fi

echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
