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

DATA_DIR=".data/DroneRGBT_counting"
SAVE_DIR="./checkpoints"
NPROC=4
DEVICE="0,1,2,3"
BATCH_SIZE=1
MAX_EPOCH=100
DET_WEIGHT=1.0
EXTRA_ARGS=""

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

# Build torchrun command. We don't pass --local_rank (torchrun sets it in env).
CMD=(torchrun --nproc_per_node=${NPROC} Fine-tune/train.py
     --task detection
     --freeze-backbone
     --freeze-unet
     --freeze-counter
     --resume .weights/drone_rgbt_best_494_781.pth
     --downsample-ratio 8
     --data-dir "${DATA_DIR}"
     --save-dir "${SAVE_DIR}"
     --batch-size "${BATCH_SIZE}"
     --max-epoch "${MAX_EPOCH}"
     --det-weight "${DET_WEIGHT}"
     --device "${DEVICE}")

if [[ -n "$EXTRA_ARGS" ]]; then
  CMD+=( $EXTRA_ARGS )
fi

echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
