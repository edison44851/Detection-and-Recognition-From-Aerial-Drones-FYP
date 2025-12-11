#!/usr/bin/env bash
# Comprehensive training entry script for detector-head and counting tasks.
# Supports all CLI parameters from Fine-tune/train.py with organized categories.
#
# Usage:
#   ./tools/train_entry.sh --data-dir /path/to/data [--options]
#
# Examples:
#   # Detection head training (CenterNet-style)
#   ./tools/train_entry.sh --data-dir .data/DroneRGBT_converted --task detection --nproc 4
#
#   # Custom settings
#   ./tools/train_entry.sh --data-dir .data --task detection --head-conv 128 --use-deconv --nms-kernel 5

set -euo pipefail

# ============================================================================
# DEFAULT CONFIGURATION - Organized by Category
# ============================================================================

# --- GPU / Distributed Training ---
NPROC=4
DEVICE="0,1,2,3"
NUM_WORKERS=4
DDP_TIMEOUT=3600  # DDP watchdog timeout in seconds (default: 1800=30min, set to 3600=1hr)

# --- Data & Paths ---
DATA_DIR="data/DroneRGBT_converted"
SAVE_DIR="./checkpoints"
RESUME="weights/drone_rgbt_best_494_781.pth"

# --- Task Selection & Freezing ---
TASK="detection"
FREEZE_BACKBONE=1
FREEZE_COUNTER=1
FREEZE_UNET=1
UNFREEZE_EPOCH=-1

# --- Training Hyperparameters (General) ---
LR=1e-5
WEIGHT_DECAY=0.0001
BATCH_SIZE=1
MAX_EPOCH=100
CROP_SIZE=224

# --- Checkpointing & Early Stopping ---
MAX_MODEL_NUM=1
VAL_EPOCH=2  # Validate every 2 epochs to reduce DDP sync overhead with prime-sized test set
VAL_START=0
SAVE_ALL_BEST=0
DET_PATIENCE=10

# --- Evaluation ---
AP_DIST_THRESH=8.0
DOWNSAMPLE_RATIO=8
OUTPUT_STRIDE=4

# --- Detection Head Architecture (CenterNet-style) ---
HEAD_CONV=256
USE_DECONV=1
NMS_KERNEL=3
USE_DET_ADAPTOR=1
USE_FPN=1

# --- Phase 1: Keypoint-Only Mode ---
KEYPOINT_MODE=1
FIXED_BOX_SIZE=16

# --- Detection Loss Configuration ---
DET_WEIGHT=1.0
DET_POS_WEIGHT=7.0
USE_BCE_LOGITS=1
DET_USE_GN=1
DET_SIGMA=0.8
HEAD_LR=0.002

# --- Focal Loss (Detection) ---
USE_FOCAL_HEATMAP=1
FOCAL_ALPHA=0.75
FOCAL_GAMMA=1.5

# --- Hard Negative Mining & Size Loss ---
DET_NEG_TOPK_RATIO=0.1
USE_IOU_SIZE=1
IOU_WEIGHT=0.3

# --- Inference-time NMS Options ---
EVAL_NMS="radius"
EVAL_NMS_RADIUS=2.0
EVAL_SOFT_NMS_SIGMA=0.5

# --- Counting Task Parameters (DM-Count) ---
WOT=0.1
WTV=0.01
REG=10.0
NUM_OF_ITER_IN_OT=100
NORM_COOD=0
WRD=0.1

# --- Extra Arguments ---
EXTRA_ARGS=""

# ============================================================================
# HELP TEXT
# ============================================================================

print_usage(){
  cat << 'EOF'
Training entry script with organized parameter categories.

USAGE:
  ./tools/train_entry.sh --data-dir PATH [OPTIONS]

REQUIRED:
  --data-dir PATH              Training data directory

GPU & DISTRIBUTED:
  --nproc N                    Number of processes per node (default: 4)
  --device DEVICES             GPU devices (default: 0,1,2,3)
  --num-workers N              Data loading workers (default: 8)

TASK & FREEZING:
  --task TASK                  Task: detection|counting (default: detection)
  --freeze-backbone            Freeze Swin backbone (default: yes)
  --freeze-counter             Freeze counting head (default: yes)
  --freeze-unet                Freeze U-Net (default: yes)
  --unfreeze-epoch N           Epoch to unfreeze (-1=never, default: -1)

DATA & PATHS:
  --save-dir PATH              Output checkpoint dir (default: ./checkpoints)
  --resume PATH                Resume checkpoint (default: .weights/drone_rgbt_best_494_781.pth)

TRAINING GENERAL:
  --lr LR                      Learning rate (default: 1e-5)
  --weight-decay W             Weight decay (default: 1e-4)
  --batch-size N               Batch size (default: 1)
  --max-epoch N                Max epochs (default: 100)
  --crop-size N                Crop size (default: 224)

EVALUATION & CHECKPOINTING:
  --ap-dist-thresh DIST        AP distance threshold in pixels (default: 8.0)
  --downsample-ratio N         Downsample ratio (default: 8)
  --output-stride N            Output stride (default: 4)
  --det-patience N             Early stopping patience (default: 10)
  --val-epoch N                Validation frequency (default: 1)
  --val-start N                Start validation at epoch (default: 0)
  --max-model-num N            Max checkpoints to keep (default: 1)

DETECTION HEAD (CenterNet-style):
  --head-conv N                Head conv channels (default: 256)
  --use-deconv                 Use deconv upsampling (default: yes)
  --nms-kernel N               NMS kernel size (default: 3)
  --use-det-adaptor            Use detection adaptor (default: yes)

PHASE 1 - KEYPOINT MODE:
  --keypoint-mode              Enable keypoint-only mode (no size head)
  --fixed-box-size N           Fixed box size in pixels for inference (default: 16)

DETECTION LOSS:
  --det-weight W               Detection loss weight (default: 1.0)
  --det-pos-weight W           Positive heatmap weight (default: 7.0)
  --use-bce-logits             Use BCEWithLogitsLoss (default: yes)
  --det-use-gn                 Use GroupNorm in head (default: yes)
  --det-sigma S                Gaussian sigma for heatmaps (default: 2.0)
  --head-lr LR                 Detection head learning rate (default: 0.001)

FOCAL LOSS (Detection):
  --use-focal-heatmap          Enable focal loss (default: yes)
  --focal-alpha A              Focal loss alpha (default: 0.25)
  --focal-gamma G              Focal loss gamma (default: 1.5)

HARD NEGATIVES & SIZE LOSS:
  --det-neg-topk-ratio R       Hard negative ratio (default: 0.1)
  --use-iou-size               Enable IoU size loss (default: yes)
  --iou-weight W               IoU loss weight (default: 0.5)

INFERENCE NMS:
  --eval-nms TYPE              NMS type: radius|soft (default: radius)
  --eval-nms-radius R          Radius NMS radius (default: 4.0)
  --eval-soft-nms-sigma S      Soft-NMS sigma (default: 0.5)

COUNTING (DM-Count):
  --wot W                      OT loss weight (default: 0.1)
  --wtv W                      TV loss weight (default: 0.01)
  --reg R                      Sinkhorn regularization (default: 10.0)
  --num-of-iter-in-ot N        OT iterations (default: 100)
  --norm-cood N                Normalize coordinates (default: 0)
  --wrd W                      RD loss weight (default: 0.1)

OTHER:
  --save-dir PATH              Directory to save models
  --max-model-num N            Max models to keep
  --save-all-best              Save all best models
  --                           Separate to pass extra args

EXAMPLES:
  # Detection with 4 GPUs
  ./tools/train_entry.sh --data-dir .data/DroneRGBT_converted --task detection --nproc 4

  # Detection with custom head config
  ./tools/train_entry.sh --data-dir .data --task detection --head-conv 128 --focal-gamma 2.0

  # Single GPU with custom learning rates
  ./tools/train_entry.sh --data-dir .data --nproc 1 --lr 5e-5 --head-lr 1e-3

EOF
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    # GPU & Distributed
    --nproc)
      NPROC="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2;;
    
    # Data & Paths
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --save-dir)
      SAVE_DIR="$2"; shift 2;;
    --resume)
      RESUME="$2"; shift 2;;
    
    # Task & Freezing
    --task)
      TASK="$2"; shift 2;;
    --freeze-backbone)
      FREEZE_BACKBONE=1; shift 1;;
    --no-freeze-backbone)
      FREEZE_BACKBONE=0; shift 1;;
    --freeze-counter)
      FREEZE_COUNTER=1; shift 1;;
    --no-freeze-counter)
      FREEZE_COUNTER=0; shift 1;;
    --freeze-unet)
      FREEZE_UNET=1; shift 1;;
    --no-freeze-unet)
      FREEZE_UNET=0; shift 1;;
    --unfreeze-epoch)
      UNFREEZE_EPOCH="$2"; shift 2;;
    
    # Training General
    --lr)
      LR="$2"; shift 2;;
    --weight-decay)
      WEIGHT_DECAY="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --max-epoch)
      MAX_EPOCH="$2"; shift 2;;
    --crop-size)
      CROP_SIZE="$2"; shift 2;;
    
    # Evaluation & Checkpointing
    --ap-dist-thresh)
      AP_DIST_THRESH="$2"; shift 2;;
    --downsample-ratio)
      DOWNSAMPLE_RATIO="$2"; shift 2;;
    --output-stride)
      OUTPUT_STRIDE="$2"; shift 2;;
    --det-patience)
      DET_PATIENCE="$2"; shift 2;;
    --val-epoch)
      VAL_EPOCH="$2"; shift 2;;
    --val-start)
      VAL_START="$2"; shift 2;;
    --max-model-num)
      MAX_MODEL_NUM="$2"; shift 2;;
    --save-all-best)
      SAVE_ALL_BEST=1; shift 1;;
    
    # Detection Head (CenterNet-style)
    --head-conv)
      HEAD_CONV="$2"; shift 2;;
    --use-deconv)
      USE_DECONV=1; shift 1;;
    --no-deconv)
      USE_DECONV=0; shift 1;;
    --nms-kernel)
      NMS_KERNEL="$2"; shift 2;;
    --use-det-adaptor)
      USE_DET_ADAPTOR=1; shift 1;;
    --no-det-adaptor)
      USE_DET_ADAPTOR=0; shift 1;;
    --use-fpn)
      USE_FPN=1; shift 1;;
    --no-fpn)
      USE_FPN=0; shift 1;;
    
    # Phase 1: Keypoint Mode
    --keypoint-mode)
      KEYPOINT_MODE=1; shift 1;;
    --no-keypoint-mode)
      KEYPOINT_MODE=0; shift 1;;
    --fixed-box-size)
      FIXED_BOX_SIZE="$2"; shift 2;;
    
    # Detection Loss
    --det-weight)
      DET_WEIGHT="$2"; shift 2;;
    --det-pos-weight)
      DET_POS_WEIGHT="$2"; shift 2;;
    --use-bce-logits)
      USE_BCE_LOGITS=1; shift 1;;
    --no-bce-logits)
      USE_BCE_LOGITS=0; shift 1;;
    --det-use-gn)
      DET_USE_GN=1; shift 1;;
    --no-det-gn)
      DET_USE_GN=0; shift 1;;
    --det-sigma)
      DET_SIGMA="$2"; shift 2;;
    --head-lr)
      HEAD_LR="$2"; shift 2;;
    
    # Focal Loss
    --use-focal-heatmap)
      USE_FOCAL_HEATMAP=1; shift 1;;
    --no-focal-heatmap)
      USE_FOCAL_HEATMAP=0; shift 1;;
    --focal-alpha)
      FOCAL_ALPHA="$2"; shift 2;;
    --focal-gamma)
      FOCAL_GAMMA="$2"; shift 2;;
    
    # Hard Negatives & Size Loss
    --det-neg-topk-ratio)
      DET_NEG_TOPK_RATIO="$2"; shift 2;;
    --use-iou-size)
      USE_IOU_SIZE=1; shift 1;;
    --no-iou-size)
      USE_IOU_SIZE=0; shift 1;;
    --iou-weight)
      IOU_WEIGHT="$2"; shift 2;;
    
    # Inference NMS
    --eval-nms)
      EVAL_NMS="$2"; shift 2;;
    --eval-nms-radius)
      EVAL_NMS_RADIUS="$2"; shift 2;;
    --eval-soft-nms-sigma)
      EVAL_SOFT_NMS_SIGMA="$2"; shift 2;;
    
    # Counting (DM-Count)
    --wot)
      WOT="$2"; shift 2;;
    --wtv)
      WTV="$2"; shift 2;;
    --reg)
      REG="$2"; shift 2;;
    --num-of-iter-in-ot)
      NUM_OF_ITER_IN_OT="$2"; shift 2;;
    --norm-cood)
      NORM_COOD="$2"; shift 2;;
    --wrd)
      WRD="$2"; shift 2;;
    
    # Help & Extra
    --)
      shift
      EXTRA_ARGS="$*"
      break;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown argument: $1"; print_usage; exit 1;;
  esac
done

# ============================================================================
# VALIDATION & DEFAULTS
# ============================================================================

if [[ -z "$DATA_DIR" ]]; then
  echo "ERROR: --data-dir is required"
  print_usage
  exit 1
fi

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "=================================="
echo "  Training Configuration"
echo "=================================="
echo ""
echo "--- GPU & Distributed ---"
echo "  nproc:        $NPROC"
echo "  device:       $DEVICE"
echo "  num-workers:  $NUM_WORKERS"
echo "  ddp-timeout:  ${DDP_TIMEOUT}s"
echo ""
echo "--- Data & Paths ---"
echo "  data-dir:     $DATA_DIR"
echo "  save-dir:     $SAVE_DIR"
echo "  resume:       $RESUME"
echo ""
echo "--- Task & Freezing ---"
echo "  task:              $TASK"
echo "  freeze-backbone:   $FREEZE_BACKBONE"
echo "  freeze-counter:    $FREEZE_COUNTER"
echo "  freeze-unet:       $FREEZE_UNET"
echo "  unfreeze-epoch:    $UNFREEZE_EPOCH"
echo ""
echo "--- Training General ---"
echo "  lr:            $LR"
echo "  weight-decay:  $WEIGHT_DECAY"
echo "  batch-size:    $BATCH_SIZE"
echo "  max-epoch:     $MAX_EPOCH"
echo "  crop-size:     $CROP_SIZE"
echo ""
echo "--- Evaluation & Checkpointing ---"
echo "  ap-dist-thresh:  $AP_DIST_THRESH"
echo "  downsample-ratio: $DOWNSAMPLE_RATIO"
echo "  output-stride:   $OUTPUT_STRIDE"
echo "  det-patience:    $DET_PATIENCE"
echo "  val-epoch:       $VAL_EPOCH"
echo "  val-start:       $VAL_START"
echo "  max-model-num:   $MAX_MODEL_NUM"
echo ""
echo "--- Detection Head (CenterNet) ---"
echo "  head-conv:      $HEAD_CONV"
echo "  use-deconv:     $USE_DECONV"
echo "  nms-kernel:     $NMS_KERNEL"
echo "  use-det-adaptor: $USE_DET_ADAPTOR"
echo "  use-fpn:        $USE_FPN"
echo ""
echo "--- Phase 1: Keypoint Mode ---"
echo "  keypoint-mode:   $KEYPOINT_MODE"
echo "  fixed-box-size:  $FIXED_BOX_SIZE"
echo ""
echo "--- Detection Loss ---"
echo "  det-weight:       $DET_WEIGHT"
echo "  det-pos-weight:   $DET_POS_WEIGHT"
echo "  use-bce-logits:   $USE_BCE_LOGITS"
echo "  det-use-gn:       $DET_USE_GN"
echo "  det-sigma:        $DET_SIGMA"
echo "  head-lr:          $HEAD_LR"
echo ""
echo "--- Focal Loss ---"
echo "  use-focal-heatmap: $USE_FOCAL_HEATMAP"
echo "  focal-alpha:       $FOCAL_ALPHA"
echo "  focal-gamma:       $FOCAL_GAMMA"
echo ""
echo "--- Hard Negatives & Size Loss ---"
echo "  det-neg-topk-ratio: $DET_NEG_TOPK_RATIO"
echo "  use-iou-size:       $USE_IOU_SIZE"
echo "  iou-weight:         $IOU_WEIGHT"
echo ""
echo "--- Inference NMS ---"
echo "  eval-nms:          $EVAL_NMS"
echo "  eval-nms-radius:   $EVAL_NMS_RADIUS"
echo "  eval-soft-nms-sigma: $EVAL_SOFT_NMS_SIGMA"
echo ""
echo "--- Counting (DM-Count) ---"
echo "  wot:  $WOT"
echo "  wtv:  $WTV"
echo "  reg:  $REG"
echo "  num-of-iter-in-ot: $NUM_OF_ITER_IN_OT"
echo "  norm-cood: $NORM_COOD"
echo "  wrd:  $WRD"
echo ""
echo "=================================="
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================

if [[ "$NPROC" -eq 1 ]]; then
  # Single-process mode
  CMD=(python3 Fine-tune/train.py
    --data-dir "${DATA_DIR}"
    --save-dir "${SAVE_DIR}"
    --resume "${RESUME}"
    --task "${TASK}"
    $( [[ "$FREEZE_BACKBONE" -eq 1 ]] && echo "--freeze-backbone" )
    $( [[ "$FREEZE_COUNTER" -eq 1 ]] && echo "--freeze-counter" )
    $( [[ "$FREEZE_UNET" -eq 1 ]] && echo "--freeze-unet" )
    --unfreeze-epoch "${UNFREEZE_EPOCH}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --batch-size "${BATCH_SIZE}"
    --max-epoch "${MAX_EPOCH}"
    --crop-size "${CROP_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --ap-dist-thresh "${AP_DIST_THRESH}"
    --downsample-ratio "${DOWNSAMPLE_RATIO}"
    --output-stride "${OUTPUT_STRIDE}"
    --det-patience "${DET_PATIENCE}"
    --val-epoch "${VAL_EPOCH}"
    --val-start "${VAL_START}"
    --max-model-num "${MAX_MODEL_NUM}"
    $( [[ "$SAVE_ALL_BEST" -eq 1 ]] && echo "--save-all-best" )
    --head-conv "${HEAD_CONV}"
    $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" )
    --nms-kernel "${NMS_KERNEL}"
    $( [[ "$USE_DET_ADAPTOR" -eq 1 ]] && echo "--use-det-adaptor" )
    $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" )
    $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" )
    --fixed-box-size "${FIXED_BOX_SIZE}"
    --det-weight "${DET_WEIGHT}"
    --det-pos-weight "${DET_POS_WEIGHT}"
    $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" )
    $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" )
    --det-sigma "${DET_SIGMA}"
    --head-lr "${HEAD_LR}"
    $( [[ "$USE_FOCAL_HEATMAP" -eq 1 ]] && echo "--use-focal-heatmap" )
    --focal-alpha "${FOCAL_ALPHA}"
    --focal-gamma "${FOCAL_GAMMA}"
    --det-neg-topk-ratio "${DET_NEG_TOPK_RATIO}"
    $( [[ "$USE_IOU_SIZE" -eq 1 ]] && echo "--use-iou-size" )
    --iou-weight "${IOU_WEIGHT}"
    $( [[ -n "$EVAL_NMS" ]] && echo "--eval-nms $EVAL_NMS" )
    --eval-nms-radius "${EVAL_NMS_RADIUS}"
    --eval-soft-nms-sigma "${EVAL_SOFT_NMS_SIGMA}"
    --wot "${WOT}"
    --wtv "${WTV}"
    --reg "${REG}"
    --num-of-iter-in-ot "${NUM_OF_ITER_IN_OT}"
    --norm-cood "${NORM_COOD}"
    --wrd "${WRD}"
    --device "${DEVICE}"
    --local_rank 0)
else
  # Multi-GPU mode with torchrun
  export TORCH_DISTRIBUTED_TIMEOUT="${DDP_TIMEOUT}"
  CMD=(torchrun --nproc_per_node="${NPROC}"
    Fine-tune/train.py
    --data-dir "${DATA_DIR}"
    --save-dir "${SAVE_DIR}"
    --resume "${RESUME}"
    --task "${TASK}"
    $( [[ "$FREEZE_BACKBONE" -eq 1 ]] && echo "--freeze-backbone" )
    $( [[ "$FREEZE_COUNTER" -eq 1 ]] && echo "--freeze-counter" )
    $( [[ "$FREEZE_UNET" -eq 1 ]] && echo "--freeze-unet" )
    --unfreeze-epoch "${UNFREEZE_EPOCH}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --batch-size "${BATCH_SIZE}"
    --max-epoch "${MAX_EPOCH}"
    --crop-size "${CROP_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --ap-dist-thresh "${AP_DIST_THRESH}"
    --downsample-ratio "${DOWNSAMPLE_RATIO}"
    --output-stride "${OUTPUT_STRIDE}"
    --det-patience "${DET_PATIENCE}"
    --val-epoch "${VAL_EPOCH}"
    --val-start "${VAL_START}"
    --max-model-num "${MAX_MODEL_NUM}"
    $( [[ "$SAVE_ALL_BEST" -eq 1 ]] && echo "--save-all-best" )
    --head-conv "${HEAD_CONV}"
    $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" )
    --nms-kernel "${NMS_KERNEL}"
    $( [[ "$USE_DET_ADAPTOR" -eq 1 ]] && echo "--use-det-adaptor" )
    $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" )
    $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" )
    --fixed-box-size "${FIXED_BOX_SIZE}"
    --det-weight "${DET_WEIGHT}"
    --det-pos-weight "${DET_POS_WEIGHT}"
    $( [[ "$USE_BCE_LOGITS" -eq 1 ]] && echo "--use-bce-logits" )
    $( [[ "$DET_USE_GN" -eq 1 ]] && echo "--det-use-gn" )
    --det-sigma "${DET_SIGMA}"
    --head-lr "${HEAD_LR}"
    $( [[ "$USE_FOCAL_HEATMAP" -eq 1 ]] && echo "--use-focal-heatmap" )
    --focal-alpha "${FOCAL_ALPHA}"
    --focal-gamma "${FOCAL_GAMMA}"
    --det-neg-topk-ratio "${DET_NEG_TOPK_RATIO}"
    $( [[ "$USE_IOU_SIZE" -eq 1 ]] && echo "--use-iou-size" )
    --iou-weight "${IOU_WEIGHT}"
    $( [[ -n "$EVAL_NMS" ]] && echo "--eval-nms $EVAL_NMS" )
    --eval-nms-radius "${EVAL_NMS_RADIUS}"
    --eval-soft-nms-sigma "${EVAL_SOFT_NMS_SIGMA}"
    --wot "${WOT}"
    --wtv "${WTV}"
    --reg "${REG}"
    --num-of-iter-in-ot "${NUM_OF_ITER_IN_OT}"
    --norm-cood "${NORM_COOD}"
    --wrd "${WRD}"
    --device "${DEVICE}")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  CMD+=( $EXTRA_ARGS )
fi

echo "Command: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
