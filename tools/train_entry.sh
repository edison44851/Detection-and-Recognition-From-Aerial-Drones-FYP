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

# --- Data & Paths ---
DATA_DIR=".data/DroneRGBT_converted"
SAVE_DIR="./checkpoints_phase6"
RESUME=".weights/drone_rgbt_best_494_781.pth"

# --- Device & Distributed ---
NPROC=2
DEVICE="1,2"
NUM_WORKERS=1
DDP_TIMEOUT=3600  # DDP watchdog timeout in seconds (default: 1800=30min, set to 3600=1hr)

# --- Task Selection & Freezing ---
TASK="detection"
FREEZE_BACKBONE=1
FREEZE_COUNTER=1
FREEZE_UNET=1
UNFREEZE_EPOCH=-1

# --- Training General ---
CROP_SIZE=224
BATCH_SIZE=4
LR=1e-5
WEIGHT_DECAY=0.0001
MAX_EPOCH=300
VAL_EPOCH=2
VAL_START=10
MAX_MODEL_NUM=1

# --- Detection Architecture ---
DOWNSAMPLE_RATIO=4
OUTPUT_STRIDE=4
HEAD_CONV=256
USE_DECONV=1
NMS_KERNEL=3
USE_FPN=1
USE_DET_ADAPTOR=1
DET_USE_GN=1

# --- Keypoint Mode (Phase 1) ---
KEYPOINT_MODE=1

# --- Detection Loss & Heatmap ---
DET_WEIGHT=1.0
DET_POS_WEIGHT=12.0
DET_SIGMA=0.8
USE_BCE_LOGITS=1
DET_NEG_TOPK_RATIO=0.25
BG_SUPPRESSION_WEIGHT=0.03  # Reduced from 0.1 to allow more predictions (was too aggressive)
LABEL_SMOOTHING=0.0         # Disabled (was 0.05, made model under-confident)

# --- Focal Loss (Detection) ---
USE_FOCAL_HEATMAP=1
FOCAL_ALPHA=0.85
FOCAL_GAMMA=2.5

# --- Size Loss & IoU ---
USE_IOU_SIZE=1
IOU_WEIGHT=0.3

# --- Detection Thresholds (Eval) ---
DET_SCORE_THRESHOLD=0.3

# --- Detection Head Learning Rate ---
HEAD_LR=0.003

# --- Evaluation & Early Stopping ---
AP_DIST_THRESH=8.0
EVAL_NMS="radius"
EVAL_NMS_RADIUS=4.0
EVAL_SOFT_NMS_SIGMA=0.5
DET_PATIENCE=10

# --- Data Augmentation (Detection) ---
AUG_SCALE_MIN=0.7
AUG_SCALE_MAX=1.5
AUG_FLIP=1
AUG_CROP_SIZE=224

# --- Thermal Preprocessing (Phase B) ---
THERMAL_CLAHE=1
THERMAL_CLAHE_CLIP=2.0

# --- Counting Task Parameters (DM-Count) ---
WOT=0.1
WTV=0.01
WRD=0.1
REG=10.0
NUM_OF_ITER_IN_OT=100
NORM_COOD=0

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

DATA & PATHS:
  --save-dir PATH              Output checkpoint dir
  --resume PATH                Resume checkpoint

DEVICE & DISTRIBUTED:
  --nproc N                    Number of processes per node (default: 1)
  --device DEVICES             GPU device IDs (default: 0)
  --num-workers N              Data loading workers (default: 8)

TASK SELECTION & FREEZING:
  --task TASK                  Task: detection|counting (default: detection)
  --freeze-backbone            Freeze Swin backbone (default: yes)
  --freeze-counter             Freeze counting head (default: yes)
  --freeze-unet                Freeze U-Net (default: yes)
  --unfreeze-epoch N           Epoch to unfreeze backbone (default: -1)

TRAINING GENERAL:
  --crop-size N                Crop size (default: 224)
  --batch-size N               Batch size (default: 4)
  --lr LR                      Learning rate (default: 1e-5)
  --weight-decay W             Weight decay (default: 1e-4)
  --max-epoch N                Max epochs (default: 100)
  --val-epoch N                Validation frequency (default: 2)
  --val-start N                Start validation at epoch (default: 10)
  --max-model-num N            Max checkpoints to keep (default: 1)

DETECTION ARCHITECTURE:
  --downsample-ratio N         Downsample ratio (default: 8)
  --output-stride N            Output stride (default: 4)
  --head-conv N                Head conv channels (default: 256)
  --use-deconv                 Use deconv upsampling (default: yes)
  --nms-kernel N               NMS kernel size (default: 3)
  --use-fpn                    Enable FPN neck (default: yes)
  --use-det-adaptor            Use detection adaptor (default: yes)
  --det-use-gn                 Use GroupNorm in head (default: yes)

KEYPOINT MODE (PHASE 1):
  --keypoint-mode              Enable keypoint-only mode (default: yes)

DETECTION LOSS & HEATMAP:
  --det-weight W               Detection loss weight (default: 0.1)
  --det-pos-weight W           Positive heatmap weight (default: 7.0)
  --det-sigma S                Gaussian sigma for heatmaps (default: 0.8)
  --use-bce-logits             Use BCEWithLogitsLoss (default: yes)
  --det-neg-topk-ratio R       Hard negative ratio (default: 0.1)

FOCAL LOSS (DETECTION):
  --use-focal-heatmap          Enable focal loss (default: yes)
  --focal-alpha A              Focal loss alpha (default: 0.75)
  --focal-gamma G              Focal loss gamma (default: 1.5)

SIZE LOSS & IoU:
  --use-iou-size               Enable IoU size loss (default: yes)
  --iou-weight W               IoU loss weight (default: 0.3)

DETECTION THRESHOLDS (EVAL):
  --det-score-threshold S      Base detection score threshold (default: 0.3)

DETECTION HEAD LEARNING RATE:
  --head-lr LR                 Detection head learning rate (default: 0.0002)

EVALUATION & EARLY STOPPING:
  --ap-dist-thresh DIST        AP distance threshold in pixels (default: 8.0)
  --eval-nms TYPE              NMS type: radius|soft (default: radius)
  --eval-nms-radius R          Radius NMS radius (default: 2.0)
  --eval-soft-nms-sigma S      Soft-NMS sigma (default: 0.5)
  --det-patience N             Early stopping patience (default: 10)

DATA AUGMENTATION (DETECTION):
  --aug-scale-min S            Minimum scale factor (default: 0.5)
  --aug-scale-max S            Maximum scale factor (default: 2.0)
  --aug-flip                   Enable horizontal flip (default: yes)
  --aug-crop-size N            Random crop size (default: 224, 0 to disable)

THERMAL PREPROCESSING (PHASE B):
  --thermal-clahe              Enable CLAHE enhancement (default: yes)
  --thermal-clahe-clip C       CLAHE clip limit (default: 2.0)

COUNTING (DM-COUNT):
  --wot W                      OT loss weight (default: 0.1)
  --wtv W                      TV loss weight (default: 0.01)
  --wrd W                      RD loss weight (default: 0.1)
  --reg R                      Sinkhorn regularization (default: 10.0)
  --num-of-iter-in-ot N        OT iterations (default: 100)
  --norm-cood N                Normalize coordinates (default: 0)

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
    # Data & Paths
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --save-dir)
      SAVE_DIR="$2"; shift 2;;
    --resume)
      RESUME="$2"; shift 2;;
    
    # Device & Distributed
    --nproc)
      NPROC="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2;;
    
    # Task & Freezing
    --task)
      TASK="$2"; shift 2;;
    --freeze-backbone)
      FREEZE_BACKBONE=1; shift 1;;
    --freeze-counter)
      FREEZE_COUNTER=1; shift 1;;
    --freeze-unet)
      FREEZE_UNET=1; shift 1;;
    --unfreeze-epoch)
      UNFREEZE_EPOCH="$2"; shift 2;;
    
    # Training General
    --crop-size)
      CROP_SIZE="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --lr)
      LR="$2"; shift 2;;
    --weight-decay)
      WEIGHT_DECAY="$2"; shift 2;;
    --max-epoch)
      MAX_EPOCH="$2"; shift 2;;
    --val-epoch)
      VAL_EPOCH="$2"; shift 2;;
    --val-start)
      VAL_START="$2"; shift 2;;
    --max-model-num)
      MAX_MODEL_NUM="$2"; shift 2;;
    
    # Detection Architecture
    --downsample-ratio)
      DOWNSAMPLE_RATIO="$2"; shift 2;;
    --output-stride)
      OUTPUT_STRIDE="$2"; shift 2;;
    --head-conv)
      HEAD_CONV="$2"; shift 2;;
    --use-deconv)
      USE_DECONV=1; shift 1;;
    --nms-kernel)
      NMS_KERNEL="$2"; shift 2;;
    --use-fpn)
      USE_FPN=1; shift 1;;
    --use-det-adaptor)
      USE_DET_ADAPTOR=1; shift 1;;
    --det-use-gn)
      DET_USE_GN=1; shift 1;;
    
    # Keypoint Mode (Phase 1)
    --keypoint-mode)
      KEYPOINT_MODE=1; shift 1;;
    
    # Detection Loss & Heatmap
    --det-weight)
      DET_WEIGHT="$2"; shift 2;;
    --det-pos-weight)
      DET_POS_WEIGHT="$2"; shift 2;;
    --det-sigma)
      DET_SIGMA="$2"; shift 2;;
    --use-bce-logits)
      USE_BCE_LOGITS=1; shift 1;;
    --det-neg-topk-ratio)
      DET_NEG_TOPK_RATIO="$2"; shift 2;;
    
    # Focal Loss (Detection)
    --use-focal-heatmap)
      USE_FOCAL_HEATMAP=1; shift 1;;
    --focal-alpha)
      FOCAL_ALPHA="$2"; shift 2;;
    --focal-gamma)
      FOCAL_GAMMA="$2"; shift 2;;
    
    # Size Loss & IoU
    --use-iou-size)
      USE_IOU_SIZE=1; shift 1;;
    --iou-weight)
      IOU_WEIGHT="$2"; shift 2;;
    
    # Detection Thresholds (Eval)
    --det-score-threshold)
      DET_SCORE_THRESHOLD="$2"; shift 2;;
    
    # Detection Head Learning Rate
    --head-lr)
      HEAD_LR="$2"; shift 2;;
    
    # Evaluation & Early Stopping
    --ap-dist-thresh)
      AP_DIST_THRESH="$2"; shift 2;;
    --eval-nms)
      EVAL_NMS="$2"; shift 2;;
    --eval-nms-radius)
      EVAL_NMS_RADIUS="$2"; shift 2;;
    --eval-soft-nms-sigma)
      EVAL_SOFT_NMS_SIGMA="$2"; shift 2;;
    --det-patience)
      DET_PATIENCE="$2"; shift 2;;
    
    # Data Augmentation (Detection)
    --aug-scale-min)
      AUG_SCALE_MIN="$2"; shift 2;;
    --aug-scale-max)
      AUG_SCALE_MAX="$2"; shift 2;;
    --aug-flip)
      AUG_FLIP=1; shift 1;;
    --aug-crop-size)
      AUG_CROP_SIZE="$2"; shift 2;;
    
    # Thermal Preprocessing (Phase B)
    --thermal-clahe)
      THERMAL_CLAHE=1; shift 1;;
    --thermal-clahe-clip)
      THERMAL_CLAHE_CLIP="$2"; shift 2;;
    
    # Counting (DM-Count)
    --wot)
      WOT="$2"; shift 2;;
    --wtv)
      WTV="$2"; shift 2;;
    --wrd)
      WRD="$2"; shift 2;;
    --reg)
      REG="$2"; shift 2;;
    --num-of-iter-in-ot)
      NUM_OF_ITER_IN_OT="$2"; shift 2;;
    --norm-cood)
      NORM_COOD="$2"; shift 2;;
    
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
echo "--- Data & Paths ---"
echo "  data-dir:     $DATA_DIR"
echo "  save-dir:     $SAVE_DIR"
echo "  resume:       $RESUME"
echo ""
echo "--- Device & Distributed ---"
echo "  nproc:        $NPROC"
echo "  device:       $DEVICE"
echo "  num-workers:  $NUM_WORKERS"
echo "  ddp-timeout:  ${DDP_TIMEOUT}s"
echo ""
echo "--- Task & Freezing ---"
echo "  task:              $TASK"
echo "  freeze-backbone:   $FREEZE_BACKBONE"
echo "  freeze-counter:    $FREEZE_COUNTER"
echo "  freeze-unet:       $FREEZE_UNET"
echo "  unfreeze-epoch:    $UNFREEZE_EPOCH"
echo ""
echo "--- Training General ---"
echo "  crop-size:     $CROP_SIZE"
echo "  batch-size:    $BATCH_SIZE"
echo "  lr:            $LR"
echo "  weight-decay:  $WEIGHT_DECAY"
echo "  max-epoch:     $MAX_EPOCH"
echo "  val-epoch:     $VAL_EPOCH"
echo "  val-start:     $VAL_START"
echo "  max-model-num: $MAX_MODEL_NUM"
echo ""
echo "--- Detection Architecture ---"
echo "  downsample-ratio: $DOWNSAMPLE_RATIO"
echo "  output-stride:    $OUTPUT_STRIDE"
echo "  head-conv:        $HEAD_CONV"
echo "  use-deconv:       $USE_DECONV"
echo "  nms-kernel:       $NMS_KERNEL"
echo "  use-fpn:          $USE_FPN"
echo "  use-det-adaptor:  $USE_DET_ADAPTOR"
echo "  det-use-gn:       $DET_USE_GN"
echo ""
echo "--- Keypoint Mode (Phase 1) ---"
echo "  keypoint-mode: $KEYPOINT_MODE"
echo ""
echo "--- Detection Loss & Heatmap ---"
echo "  det-weight:         $DET_WEIGHT"
echo "  det-pos-weight:     $DET_POS_WEIGHT"
echo "  det-sigma:          $DET_SIGMA"
echo "  use-bce-logits:     $USE_BCE_LOGITS"
echo "  det-neg-topk-ratio: $DET_NEG_TOPK_RATIO"
echo ""
echo "--- Focal Loss (Detection) ---"
echo "  use-focal-heatmap: $USE_FOCAL_HEATMAP"
echo "  focal-alpha:       $FOCAL_ALPHA"
echo "  focal-gamma:       $FOCAL_GAMMA"
echo ""
echo "--- Size Loss & IoU ---"
echo "  use-iou-size: $USE_IOU_SIZE"
echo "  iou-weight:   $IOU_WEIGHT"
echo ""
echo "--- Detection Thresholds (Eval) ---"
echo "  det-score-threshold: $DET_SCORE_THRESHOLD"
echo ""
echo "--- Detection Head Learning Rate ---"
echo "  head-lr: $HEAD_LR"
echo ""
echo "--- Evaluation & Early Stopping ---"
echo "  ap-dist-thresh:    $AP_DIST_THRESH"
echo "  eval-nms:          $EVAL_NMS"
echo "  eval-nms-radius:   $EVAL_NMS_RADIUS"
echo "  eval-soft-nms-sigma: $EVAL_SOFT_NMS_SIGMA"
echo "  det-patience:      $DET_PATIENCE"
echo ""
echo "--- Data Augmentation (Detection) ---"
echo "  aug-scale-min: $AUG_SCALE_MIN"
echo "  aug-scale-max: $AUG_SCALE_MAX"
echo "  aug-flip:      $AUG_FLIP"
echo "  aug-crop-size: $AUG_CROP_SIZE"
echo ""
echo "--- Thermal Preprocessing (Phase B) ---"
echo "  thermal-clahe:       $THERMAL_CLAHE"
echo "  thermal-clahe-clip:  $THERMAL_CLAHE_CLIP"
echo ""
echo "--- Counting (DM-Count) ---"
echo "  wot:  $WOT"
echo "  wtv:  $WTV"
echo "  wrd:  $WRD"
echo "  reg:  $REG"
echo "  num-of-iter-in-ot: $NUM_OF_ITER_IN_OT"
echo "  norm-cood:         $NORM_COOD"
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
    --det-score-threshold "${DET_SCORE_THRESHOLD}"
    --head-conv "${HEAD_CONV}"
    $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" )
    --nms-kernel "${NMS_KERNEL}"
    $( [[ "$USE_DET_ADAPTOR" -eq 1 ]] && echo "--use-det-adaptor" )
    $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" )
    $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" )
    --aug-scale-min "${AUG_SCALE_MIN}"
    --aug-scale-max "${AUG_SCALE_MAX}"
    $( [[ "$AUG_FLIP" -eq 1 ]] && echo "--aug-flip" )
    --aug-crop-size "${AUG_CROP_SIZE}"
    $( [[ "$THERMAL_CLAHE" -eq 1 ]] && echo "--thermal-clahe" )
    --thermal-clahe-clip "${THERMAL_CLAHE_CLIP}"
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
    --bg-suppression-weight "${BG_SUPPRESSION_WEIGHT}"
    --label-smoothing "${LABEL_SMOOTHING}"
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
    --det-score-threshold "${DET_SCORE_THRESHOLD}"
    --head-conv "${HEAD_CONV}"
    $( [[ "$USE_DECONV" -eq 1 ]] && echo "--use-deconv" )
    --nms-kernel "${NMS_KERNEL}"
    $( [[ "$USE_DET_ADAPTOR" -eq 1 ]] && echo "--use-det-adaptor" )
    $( [[ "$USE_FPN" -eq 1 ]] && echo "--use-fpn" )
    $( [[ "$KEYPOINT_MODE" -eq 1 ]] && echo "--keypoint-mode" )
    --aug-scale-min "${AUG_SCALE_MIN}"
    --aug-scale-max "${AUG_SCALE_MAX}"
    $( [[ "$AUG_FLIP" -eq 1 ]] && echo "--aug-flip" )
    --aug-crop-size "${AUG_CROP_SIZE}"
    $( [[ "$THERMAL_CLAHE" -eq 1 ]] && echo "--thermal-clahe" )
    --thermal-clahe-clip "${THERMAL_CLAHE_CLIP}"
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
    --bg-suppression-weight "${BG_SUPPRESSION_WEIGHT}"
    --label-smoothing "${LABEL_SMOOTHING}"
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
