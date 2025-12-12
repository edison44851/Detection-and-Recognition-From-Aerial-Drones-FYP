#!/usr/bin/env bash
set -euo pipefail

# Generic grid runner for post-training diagnostics
# - Dynamically builds loops for any parameters you define below
# - If a parameter is defined as a list (multiple values), it will be swept
# - If a parameter is a boolean (store-true) flag, define values as (1 0) to test on/off
# - You also provide a short alias for directory naming for each parameter
# - Uses Fine-tune/test_detection_vis.py directly; does NOT modify your diagnostics wrapper
#
# Usage:
#   bash tools/run_posttrain_grid.sh [CKPT] [DATA_DIR] [OUT_ROOT] [NUM] [DOWNSAMPLE]
#
# Example:
#   bash tools/run_posttrain_grid.sh \
#     checkpoints/1201-215341/best_model.pth \
#     .data/DroneRGBT_converted \
#     ./.tmp_posttrain/grid_1201-215341 \
#     64 8


# ------------------------------
# Model architecture parameters (set per stage as needed)
# These can be set at the top for each run or stage
# Example: head_conv=256, use_deconv=1, use_fpn=0, keypoint_mode=0

HEAD_CONV=256
USE_DECONV=0
USE_FPN=0
KEYPOINT_MODE=0

# Global min_score for all modes (set to lowest value you want to allow)
MIN_SCORE=0.01

# Performance parameters (apply to all modes)
NUM_VIS=64
BATCH_SIZE=8
NUM_WORKERS=4

CKPT=${1:-checkpoints/1130-145629_shallow_centerhead/best_model.pth}
DATA_DIR=${2:-.data/DroneRGBT_converted}
OUT_ROOT=${3:-./.tmp_posttrain/grid_1130-145629_shallow_centerhead}
NUM=${4:-10000}
DOWNSAMPLE=${5:-4}

mkdir -p "$OUT_ROOT"

echo "Grid diagnostics (generic)"
echo "  ckpt: $CKPT"
echo "  data: $DATA_DIR"
echo "  out:  $OUT_ROOT"

# ------------------------------
# Parameter registry
# ------------------------------
# Declare mappings once. Add new params here with their CLI flag, alias, and type.
# TYPE: scalar | bool (bool = store-true flag, included only when value==1)
declare -A PARAM_FLAG PARAM_ALIAS PARAM_TYPE


# --- Parameter registry: update to match test_detection_vis.py flags ---
PARAM_FLAG[score_thresh]=--score-thresh
PARAM_ALIAS[score_thresh]=st
PARAM_TYPE[score_thresh]=scalar

PARAM_FLAG[min_score]=--min-score
PARAM_ALIAS[min_score]=ms
PARAM_TYPE[min_score]=scalar

PARAM_FLAG[nms_radius]=--nms-radius
PARAM_ALIAS[nms_radius]=r
PARAM_TYPE[nms_radius]=scalar

PARAM_FLAG[soft_nms_sigma]=--soft-nms-sigma
PARAM_ALIAS[soft_nms_sigma]=sn
PARAM_TYPE[soft_nms_sigma]=scalar

PARAM_FLAG[max_dets]=--max-dets
PARAM_ALIAS[max_dets]=k
PARAM_TYPE[max_dets]=scalar

PARAM_FLAG[tile_size]=--tile-size
PARAM_ALIAS[tile_size]=ts
PARAM_TYPE[tile_size]=scalar

PARAM_FLAG[tile_overlap]=--tile-overlap
PARAM_ALIAS[tile_overlap]=ov
PARAM_TYPE[tile_overlap]=scalar

PARAM_FLAG[indices_file]=--indices-file
PARAM_ALIAS[indices_file]=idx
PARAM_TYPE[indices_file]=scalar

PARAM_FLAG[scores_csv]=--scores-csv
PARAM_ALIAS[scores_csv]=csv
PARAM_TYPE[scores_csv]=scalar

PARAM_FLAG[scores_hist]=--scores-hist
PARAM_ALIAS[scores_hist]=hist
PARAM_TYPE[scores_hist]=scalar

# ------------------------------
# Mode-specific configuration
# ------------------------------
# Define which parameters to sweep per mode and their values.
# To sweep, list multiple values. For fixed, list a single value.

# RAW mode: sweep score_thresh, min_score is global
RAW_PARAMS=(score_thresh max_dets)
RAW_values_score_thresh=(0.01 0.05 0.10 0.15)
RAW_values_max_dets=(1000)

# TILES mode: sweep score_thresh, min_score is global
TILES_PARAMS=(score_thresh nms_radius soft_nms_sigma max_dets tile_size tile_overlap)
TILES_values_score_thresh=(0.20 0.25 0.30)
TILES_values_nms_radius=(4)
TILES_values_soft_nms_sigma=()
TILES_values_max_dets=(150)
TILES_values_tile_size=(512)
TILES_values_tile_overlap=(0.15 0.20 0.25)

# ORIG mode: sweep score_thresh, min_score is global
ORIG_PARAMS=(score_thresh nms_radius max_dets)
ORIG_values_score_thresh=(0.20 0.25 0.30)
ORIG_values_nms_radius=(4)
ORIG_values_soft_nms_sigma=()
ORIG_values_max_dets=(150)


# No soft_nms_sigma in ORIG for now
# ------------------------------
# Helpers
# ------------------------------
sanitize_val() {
  # leave decimals as-is; sufficient for directory names
  local v="$1"
  echo "$v"
}

# get values array for mode+param into a nameref 'out'
get_values() {
  local mode="$1"; local param="$2"; local __out="$3"
  local var_name="${mode}_values_${param}"
  # If array not defined, return empty
  local -n out_ref="$__out"
  if ! declare -p "$var_name" &>/dev/null; then
    out_ref=()
    return
  fi
  # Nameref to source array and copy
  local -n src_ref="$var_name"
  out_ref=("${src_ref[@]}")
}

# Build command args and directory suffix for a given param/value
build_arg_and_suffix() {
  local param="$1"; shift
  local val="$1"; shift
  local flag="${PARAM_FLAG[$param]}"
  local alias="${PARAM_ALIAS[$param]}"
  local ptype="${PARAM_TYPE[$param]:-scalar}"
  local args=""; local suffix=""
  if [[ "$ptype" == "bool" ]]; then
    if [[ "$val" == "1" ]]; then
      args+=" $flag"
      suffix+="_${alias}"
    fi
  else
    # skip if value list is empty marker
    if [[ -n "$val" ]]; then
      args+=" $flag $val"
      suffix+="_${alias}_$(sanitize_val "$val")"
    fi
  fi
  echo "$args|||$suffix"
}

# Recursive combo builder
recurse_combos() {
  local mode="$1"; shift
  local params_name="$1"; shift
  local -n params_ref="$params_name"
  local idx="$1"; shift
  local accum_args="$1"; shift
  local accum_suffix="$1"; shift
  local out_dir_base="$1"; shift
  local extra_fixed_args="$1"; shift
  local indices_file="$1"; shift

  if (( idx >= ${#params_ref[@]} )); then
    local out_dir="${out_dir_base}${accum_suffix}"
    mkdir -p "$out_dir"
    echo "\n[${mode^^}] args=$accum_args"
    # Build python command
    local cmd=( python3 Fine-tune/test_detection_vis.py
      --data-dir "$DATA_DIR"
      --ckpt "$CKPT"
      --out "$out_dir"
      --num "$NUM" --downsample-ratio "$DOWNSAMPLE"
      --ap-dist-thresh 8.0
      --min-score "$MIN_SCORE"
      --num-vis "$NUM_VIS"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
    )
    # add extra fixed args first (including keypoint mode if enabled)
    if [[ -n "$extra_fixed_args" ]]; then
      # shellcheck disable=SC2206
      cmd+=( $extra_fixed_args )
    fi
    # Add head architecture flags (should match training config)
    if [[ -n "$HEAD_CONV" ]]; then
      cmd+=( --head-conv "$HEAD_CONV" )
    fi
    if [[ "$USE_DECONV" == "1" ]]; then
      cmd+=( --use-deconv )
    fi
    if [[ "$USE_FPN" == "1" ]]; then
      cmd+=( --use-fpn )
    fi
    if [[ "$KEYPOINT_MODE" == "1" ]]; then
      cmd+=( --keypoint-mode )
    fi
    # add accumulated sweep args
    if [[ -n "$accum_args" ]]; then
      # shellcheck disable=SC2206
      cmd+=( $accum_args )
    fi
    # indices if provided
    if [[ -n "${indices_file:-}" ]]; then
      cmd+=( --indices-file "$indices_file" )
    fi
    # always save diagnostics artifacts
    cmd+=( --scores-csv scores.csv --scores-hist scores.png )
    # run
    "${cmd[@]}"
    # if no base indices set yet, capture from this run (for RAW)
    if [[ -z "$BASE_INDICES_FILE" && -f "$out_dir/selected_indices.txt" ]]; then
      BASE_INDICES_FILE="$out_dir/selected_indices.txt"
    fi
    return
  fi

  local param="${params_ref[$idx]}"
  local values=()
  get_values "$mode" "$param" values
  # if empty values, skip this param
  if (( ${#values[@]} == 0 )); then
    recurse_combos "$mode" "$params_name" "$((idx+1))" "$accum_args" "$accum_suffix" "$out_dir_base" "$extra_fixed_args" "$indices_file"
    return
  fi
  for v in "${values[@]}"; do
    local built; built=$(build_arg_and_suffix "$param" "$v")
    local arg_part="${built%%\|\|\|*}"
    local suf_part="${built#*|||}"
    recurse_combos "$mode" "$params_name" "$((idx+1))" "$accum_args $arg_part" "$accum_suffix$suf_part" "$out_dir_base" "$extra_fixed_args" "$indices_file"
  done
}

# Build space-separated fixed args string from single-valued params subset
build_fixed_args() {
  local mode="$1"; shift
  local -n params_ref="$1"; shift
  local fixed_subset=("$@")
  local args=""
  for p in "${fixed_subset[@]}"; do
    local values=()
    get_values "$mode" "$p" values
    if (( ${#values[@]} >= 1 )); then
      local v="${values[0]}"
      local ptype="${PARAM_TYPE[$p]:-scalar}"
      if [[ "$ptype" == "bool" ]]; then
        if [[ "$v" == "1" ]]; then
          args+=" ${PARAM_FLAG[$p]}"
        fi
      else
        args+=" ${PARAM_FLAG[$p]} \"$v\""
      fi
    fi
  done
  echo "$args"
}

# ------------------------------
# Run grids per mode
# ------------------------------
BASE_INDICES_FILE=""

# RAW
{
  mode="RAW"; mode_lc="RAW" # names for get_values use RAW_values_
  out_base="$OUT_ROOT/raw"
  mkdir -p "$out_base"
  # fixed args: disable NMS for raw predictions
  fixed_args="--no-nms"
  recurse_combos "RAW" RAW_PARAMS 0 "" "" "$out_base" "$fixed_args" ""
}

# TILES (reuse indices if available)
{
  mode="TILES"; out_base="$OUT_ROOT/tiles"
  mkdir -p "$out_base"
  fixed_args=""
  recurse_combos "TILES" TILES_PARAMS 0 "" "" "$out_base" "$fixed_args" "${BASE_INDICES_FILE:-}"
}

# ORIG (reuse indices if available)
{
  mode="ORIG"; out_base="$OUT_ROOT/orig"
  mkdir -p "$out_base"
  fixed_args=""
  recurse_combos "ORIG" ORIG_PARAMS 0 "" "" "$out_base" "$fixed_args" "${BASE_INDICES_FILE:-}"
}

echo "\nGrid diagnostics finished. Outputs in: $OUT_ROOT"
echo "  raw:   $OUT_ROOT/raw"
echo "  tiles: $OUT_ROOT/tiles"
echo "  orig:  $OUT_ROOT/orig"

exit 0
