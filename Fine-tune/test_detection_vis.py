#!/usr/bin/env python3
"""Run detection inference on several random samples and save visualization images.

Saves visualizations to `./visuals_detection` by default. Uses the dataset loader
`DetectionDataset` and expects a model checkpoint path. The script will
load the checkpoint into `DetectionModel` if available, otherwise try to load into
`Swin_BM_RGBT` as a fallback.

Usage:
  python Fine-tune/test_detection_vis.py --data-dir .data/DroneRGBT_counting --ckpt checkpoints/1122-222336/best_model.pth --out visuals_detection --num 8

"""
# os not required
import argparse
import random
from pathlib import Path
from typing import cast
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import logging

import csv
import matplotlib.pyplot as plt
import cv2
from utils.detection_eval import compute_ap

from datasets.dm_detection import DetectionDataset
from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.det_model import DetectionHeadWrapper
from utils.detection_eval import heatmap_peaks


RGB_NORM_MEAN = np.array([0.407, 0.389, 0.396], dtype=np.float32)
RGB_NORM_STD = np.array([0.241, 0.246, 0.242], dtype=np.float32)
T_NORM_MEAN = np.array([0.499, 0.168, 0.431], dtype=np.float32)
T_NORM_STD = np.array([0.308, 0.168, 0.181], dtype=np.float32)


def collate_fn(batch):
    """Custom collate function to handle variable-length ground truth points."""
    rgb = torch.stack([item['rgb'] for item in batch])
    t = torch.stack([item['t'] for item in batch])
    # Keep points as list (variable length)
    points = [item.get('points', torch.zeros((0, 2))) for item in batch]
    ids = [item.get('id', f'sample_{i}') for i, item in enumerate(batch)]
    return {'rgb': rgb, 't': t, 'points': points, 'id': ids}


def nms_radius(preds, radius):
    """Radius-based NMS. preds: list of (x,y,score). Returns filtered list."""
    if len(preds) == 0:
        return []
    pts = sorted(preds, key=lambda x: x[2], reverse=True)
    keep = []
    taken = [False] * len(pts)
    for i, (x, y, s) in enumerate(pts):
        if taken[i]:
            continue
        keep.append((x, y, s))
        # suppress others within radius
        for j in range(i + 1, len(pts)):
            if taken[j]:
                continue
            x2, y2, _ = pts[j]
            if (x - x2) ** 2 + (y - y2) ** 2 <= radius * radius:
                taken[j] = True
    return keep


def soft_nms_gaussian(preds, sigma, score_thresh=None):
    """Soft-NMS (Gaussian). preds: list of (x,y,score).
    Returns filtered list with scores decayed by proximity.
    """
    if len(preds) == 0:
        return []
    # work on a copy sorted by score
    pts = sorted([(float(x), float(y), float(s)) for x, y, s in preds], key=lambda x: x[2], reverse=True)
    keep = []
    for i in range(len(pts)):
        xi, yi, si = pts[i]
        if si <= 0:
            continue
        # decay others
        for j in range(i + 1, len(pts)):
            xj, yj, sj = pts[j]
            if sj <= 0:
                continue
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            # Gaussian decay
            decay = np.exp(-d2 / (2.0 * (sigma ** 2)))
            pts[j] = (xj, yj, sj * (1.0 - decay))
        # apply score threshold if provided
        if score_thresh is None or si >= score_thresh:
            keep.append((xi, yi, si))
    # final thresholding
    if score_thresh is not None:
        keep = [p for p in keep if p[2] >= score_thresh]
    return keep


def vis_on_image(img_rgb, img_t, preds_px, downsample_ratio, save_path):
    # img_rgb, img_t are HxWx3 uint8
    vis = img_rgb.copy()
    # draw predicted centers on RGB and thermal side-by-side
    h, w = vis.shape[:2]
    sep = 8
    canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)
    canvas[:, :w] = vis
    canvas[:, w + sep: w * 2 + sep] = img_t

    # draw GT points (blue X) and predicted centers
    # preds_px is list of tuples (cx, cy, score, matched_flag?) - matched_flag optional
    for p in preds_px:
        if len(p) == 4:
            cx, cy, score, matched = p
        else:
            cx, cy, score = p
            matched = None
        ix = int(round(cx))
        iy = int(round(cy))
        # color: matched TP -> green, FP -> red, unknown -> yellow
        if matched is True:
            color = (0, 255, 0)
        elif matched is False:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        # draw on RGB half
        cv2.circle(canvas, (ix, iy), 6, color, 2)
        cv2.putText(canvas, f"{score:.2f}", (ix + 6, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # draw on thermal half
        cv2.circle(canvas, (w + sep + ix, iy), 6, color, 2)

    if save_path is not None:
        cv2.imwrite(str(save_path), canvas)
    return canvas


def preprocess_image(tensor_img, mean=None, std=None, to_bgr=True):
    """Convert a tensor image to uint8 for OpenCV visualization.

    If mean/std are provided, this function de-normalizes first. Output is BGR
    by default so OpenCV drawing and imwrite preserve expected colors.
    """
    if isinstance(tensor_img, torch.Tensor):
        if tensor_img.dim() == 4:
            tensor_img = tensor_img[0]
        arr = tensor_img.detach().cpu().float().numpy()
        arr = np.transpose(arr, (1, 2, 0))

        if mean is not None and std is not None and arr.shape[2] == len(mean):
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
            std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
            arr = arr * std_arr + mean_arr
        elif np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if to_bgr and arr.shape[2] >= 3:
            arr = arr[:, :, :3][:, :, ::-1].copy()
        return arr

    arr = np.asarray(tensor_img)
    if to_bgr and arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3][:, :, ::-1].copy()
    return arr


def infer_and_visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio)
    N = len(ds)
    
    # Use DataLoader for parallel loading and batching with custom collate function
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                       pin_memory=True, collate_fn=collate_fn)
    
    # Determine visualization indices
    if args.indices_file:
        with open(args.indices_file, 'r') as f:
            vis_indices = set([int(x.strip()) for x in f.readlines() if x.strip()])
        vis_indices = {x for x in vis_indices if 0 <= x < N}
        logging.info(f"Using {len(vis_indices)} visualization indices from {args.indices_file}")
    else:
        # Select random indices for visualization only
        k_vis = min(args.num_vis, N)
        random.seed(args.seed)
        vis_indices = set(random.sample(range(N), k_vis))
        # Save visualization indices for reproducibility
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        sel_path = out_dir / 'selected_indices.txt'
        with open(sel_path, 'w') as sf:
            sf.write('\n'.join([str(x) for x in sorted(vis_indices)]))
        logging.info(f"Selected {len(vis_indices)} samples for visualization, indices written to {sel_path}")
    
    # Process limit (default: all)
    process_limit = args.num if args.num < N else N
    logging.info(f"Processing {process_limit} images, visualizing {len(vis_indices)} images")

    # load model
    # Instantiate the backbone and attach a detection head that will use the
    # U-Net RGB-T fusion. Load whatever keys the checkpoint contains (backbone,
    # unet, reg_layer, det_head if present) permissively.
    model = Swin_BM_RGBT(pre_train=False)
    
    # Create det_adaptor if it was used during training (match GN/BN)
    in_ch = 768
    if getattr(args, 'det_use_gn', False):
        gn_groups = 1
        for g in (32, 16, 8, 4, 2, 1):
            if in_ch % g == 0:
                gn_groups = g
                break
        norm_layer = nn.GroupNorm(gn_groups, in_ch)
    else:
        norm_layer = nn.BatchNorm2d(in_ch)
    model.det_adaptor = nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=1),
        norm_layer,
        nn.ReLU(inplace=True)
    )
    
    # Create detection head wrapper with parameters matching training configuration
    head_conv = getattr(args, 'head_conv', 256)
    use_deconv = getattr(args, 'use_deconv', True)
    use_fpn = getattr(args, 'use_fpn', False)
    det_head = DetectionHeadWrapper(
        in_channels=768,
        head_conv=head_conv,
        use_deconv=use_deconv,
        keypoint_only=getattr(args, 'keypoint_mode', False),
        use_fpn=use_fpn,
        use_gn=getattr(args, 'det_use_gn', False),
        use_logits=getattr(args, 'use_bce_logits', False)
    )
    try:
        model.attach_det_head(det_head)
    except Exception:
        model.det_head = det_head

    ckpt = torch.load(args.ckpt, map_location=device)
    # Strip 'module.' prefix from DDP checkpoint if present
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    # permissive load into model (will ignore missing keys)
    try:
        model.load_state_dict(ckpt if isinstance(ckpt, dict) and 'model_state_dict' not in ckpt else ckpt.get('model_state_dict', ckpt), strict=False)
    except Exception:
        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception:
            pass
    model.to(device)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []
    csv_rows = []
    # For AP calculation using detection_eval
    preds_per_image = []  # list of list of (x, y, score)
    gts_per_image = []    # list of np.ndarray (N,2)

    processed_count = 0
    for batch_idx, batch in enumerate(loader):
        if processed_count >= process_limit:
            break
        
        rgb_batch = batch['rgb'].to(device)  # [B, 3, H, W]
        t_batch = batch['t'].to(device)      # [B, 3, H, W]
        gt_pts_batch = batch.get('points', None)
        ids_batch = batch.get('id', None)
        
        batch_size_actual = rgb_batch.shape[0]
        
        # Batch inference
        with torch.no_grad():
            res = model(rgb_batch, t_batch)
            if isinstance(res, tuple) and len(res) >= 2:
                dets = res[1]
            else:
                dets = (None, None, None)
            heat_pred = dets[0]  # [B, 1, H_out, W_out]
            offset_pred = dets[2]  # [B, 2, H_out, W_out] or None
        
        # Process each image in batch
        for i in range(batch_size_actual):
            idx = batch_idx * args.batch_size + i
            if idx >= process_limit:
                break
            processed_count += 1
            
            # Get image ID
            img_id = ids_batch[i] if ids_batch is not None else f"sample_{idx:04d}"
            
            # Extract predictions for this image
            preds_px = []
            if heat_pred is not None:
                heat_np = heat_pred[i].detach().cpu().numpy()  # [1, H_out, W_out]
                off_np = offset_pred[i].detach().cpu().numpy() if offset_pred is not None else None  # [2, H_out, W_out]
                hm = heat_np[0]
                
                if np.nanmin(hm) < 0.0 or np.nanmax(hm) > 1.0:
                    hm = np.clip(hm, -50.0, 50.0)
                    hm = 1.0 / (1.0 + np.exp(-hm))
                
                # Apply NMS during peak extraction (CenterNet-style max-pooling NMS)
                use_nms = True  # Always apply peak extraction NMS
                nms_kernel = getattr(args, 'nms_kernel', 3)
                peaks = heatmap_peaks(
                    hm,
                    min_score=args.min_score,
                    use_nms=use_nms,
                    nms_kernel=nms_kernel
                )
                
                for x_out, y_out, score in peaks:
                    offx = float(off_np[0, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    offy = float(off_np[1, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    cx = (x_out + offx) * args.downsample_ratio
                    cy = (y_out + offy) * args.downsample_ratio
                    preds_px.append((cx, cy, float(score)))

            # get GT points (numpy array Nx2) in pixel coords
            gt_pts = gt_pts_batch[i] if gt_pts_batch is not None else None
            if gt_pts is None:
                gt_pts = np.zeros((0, 2), dtype=np.float32)
            else:
                if isinstance(gt_pts, torch.Tensor):
                    gt_pts = gt_pts.cpu().numpy()
                gt_pts = np.asarray(gt_pts, dtype=np.float32)
                if gt_pts.ndim == 1:
                    gt_pts = np.zeros((0, 2), dtype=np.float32)
            gt_pts = cast(np.ndarray, gt_pts)

            # For AP calculation: store per-image preds and GTs
            preds_per_image.append([(float(p[0]), float(p[1]), float(p[2])) for p in preds_px])
            gts_per_image.append(gt_pts if len(gt_pts) > 0 else np.zeros((0, 2)))

            # Apply optional post-extraction NMS (matches training's eval_nms settings)
            if getattr(args, 'eval_nms_radius', None):
                preds_px = nms_radius(preds_px, float(args.eval_nms_radius))
            if getattr(args, 'eval_soft_nms_sigma', None):
                preds_px = soft_nms_gaussian(preds_px, float(args.eval_soft_nms_sigma), score_thresh=None)
            # Limit to max detections (matching training's max_detections)
            if args.max_dets and args.max_dets > 0:
                preds_px = sorted(preds_px, key=lambda x: x[2], reverse=True)[:int(args.max_dets)]
            # perform greedy matching between preds and GTs using distance threshold
            dist_thresh = args.ap_dist_thresh
            # sort preds by score descending
            preds_sorted = sorted(preds_px, key=lambda x: x[2], reverse=True)
            matched_pred_flags = [False] * len(preds_sorted)
            gt_matched = [False] * (len(gt_pts) if len(gt_pts) > 0 else 0)
            tp = 0
            fp = 0
            fn = 0
            # for each pred, match to nearest unmatched GT
            for j, (px, py, score) in enumerate(preds_sorted):
                if len(gt_pts) == 0:
                    matched = False
                    matched_pred_flags[j] = False
                    fp += 1
                    continue
                dists = np.sqrt((gt_pts[:, 0] - px) ** 2 + (gt_pts[:, 1] - py) ** 2)
                unmatched_idx = [k for k, m in enumerate(gt_matched) if not m]
                if len(unmatched_idx) == 0:
                    matched = False
                    matched_pred_flags[j] = False
                    fp += 1
                    continue
                dists_un = dists[unmatched_idx]
                minpos = int(np.argmin(dists_un))
                gt_idx = unmatched_idx[minpos]
                if dists_un[minpos] <= dist_thresh:
                    matched = True
                    matched_pred_flags[j] = True
                    gt_matched[gt_idx] = True
                    tp += 1
                else:
                    matched = False
                    matched_pred_flags[j] = False
                    fp += 1

            fn = (len(gt_pts) - sum(gt_matched)) if len(gt_pts) > 0 else 0

            # Only visualize selected indices
            if idx in vis_indices:
                # Preprocess images only for visualization
                rgb_vis = preprocess_image(rgb_batch[i], mean=RGB_NORM_MEAN, std=RGB_NORM_STD, to_bgr=True)
                t_vis = preprocess_image(t_batch[i], mean=T_NORM_MEAN, std=T_NORM_STD, to_bgr=True)
                
                # annotate preds_px with matched flags in original order (not sorted)
                preds_with_flags = []
                sorted_map = { (p[0], p[1], p[2]): j for j,p in enumerate(preds_sorted) }
                for p in preds_px:
                    key = (p[0], p[1], p[2])
                    si = sorted_map.get(key, None)
                    if si is None:
                        preds_with_flags.append((p[0], p[1], p[2], None))
                    else:
                        preds_with_flags.append((p[0], p[1], p[2], bool(matched_pred_flags[si])))

                # create overlay image with both halves and markers
                canvas = vis_on_image(rgb_vis, t_vis, preds_with_flags, args.downsample_ratio, None)

                # draw GT points on canvas (blue X on RGB and thermal)
                h_img, w_img = rgb_vis.shape[:2]
                sep = 8
                for g in gt_pts:
                    gx = int(round(float(g[0])))
                    gy = int(round(float(g[1])))
                    # draw X on RGB half
                    cv2.line(canvas, (gx - 4, gy - 4), (gx + 4, gy + 4), (255, 0, 0), 2)
                    cv2.line(canvas, (gx - 4, gy + 4), (gx + 4, gy - 4), (255, 0, 0), 2)
                    # draw on thermal half
                    cv2.line(canvas, (w_img + sep + gx - 4, gy - 4), (w_img + sep + gx + 4, gy + 4), (255, 0, 0), 2)
                    cv2.line(canvas, (w_img + sep + gx - 4, gy + 4), (w_img + sep + gx + 4, gy - 4), (255, 0, 0), 2)

                # save using dataset id for clarity
                save_path = out_dir / f"{img_id}.jpg"
                cv2.imwrite(str(save_path), canvas)
                logging.debug(f'Saved visualization {save_path}')
            
            report_lines.append(f"{img_id}: TP={tp} FP={fp} FN={fn} #GT={len(gt_pts)} #Preds={len(preds_px)}")

            # append per-prediction rows for CSV: image_id, cx, cy, score, matched
            preds_with_flags_csv = []
            sorted_map = { (p[0], p[1], p[2]): j for j,p in enumerate(preds_sorted) }
            for p in preds_px:
                key = (p[0], p[1], p[2])
                si = sorted_map.get(key, None)
                matched_val = None if si is None else bool(matched_pred_flags[si])
                cx, cy, score = p
                csv_rows.append([img_id, float(cx), float(cy), float(score), '' if matched_val is None else ('TP' if matched_val else 'FP')])

    # write report summary
    if len(report_lines) > 0:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for line in report_lines:
            parts = line.split()
            for part in parts:
                if part.startswith('TP='):
                    total_tp += int(part.split('=')[1])
                if part.startswith('FP='):
                    total_fp += int(part.split('=')[1])
                if part.startswith('FN='):
                    total_fn += int(part.split('=')[1])

        # F1 calculation
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AP calculation using detection_eval.compute_ap
        ap, ap_precisions, ap_recalls = compute_ap(preds_per_image, gts_per_image, dist_thresh=args.ap_dist_thresh)

        report_path = out_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write('Per-sample TP/FP/FN summary:\n')
            for line in report_lines:
                f.write(line + '\n')
            f.write('\n')
            f.write(f'Total TP={total_tp} FP={total_fp} FN={total_fn}\n')
            f.write(f'Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} AP={ap:.4f}\n')
        print('Saved report to', report_path)

    # write CSV of per-prediction scores if requested
    if args.scores_csv is not None:
        csv_path = out_dir / args.scores_csv
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['image_id', 'cx', 'cy', 'score', 'label'])
            for r in csv_rows:
                writer.writerow(r)
        print('Saved scores CSV to', csv_path)

    # produce histogram of TP vs FP scores if requested and data available
    if args.scores_hist is not None and len(csv_rows) > 0:
        tp_scores = [r[3] for r in csv_rows if r[4] == 'TP']
        fp_scores = [r[3] for r in csv_rows if r[4] == 'FP']
        plt.figure(figsize=(6,4))
        bins = np.linspace(0.0, 1.0, 50).tolist()
        if len(fp_scores) > 0:
            plt.hist(fp_scores, bins=bins, alpha=0.6, label=f'FP (n={len(fp_scores)})', color='red')
        if len(tp_scores) > 0:
            plt.hist(tp_scores, bins=bins, alpha=0.6, label=f'TP (n={len(tp_scores)})', color='green')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Predicted score distribution (TP vs FP)')
        hist_path = out_dir / args.scores_hist
        plt.tight_layout()
        plt.savefig(str(hist_path))
        plt.close()
        print('Saved score histogram to', hist_path)


def parse_args():
    """Parse command-line arguments for detection inference and visualization.
    
    This script mirrors training-time evaluation exactly using the same parameters:
    
    PARAMETER MAPPING TO TRAINING EVALUATION:
    ========================================
    
    Peak Extraction (during inference):
      --min-score         Training: det_score_threshold (default 0.3)
      --nms-kernel        Training: nms_kernel (default 3, CenterNet max-pooling)
    
    Post-Extraction Filtering (optional, off by default to match training):
      --eval-nms-radius   Training: eval_nms_radius (optional, default None)
      --eval-soft-nms-sigma Training: eval_soft_nms_sigma (optional, default None)
    
    Output Limiting:
      --max-dets          Training: max_detections (default 300)
    
    Matching & Metrics:
      --ap-dist-thresh    Training: ap_dist_thresh (default 8.0 pixels)
    
    DEFAULT BEHAVIOR:
      When run with no NMS arguments, this script produces IDENTICAL evaluation
      results to what the model achieved during training. The AP metric should
      match exactly what's reported in training logs.
    
    EXPERIMENTAL (Post-extraction NMS):
      Add --eval-nms-radius or --eval-soft-nms-sigma to experiment with
      stricter NMS settings. These are NOT used during training evaluation,
      so results will differ from training metrics.
    
    Example (matches training evaluation):
      python test_detection_vis.py --data-dir .data/DroneRGBT_converted \\
        --ckpt checkpoints_phase5/best_model.pth --out visuals_detection
    
    Example (with experimental NMS):
      python test_detection_vis.py --data-dir .data/DroneRGBT_converted \\
        --ckpt checkpoints_phase5/best_model.pth --out visuals_detection \\
        --eval-nms-radius 2.0 --eval-soft-nms-sigma 1.0
    """
    parser = argparse.ArgumentParser(description="Visualize and evaluate detection model inference.")
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--out', default='visuals_detection')
    parser.add_argument('--num', type=int, default=10000,
                        help='number of images to process for inference (default: all)')
    parser.add_argument('--num-vis', type=int, default=8,
                        help='number of images to visualize and save')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size for inference (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading (default: 4)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--min-score', type=float, default=0.3,
                        help='minimum score threshold for peak detection (default: 0.3, matches training threshold)')
    parser.add_argument('--ap-dist-thresh', type=float, default=8.0,
                        help='distance threshold (pixels) used to match predictions to GT for the report')
    parser.add_argument('--max-dets', type=int, default=300,
                        help='keep top-K detections per image (default: 300, matches training)')
    parser.add_argument('--nms-kernel', type=int, default=3,
                        help='NMS kernel size for max-pooling NMS during peak extraction (default: 3, CenterNet-style)')
    parser.add_argument('--eval-nms-radius', type=float, default=None,
                        help='radius (pixels) for radius-based NMS on extracted peaks (matches training --eval-nms-radius)')
    parser.add_argument('--eval-soft-nms-sigma', type=float, default=None,
                        help='sigma (pixels) for Gaussian soft-NMS decay (matches training --eval-soft-nms-sigma)')
    parser.add_argument('--scores-csv', type=str, default='scores.csv',
                        help='filename to write per-prediction scores/labels into (written into --out dir); set to empty to skip')
    parser.add_argument('--scores-hist', type=str, default='scores_hist.png',
                        help='filename for TP/FP score histogram (written into --out dir); set to empty to skip')
    parser.add_argument('--tile-size', type=int, default=None,
                        help='optional tile size (pixels) for sliding-window inference')
    parser.add_argument('--tile-overlap', type=float, default=None,
                        help='tile overlap fraction (0-0.9), e.g., 0.25')
    parser.add_argument('--indices-file', type=str, default=None,
                        help='optional path to a file with one dataset index per line to use for inference')
    parser.add_argument('--head-conv', type=int, default=256,
                        help='detection head conv channels (must match training, default 256)')
    parser.add_argument('--use-deconv', action='store_true', default=True,
                        help='use deconv upsampling in head (must match training, default True)')
    parser.add_argument('--use-fpn', action='store_true',
                        help='enable FPN neck (must match training)')
    parser.add_argument('--use-bce-logits', action='store_true',
                        help='use logits output in head (must match training)')
    parser.add_argument('--det-use-gn', action='store_true',
                        help='use GroupNorm in head (must match training)')
    parser.add_argument('--keypoint-mode', action='store_true',
                        help='keypoint-only mode: model has no size head (Phase 1)')
    return parser.parse_args()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = parse_args()
    infer_and_visualize(args)
