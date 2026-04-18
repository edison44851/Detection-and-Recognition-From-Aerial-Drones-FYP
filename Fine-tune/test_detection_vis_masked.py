#!/usr/bin/env python3
"""Run full-image detection inference, then filter GT/predictions with masks.

This differs from dataset masking. Images are kept unchanged for inference and
visualization. After prediction extraction, GT points and predicted centers are
filtered by the mask so only points inside white regions remain.
"""

import argparse
import csv
import logging
import random
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets.dm_detection import DetectionDataset
from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.det_model import DetectionHeadWrapper
from test_detection_vis import (
    RGB_NORM_MEAN,
    RGB_NORM_STD,
    T_NORM_MEAN,
    T_NORM_STD,
    collate_fn,
    nms_radius,
    preprocess_image,
    soft_nms_gaussian,
    vis_on_image,
)
from utils.detection_eval import compute_ap, heatmap_peaks


MASK_NAME_PATTERN = re.compile(
    r"^(train|test)_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE
)


def load_mask_map(mask_dir, split):
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_map = {}
    for mask_path in sorted(p for p in mask_dir.iterdir() if p.is_file()):
        match = MASK_NAME_PATTERN.match(mask_path.name)
        if not match:
            continue
        mask_split = match.group(1).lower()
        sample_id = match.group(2)
        if mask_split != split:
            continue
        mask_map[sample_id] = mask_path

    if not mask_map:
        raise RuntimeError(f"No mask files found for split '{split}' in {mask_dir}")
    return mask_map


def load_binary_mask(mask_path, image_shape):
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    target_h, target_w = image_shape[:2]
    if mask_bin.shape != (target_h, target_w):
        mask_bin = cv2.resize(mask_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask_bin


def filter_points_by_mask(points, mask_bin):
    if points is None:
        return np.zeros((0, 2), dtype=np.float32), 0
    points = np.asarray(points)
    if points.size == 0:
        if points.ndim == 2:
            return points, 0
        return np.zeros((0, 2), dtype=np.float32), 0

    h, w = mask_bin.shape
    keep = []
    removed = 0
    for pt in points:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            removed += 1
            continue
        if mask_bin[y, x] > 0:
            keep.append(pt)
        else:
            removed += 1

    if keep:
        kept = np.array(keep, dtype=points.dtype)
    else:
        width = points.shape[1] if points.ndim == 2 else 2
        kept = np.zeros((0, width), dtype=points.dtype)
    return kept, removed


def filter_predictions_by_mask(preds, mask_bin):
    if not preds:
        return [], 0

    h, w = mask_bin.shape
    keep = []
    removed = 0
    for pred in preds:
        x = int(round(float(pred[0])))
        y = int(round(float(pred[1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            removed += 1
            continue
        if mask_bin[y, x] > 0:
            keep.append(pred)
        else:
            removed += 1
    return keep, removed


def build_loader(args, ds, mask_map):
    if args.include_unmasked_ids:
        eval_ds = ds
        logging.info("Processing full split with mask filtering applied only where masks exist")
    else:
        keep_indices = [i for i, sample_id in enumerate(ds.ids) if sample_id in mask_map]
        if not keep_indices:
            raise RuntimeError("No dataset samples match the mask IDs")
        eval_ds = Subset(ds, keep_indices)
        logging.info("Processing only masked IDs: %d sample(s)", len(keep_indices))

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return eval_ds, loader


def infer_and_visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mask_map = load_mask_map(args.mask_dir, args.split)
    ds = DetectionDataset(args.data_dir, split=args.split, output_stride=args.downsample_ratio)
    eval_ds, loader = build_loader(args, ds, mask_map)
    dataset_size = len(eval_ds)

    if args.indices_file:
        with open(args.indices_file, 'r') as f:
            vis_indices = set(int(x.strip()) for x in f.readlines() if x.strip())
        vis_indices = {x for x in vis_indices if 0 <= x < dataset_size}
        logging.info("Using %d visualization indices from %s", len(vis_indices), args.indices_file)
    else:
        k_vis = min(args.num_vis, dataset_size)
        random.seed(args.seed)
        vis_indices = set(random.sample(range(dataset_size), k_vis)) if k_vis > 0 else set()
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        sel_path = out_dir / 'selected_indices.txt'
        with open(sel_path, 'w') as sf:
            sf.write('\n'.join(str(x) for x in sorted(vis_indices)))
        logging.info("Selected %d samples for visualization, indices written to %s", len(vis_indices), sel_path)

    process_limit = args.num if args.num < dataset_size else dataset_size
    logging.info("Processing %d images, visualizing %d images", process_limit, len(vis_indices))

    model = Swin_BM_RGBT(pre_train=False)

    in_ch = 768
    if getattr(args, 'det_use_gn', False):
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

    det_head = DetectionHeadWrapper(
        in_channels=768,
        head_conv=getattr(args, 'head_conv', 256),
        use_deconv=getattr(args, 'use_deconv', True),
        keypoint_only=getattr(args, 'keypoint_mode', False),
        use_fpn=getattr(args, 'use_fpn', False),
        use_gn=getattr(args, 'det_use_gn', False),
        use_logits=getattr(args, 'use_bce_logits', False)
    )
    try:
        model.attach_det_head(det_head)
    except Exception:
        model.det_head = det_head

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    try:
        state = ckpt if isinstance(ckpt, dict) and 'model_state_dict' not in ckpt else ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
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
    preds_per_image = []
    gts_per_image = []
    mask_cache = {}

    processed_count = 0
    for batch_idx, batch in enumerate(loader):
        if processed_count >= process_limit:
            break

        rgb_batch = batch['rgb'].to(device)
        t_batch = batch['t'].to(device)
        gt_pts_batch = batch.get('points', None)
        ids_batch = batch.get('id', None)

        batch_size_actual = rgb_batch.shape[0]

        with torch.no_grad():
            res = model(rgb_batch, t_batch)
            if isinstance(res, tuple) and len(res) >= 2:
                dets = res[1]
            else:
                dets = (None, None, None)
            heat_pred = dets[0]
            offset_pred = dets[2]

        for i in range(batch_size_actual):
            eval_idx = batch_idx * args.batch_size + i
            if eval_idx >= process_limit:
                break
            processed_count += 1

            img_id = ids_batch[i] if ids_batch is not None else f"sample_{eval_idx:04d}"

            preds_px = []
            if heat_pred is not None:
                heat_np = heat_pred[i].detach().cpu().numpy()
                off_np = offset_pred[i].detach().cpu().numpy() if offset_pred is not None else None
                hm = heat_np[0]

                if np.nanmin(hm) < 0.0 or np.nanmax(hm) > 1.0:
                    hm = np.clip(hm, -50.0, 50.0)
                    hm = 1.0 / (1.0 + np.exp(-hm))

                peaks = heatmap_peaks(
                    hm,
                    min_score=args.min_score,
                    use_nms=True,
                    nms_kernel=getattr(args, 'nms_kernel', 3),
                )
                for x_out, y_out, score in peaks:
                    offx = float(off_np[0, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    offy = float(off_np[1, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    cx = (x_out + offx) * args.downsample_ratio
                    cy = (y_out + offy) * args.downsample_ratio
                    preds_px.append((cx, cy, float(score)))

            gt_pts = gt_pts_batch[i] if gt_pts_batch is not None else None
            if gt_pts is None:
                gt_pts = np.zeros((0, 2), dtype=np.float32)
            elif isinstance(gt_pts, torch.Tensor):
                gt_pts = gt_pts.cpu().numpy()
            elif not isinstance(gt_pts, np.ndarray):
                gt_pts = np.array(gt_pts)
            if gt_pts.ndim == 1 and len(gt_pts) == 0:
                gt_pts = np.zeros((0, 2), dtype=np.float32)

            if getattr(args, 'eval_nms_radius', None):
                preds_px = nms_radius(preds_px, float(args.eval_nms_radius))
            if getattr(args, 'eval_soft_nms_sigma', None):
                preds_px = soft_nms_gaussian(preds_px, float(args.eval_soft_nms_sigma), score_thresh=None)
            if args.max_dets and args.max_dets > 0:
                preds_px = sorted(preds_px, key=lambda x: x[2], reverse=True)[:int(args.max_dets)]

            rgb_vis = preprocess_image(rgb_batch[i], mean=RGB_NORM_MEAN, std=RGB_NORM_STD, to_bgr=True)
            if img_id in mask_map:
                if img_id not in mask_cache:
                    mask_cache[img_id] = load_binary_mask(mask_map[img_id], rgb_vis.shape)
                mask_bin = mask_cache[img_id]
                gt_pts, gt_removed = filter_points_by_mask(gt_pts, mask_bin)
                preds_px, pred_removed = filter_predictions_by_mask(preds_px, mask_bin)
            else:
                gt_removed = 0
                pred_removed = 0

            preds_per_image.append([(float(p[0]), float(p[1]), float(p[2])) for p in preds_px])
            gts_per_image.append(np.array(gt_pts) if len(gt_pts) > 0 else np.zeros((0, 2)))

            preds_sorted = sorted(preds_px, key=lambda x: x[2], reverse=True)
            matched_pred_flags = [False] * len(preds_sorted)
            gt_matched = [False] * (len(gt_pts) if len(gt_pts) > 0 else 0)
            tp = 0
            fp = 0
            for j, (px, py, score) in enumerate(preds_sorted):
                if len(gt_pts) == 0:
                    matched_pred_flags[j] = False
                    fp += 1
                    continue
                dists = np.sqrt((gt_pts[:, 0] - px) ** 2 + (gt_pts[:, 1] - py) ** 2)
                unmatched_idx = [k for k, matched in enumerate(gt_matched) if not matched]
                if len(unmatched_idx) == 0:
                    matched_pred_flags[j] = False
                    fp += 1
                    continue
                dists_un = dists[unmatched_idx]
                minpos = int(np.argmin(dists_un))
                gt_idx = unmatched_idx[minpos]
                if dists_un[minpos] <= args.ap_dist_thresh:
                    matched_pred_flags[j] = True
                    gt_matched[gt_idx] = True
                    tp += 1
                else:
                    matched_pred_flags[j] = False
                    fp += 1
            fn = (len(gt_pts) - sum(gt_matched)) if len(gt_pts) > 0 else 0

            if eval_idx in vis_indices:
                t_vis = preprocess_image(t_batch[i], mean=T_NORM_MEAN, std=T_NORM_STD, to_bgr=True)
                preds_with_flags = []
                sorted_map = {(p[0], p[1], p[2]): j for j, p in enumerate(preds_sorted)}
                for p in preds_px:
                    key = (p[0], p[1], p[2])
                    si = sorted_map.get(key, None)
                    if si is None:
                        preds_with_flags.append((p[0], p[1], p[2], None))
                    else:
                        preds_with_flags.append((p[0], p[1], p[2], bool(matched_pred_flags[si])))

                canvas = vis_on_image(rgb_vis, t_vis, preds_with_flags, args.downsample_ratio, None)
                h_img, w_img = rgb_vis.shape[:2]
                sep = 8
                for g in gt_pts:
                    gx = int(round(float(g[0])))
                    gy = int(round(float(g[1])))
                    cv2.line(canvas, (gx - 4, gy - 4), (gx + 4, gy + 4), (255, 0, 0), 2)
                    cv2.line(canvas, (gx - 4, gy + 4), (gx + 4, gy - 4), (255, 0, 0), 2)
                    cv2.line(canvas, (w_img + sep + gx - 4, gy - 4), (w_img + sep + gx + 4, gy + 4), (255, 0, 0), 2)
                    cv2.line(canvas, (w_img + sep + gx - 4, gy + 4), (w_img + sep + gx + 4, gy - 4), (255, 0, 0), 2)

                save_path = out_dir / f"{img_id}.jpg"
                cv2.imwrite(str(save_path), canvas)

            report_lines.append(
                f"{img_id}: TP={tp} FP={fp} FN={fn} #GT={len(gt_pts)} #Preds={len(preds_px)} "
                f"GT_removed_by_mask={gt_removed} Pred_removed_by_mask={pred_removed}"
            )

            sorted_map = {(p[0], p[1], p[2]): j for j, p in enumerate(preds_sorted)}
            for p in preds_px:
                key = (p[0], p[1], p[2])
                si = sorted_map.get(key, None)
                matched_val = None if si is None else bool(matched_pred_flags[si])
                cx, cy, score = p
                csv_rows.append([
                    img_id,
                    float(cx),
                    float(cy),
                    float(score),
                    '' if matched_val is None else ('TP' if matched_val else 'FP')
                ])

    if report_lines:
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

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ap, _, _ = compute_ap(preds_per_image, gts_per_image, dist_thresh=args.ap_dist_thresh)

        report_path = out_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write('Per-sample TP/FP/FN summary after mask filtering:\n')
            for line in report_lines:
                f.write(line + '\n')
            f.write('\n')
            f.write(f'Total TP={total_tp} FP={total_fp} FN={total_fn}\n')
            f.write(f'Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} AP={ap:.4f}\n')
        print('Saved report to', report_path)

    if args.scores_csv:
        csv_path = out_dir / args.scores_csv
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['image_id', 'cx', 'cy', 'score', 'label'])
            for row in csv_rows:
                writer.writerow(row)
        print('Saved scores CSV to', csv_path)

    if args.scores_hist and csv_rows:
        tp_scores = [row[3] for row in csv_rows if row[4] == 'TP']
        fp_scores = [row[3] for row in csv_rows if row[4] == 'FP']
        plt.figure(figsize=(6, 4))
        bins = np.linspace(0.0, 1.0, 50)
        if fp_scores:
            plt.hist(fp_scores, bins=bins, alpha=0.6, label=f'FP (n={len(fp_scores)})', color='red')
        if tp_scores:
            plt.hist(tp_scores, bins=bins, alpha=0.6, label=f'TP (n={len(tp_scores)})', color='green')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Predicted score distribution after mask filtering')
        hist_path = out_dir / args.scores_hist
        plt.tight_layout()
        plt.savefig(str(hist_path))
        plt.close()
        print('Saved score histogram to', hist_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run full-image inference, then filter GT/predictions using white mask regions.'
    )
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--mask-dir', required=True,
                        help='directory containing <split>_<id> mask images')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--include-unmasked-ids', action='store_true',
                        help='process full split; otherwise only images that have masks are evaluated')
    parser.add_argument('--out', default='visuals_detection_masked')
    parser.add_argument('--num', type=int, default=10000,
                        help='number of images to process for inference (default: all selected images)')
    parser.add_argument('--num-vis', type=int, default=8,
                        help='number of images to visualize and save')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--downsample-ratio', type=int, default=4)
    parser.add_argument('--min-score', type=float, default=0.1)
    parser.add_argument('--ap-dist-thresh', type=float, default=8.0)
    parser.add_argument('--max-dets', type=int, default=300)
    parser.add_argument('--nms-kernel', type=int, default=3)
    parser.add_argument('--eval-nms-radius', type=float, default=11.07)
    parser.add_argument('--eval-soft-nms-sigma', type=float, default=None)
    parser.add_argument('--scores-csv', type=str, default='scores.csv')
    parser.add_argument('--scores-hist', type=str, default='scores_hist.png')
    parser.add_argument('--indices-file', type=str, default=None,
                        help='optional path to a file with one evaluation-dataset index per line')
    parser.add_argument('--head-conv', type=int, default=256,
                        help='detection head conv channels')
    parser.add_argument('--use-deconv', action='store_true', default=True,
                        help='use deconv upsampling in head')
    parser.add_argument('--use-fpn', action='store_true' , default=True,
                        help='enable FPN neck')
    parser.add_argument('--use-bce-logits', action='store_true', default=True,
                        help='use logits output in head')
    parser.add_argument('--det-use-gn', action='store_true', default=True,
                        help='use GroupNorm in head')
    parser.add_argument('--keypoint-mode', action='store_true', default=True,
                        help='keypoint-only mode: model has no size head')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    infer_and_visualize(parse_args())