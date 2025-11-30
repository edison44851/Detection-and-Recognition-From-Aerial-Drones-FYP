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
import torch
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2

from datasets.dm_detection import DetectionDataset
from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.center_head import CenterHead
from utils.detection_eval import heatmap_peaks


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


def preprocess_image(tensor_img):
    # tensor expected [C,H,W] or [1,C,H,W]
    if isinstance(tensor_img, torch.Tensor):
        if tensor_img.dim() == 4:
            tensor_img = tensor_img[0]
        arr = tensor_img.detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        # normalize to 0-255 for visualization (assume input in 0..1 or -1..1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        arr = (arr * 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return arr
    else:
        return tensor_img


def infer_and_visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio)
    N = len(ds)
    # choose indices: either from provided indices file or random unique sample
    if args.indices_file:
        with open(args.indices_file, 'r') as f:
            sel = [int(x.strip()) for x in f.readlines() if x.strip()]
        # clamp indices to dataset range
        sel = [x for x in sel if 0 <= x < N]
        print(f"Using {len(sel)} indices from {args.indices_file}")
        write_selected = False
    else:
        # select unique random indices up to dataset size
        k = min(args.num, N)
        random.seed(args.seed)
        # use random.sample to ensure uniqueness
        sel = random.sample(range(N), k)
        # safety: ensure sel contains unique indices (preserve order)
        seen = set()
        sel = [x for x in sel if not (x in seen or seen.add(x))]
        # log selected indices for reproducibility
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        sel_path = out_dir / 'selected_indices.txt'
        with open(sel_path, 'w') as sf:
            sf.write('\n'.join([str(x) for x in sel]))
        print(f"Selected {len(sel)} unique samples, indices written to {sel_path}")
        write_selected = False
    # Ensure we don't process the same image id multiple times (defensive)
    processed_ids = set()

    # load model
    # Instantiate the backbone and attach a detection head that will use the
    # U-Net RGB-T fusion. Load whatever keys the checkpoint contains (backbone,
    # unet, reg_layer, det_head if present) permissively.
    model = Swin_BM_RGBT(pre_train=False)
    det_head = CenterHead(in_channels=768)
    try:
        model.attach_det_head(det_head)
    except Exception:
        model.det_adaptor = getattr(model, 'det_adaptor', nn.Identity())
        model.det_head = det_head

    ckpt = torch.load(args.ckpt, map_location=device)
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

    def run_on_patch(rgb_patch, t_patch, x0=0, y0=0):
        rgb_vis = preprocess_image(rgb_patch)
        t_vis = preprocess_image(t_patch)
        rgb_b = rgb_patch.unsqueeze(0).to(device)
        t_b = t_patch.unsqueeze(0).to(device)
        with torch.no_grad():
            res = model(rgb_b, t_b)
            if isinstance(res, tuple) and len(res) >= 2:
                dets = res[1]
            else:
                dets = (None, None, None)
            heat_pred = dets[0]
            offset_pred = dets[2]
            preds_px = []
            if heat_pred is not None:
                heat_np = heat_pred.detach().cpu().numpy()
                off_np = offset_pred.detach().cpu().numpy() if offset_pred is not None else None
                hm = heat_np[0, 0]
                if np.nanmin(hm) < 0.0 or np.nanmax(hm) > 1.0:
                    hm = 1.0 / (1.0 + np.exp(-hm))
                peaks = heatmap_peaks(hm, min_score=args.min_score)
                for x_out, y_out, score in peaks:
                    offx = float(off_np[0, 0, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    offy = float(off_np[0, 1, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    cx = (x_out + offx) * args.downsample_ratio + x0
                    cy = (y_out + offy) * args.downsample_ratio + y0
                    preds_px.append((cx, cy, float(score)))
        return rgb_vis, t_vis, preds_px

    for idx in sel:
        sample = ds[idx]
        # prefer dataset-provided id if available
        id0 = sample.get('id', None)
        img_id = id0 if id0 is not None else f"sample_{idx:04d}"
        # skip if we've already processed this dataset id
        if img_id in processed_ids:
            print(f"Skipping duplicate image id {img_id}")
            continue
        processed_ids.add(img_id)
        rgb = sample['rgb']  # tensor [3,H,W]
        t = sample['t']
        # preserve original image for visualization
        rgb_vis = preprocess_image(rgb)
        t_vis = preprocess_image(t)

        preds_px = []
        rgb_vis = preprocess_image(rgb)
        t_vis = preprocess_image(t)
        if args.tile_size:
            # sliding-window tiling
            H, W = rgb.shape[1], rgb.shape[2]
            ts = int(args.tile_size)
            ov = float(args.tile_overlap or 0.0)
            stride = max(1, int(ts * (1.0 - ov)))
            for y0 in range(0, H, stride):
                for x0 in range(0, W, stride):
                    y1 = min(y0 + ts, H)
                    x1 = min(x0 + ts, W)
                    rgb_patch = rgb[:, y0:y1, x0:x1]
                    t_patch = t[:, y0:y1, x0:x1]
                    rv, tv, ppx = run_on_patch(rgb_patch, t_patch, x0=x0, y0=y0)
                    preds_px.extend(ppx)
        else:
            # full image
            rv, tv, ppx = run_on_patch(rgb, t, x0=0, y0=0)
            preds_px.extend(ppx)

        # get GT points (numpy array Nx2) in pixel coords
        gt_pts = sample.get('points', None)
        if gt_pts is None:
            gt_pts = []
        else:
            if isinstance(gt_pts, torch.Tensor):
                gt_pts = gt_pts.numpy()

        # apply NMS/topK before matching
        if args.nms_radius:
            preds_px = nms_radius(preds_px, float(args.nms_radius))
        if args.max_dets:
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
        for i, (px, py, score) in enumerate(preds_sorted):
            if len(gt_pts) == 0:
                matched = False
                matched_pred_flags[i] = False
                fp += 1
                continue
            dists = np.sqrt((gt_pts[:, 0] - px) ** 2 + (gt_pts[:, 1] - py) ** 2)
            unmatched_idx = [j for j, m in enumerate(gt_matched) if not m]
            if len(unmatched_idx) == 0:
                matched = False
                matched_pred_flags[i] = False
                fp += 1
                continue
            dists_un = dists[unmatched_idx]
            minpos = int(np.argmin(dists_un))
            idx = unmatched_idx[minpos]
            if dists_un[minpos] <= dist_thresh:
                matched = True
                matched_pred_flags[i] = True
                gt_matched[idx] = True
                tp += 1
            else:
                matched = False
                matched_pred_flags[i] = False
                fp += 1

        fn = (len(gt_pts) - sum(gt_matched)) if len(gt_pts) > 0 else 0

        # annotate preds_px with matched flags in original order (not sorted)
        # build map from sorted preds to original preds
        preds_with_flags = []
        sorted_map = { (p[0], p[1], p[2]): i for i,p in enumerate(preds_sorted) }
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
        print('Saved', save_path)
        report_lines.append(f"{img_id}: TP={tp} FP={fp} FN={fn} #GT={len(gt_pts)} #Preds={len(preds_px)}")

        # append per-prediction rows for CSV: image_id, cx, cy, score, matched
        for p in preds_with_flags:
            img_id = img_id
            cx, cy, score, matched = p
            csv_rows.append([img_id, float(cx), float(cy), float(score), '' if matched is None else ('TP' if matched else 'FP')])

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
        report_path = out_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write('Per-sample TP/FP/FN summary:\n')
            for line in report_lines:
                f.write(line + '\n')
            f.write('\n')
            f.write(f'Total TP={total_tp} FP={total_fp} FN={total_fn}\n')
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
        bins = np.linspace(0.0, 1.0, 50)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--out', default='visuals_detection')
    parser.add_argument('--num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--min-score', type=float, default=0.01)
    parser.add_argument('--ap-dist-thresh', type=float, default=8.0,
                        help='distance threshold (pixels) used to match predictions to GT for the report')
    parser.add_argument('--max-dets', type=int, default=None,
                        help='keep top-K detections per image after NMS')
    parser.add_argument('--nms-radius', type=float, default=None,
                        help='radius (pixels) for simple radius-NMS to suppress nearby peaks')
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer_and_visualize(args)
