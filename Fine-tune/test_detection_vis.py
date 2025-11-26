#!/usr/bin/env python3
"""Run detection inference on several random samples and save visualization images.

Saves visualizations to `./visuals_detection` by default. Uses the dataset loader
`DetectionDataset` and expects a model checkpoint path. The script will
load the checkpoint into `DetectionModel` if available, otherwise try to load into
`Swin_BM_RGBT` as a fallback.

Usage:
  python Fine-tune/test_detection_vis.py --data-dir .data/DroneRGBT_counting --ckpt checkpoints/1122-222336/best_model.pth --out visuals_detection --num 8

"""
import os
import argparse
import random
from pathlib import Path
import torch
import numpy as np
import cv2

from datasets.dm_detection import DetectionDataset
from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.center_head import CenterHead
from utils.detection_eval import heatmap_peaks


def vis_on_image(img_rgb, img_t, preds_px, downsample_ratio, save_path):
    # img_rgb, img_t are HxWx3 uint8
    vis = img_rgb.copy()
    # draw predicted centers on RGB and thermal side-by-side
    h, w = vis.shape[:2]
    sep = 8
    canvas = np.zeros((h, w * 2 + sep, 3), dtype=np.uint8)
    canvas[:, :w] = vis
    canvas[:, w + sep: w * 2 + sep] = img_t

    # draw circles at predicted centers (scale from output grid to pixel coords)
    for (cx, cy, score) in preds_px:
        # ensure ints
        ix = int(round(cx))
        iy = int(round(cy))
        # draw on both halves (RGB and thermal)
        cv2.circle(canvas, (ix, iy), 6, (0, 255, 0), 2)
        cv2.putText(canvas, f"{score:.2f}", (ix + 6, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # thermal side offset
        cv2.circle(canvas, (w + sep + ix, iy), 6, (0, 255, 0), 2)

    cv2.imwrite(str(save_path), canvas)


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
    ids = list(range(N))
    random.seed(args.seed)
    random.shuffle(ids)
    sel = ids[:args.num]

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

    for idx in sel:
        sample = ds[idx]
        rgb = sample['rgb']  # tensor [3,H,W]
        t = sample['t']
        # preserve original image for visualization
        rgb_vis = preprocess_image(rgb)
        t_vis = preprocess_image(t)

        # prepare batch dims
        rgb_b = rgb.unsqueeze(0).to(device)
        t_b = t.unsqueeze(0).to(device)
        with torch.no_grad():
            res = model(rgb_b, t_b)
            # may return density or (density, (heat,size,offset))
            if isinstance(res, tuple) and len(res) >= 2:
                dets = res[1]
            else:
                # no detection head available
                dets = (None, None, None)

            heat_pred = dets[0]
            offset_pred = dets[2]

            preds_px = []
            if heat_pred is not None:
                heat_np = heat_pred.detach().cpu().numpy()
                off_np = offset_pred.detach().cpu().numpy() if offset_pred is not None else None
                # assume shape (1,1,Hout,Wout)
                hm = heat_np[0, 0]
                peaks = heatmap_peaks(hm, min_score=args.min_score)
                for x_out, y_out, score in peaks:
                    offx = float(off_np[0, 0, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    offy = float(off_np[0, 1, int(y_out), int(x_out)]) if off_np is not None else 0.0
                    cx = (x_out + offx) * args.downsample_ratio
                    cy = (y_out + offy) * args.downsample_ratio
                    preds_px.append((cx, cy, float(score)))

        save_path = out_dir / f"sample_{idx:04d}.jpg"
        vis_on_image(rgb_vis, t_vis, preds_px, args.downsample_ratio, save_path)
        print('Saved', save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--out', default='visuals_detection')
    parser.add_argument('--num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--min-score', type=float, default=0.01)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer_and_visualize(args)
