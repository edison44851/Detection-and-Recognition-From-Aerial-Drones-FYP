#!/usr/bin/env python3
"""Run detection inference on thermal-only images.

This script allows inference when RGB pairing or GT IDs are unavailable.
It loads thermal images from a folder and feeds them into the RGBT model
using one of two modes:

- duplicate: use thermal image for both RGB and thermal branches
- zero-rgb:  use zeros for RGB branch and thermal image for thermal branch
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.det_model import DetectionHeadWrapper
from utils.detection_eval import heatmap_peaks


def nms_radius(preds, radius):
    if len(preds) == 0:
        return []
    pts = sorted(preds, key=lambda x: x[2], reverse=True)
    keep = []
    taken = [False] * len(pts)
    for i, (x, y, s) in enumerate(pts):
        if taken[i]:
            continue
        keep.append((x, y, s))
        for j in range(i + 1, len(pts)):
            if taken[j]:
                continue
            x2, y2, _ = pts[j]
            if (x - x2) ** 2 + (y - y2) ** 2 <= radius * radius:
                taken[j] = True
    return keep


def soft_nms_gaussian(preds, sigma):
    if len(preds) == 0:
        return []
    pts = sorted([(float(x), float(y), float(s)) for x, y, s in preds], key=lambda x: x[2], reverse=True)
    keep = []
    for i in range(len(pts)):
        xi, yi, si = pts[i]
        if si <= 0:
            continue
        for j in range(i + 1, len(pts)):
            xj, yj, sj = pts[j]
            if sj <= 0:
                continue
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            decay = np.exp(-d2 / (2.0 * (sigma ** 2)))
            pts[j] = (xj, yj, sj * (1.0 - decay))
        keep.append((xi, yi, si))
    return keep


def draw_predictions(img_bgr, preds_px):
    canvas = img_bgr.copy()
    for cx, cy, score in preds_px:
        ix = int(round(cx))
        iy = int(round(cy))
        color = (0, 255, 255)
        cv2.circle(canvas, (ix, iy), 6, color, 2)
        cv2.putText(canvas, f"{score:.2f}", (ix + 6, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return canvas


def build_model(args, device):
    model = Swin_BM_RGBT(pre_train=False)

    in_ch = 768
    if args.det_use_gn:
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
        nn.ReLU(inplace=True),
    )

    det_head = DetectionHeadWrapper(
        in_channels=768,
        head_conv=args.head_conv,
        use_deconv=args.use_deconv,
        keypoint_only=args.keypoint_mode,
        use_fpn=args.use_fpn,
        use_gn=args.det_use_gn,
        use_logits=args.use_bce_logits,
    )

    try:
        model.attach_det_head(det_head)
    except Exception:
        model.det_head = det_head

    ckpt = torch.load(args.ckpt, map_location=device)

    # Normalize checkpoint into a plain state dict.
    state = ckpt
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    if not isinstance(state, dict):
        logging.warning("Checkpoint format not recognized; model may run with random parameters.")
    else:
        model_state = model.state_dict()
        matched = {}
        skipped = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                matched[k] = v
            else:
                skipped += 1

        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        logging.info("Loaded %d matching tensors from checkpoint; skipped %d incompatible tensors.", len(matched), skipped)

    model.to(device)
    model.eval()
    return model


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.499, 0.168, 0.431], std=[0.308, 0.168, 0.181]),
        ]
    )

    thermal_dir = Path(args.thermal_dir)
    image_paths = sorted(
        [
            p
            for p in thermal_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
    )

    if args.num > 0:
        image_paths = image_paths[: args.num]

    if not image_paths:
        raise RuntimeError(f"No thermal images found in: {thermal_dir}")

    csv_rows = []
    report_lines = []

    for img_path in image_paths:
        pil = Image.open(img_path).convert("RGB")
        t_tensor = cast(torch.Tensor, t_transform(pil))
        t = t_tensor.unsqueeze(0).to(device)

        if args.input_mode == "duplicate":
            rgb = t.clone()
        else:
            rgb = torch.zeros_like(t)

        with torch.no_grad():
            res = model(rgb, t)
            if isinstance(res, tuple) and len(res) >= 2:
                dets = res[1]
            else:
                dets = (None, None, None)
            heat_pred = dets[0]
            offset_pred = dets[2]

        preds_px = []
        if heat_pred is not None:
            hm = heat_pred[0, 0].detach().cpu().numpy()
            off = offset_pred[0].detach().cpu().numpy() if offset_pred is not None else None

            if np.nanmin(hm) < 0.0 or np.nanmax(hm) > 1.0:
                hm = np.clip(hm, -50.0, 50.0)
                hm = 1.0 / (1.0 + np.exp(-hm))

            peaks = heatmap_peaks(hm, min_score=args.min_score, use_nms=True, nms_kernel=args.nms_kernel)
            for x_out, y_out, score in peaks:
                offx = float(off[0, int(y_out), int(x_out)]) if off is not None else 0.0
                offy = float(off[1, int(y_out), int(x_out)]) if off is not None else 0.0
                cx = (x_out + offx) * args.downsample_ratio
                cy = (y_out + offy) * args.downsample_ratio
                preds_px.append((float(cx), float(cy), float(score)))

        if args.eval_nms_radius is not None:
            preds_px = nms_radius(preds_px, float(args.eval_nms_radius))
        if args.eval_soft_nms_sigma is not None:
            preds_px = soft_nms_gaussian(preds_px, float(args.eval_soft_nms_sigma))

        if args.max_dets > 0:
            preds_px = sorted(preds_px, key=lambda x: x[2], reverse=True)[: args.max_dets]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            logging.warning("Failed to read image with cv2: %s", img_path)
            continue

        vis = draw_predictions(img_bgr, preds_px)
        save_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(save_path), vis)

        report_lines.append(f"{img_path.name}: #Preds={len(preds_px)}")
        for cx, cy, score in preds_px:
            csv_rows.append([img_path.name, cx, cy, score])

    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("Thermal-only inference summary\n")
        f.write(f"Input mode: {args.input_mode}\n")
        f.write(f"Images processed: {len(report_lines)}\n\n")
        for line in report_lines:
            f.write(line + "\n")

    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["image_id", "cx", "cy", "score"])
        for row in csv_rows:
            writer.writerow(row)

    logging.info("Saved visualizations to %s", out_dir)
    logging.info("Saved report to %s", report_path)
    logging.info("Saved scores to %s", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Thermal-only inference for RGBT detection model.")
    parser.add_argument("--thermal-dir", required=True, help="folder containing thermal images")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--out", default="visuals_thermal_only", help="output directory")
    parser.add_argument("--num", type=int, default=0, help="max number of images to process (0 means all)")

    parser.add_argument(
        "--input-mode",
        choices=["duplicate", "zero-rgb"],
        default="duplicate",
        help="duplicate=use thermal as both RGB/T; zero-rgb=use zeros for RGB branch",
    )

    parser.add_argument("--downsample-ratio", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--nms-kernel", type=int, default=3)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--eval-nms-radius", type=float, default=None)
    parser.add_argument("--eval-soft-nms-sigma", type=float, default=None)

    parser.add_argument("--head-conv", type=int, default=256)
    parser.add_argument("--use-deconv", action="store_true", default=True)
    parser.add_argument("--use-fpn", action="store_true", default=True)
    parser.add_argument("--use-bce-logits", action="store_true", default=True)
    parser.add_argument("--det-use-gn", action="store_true", default=True)
    parser.add_argument("--keypoint-mode", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    infer(args)
