#!/usr/bin/env python3
"""Ultralytics fine-tune helper for yolo_droneRGBT dataset.

Place this in the workspace `ultralytics/` folder and run:

  python ultralytics/finetune_yolo_droneRGBT.py --weights yolo26s.pt

Requires: pip install ultralytics
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Ultralytics YOLO on yolo_droneRGBT")
    p.add_argument("--weights", default="yolo26s.pt", help="pretrained weights (model) to start from")
    p.add_argument("--data", default="../yolo_droneRGBT/data_combined.yaml", help="path to dataset yaml (relative to this script)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640, help="image size")
    p.add_argument("--patience", type=int, default=10, help="early stopping patience (epochs)")
    p.add_argument("--device", default=None, help="cuda device, e.g. 0 or cpu (leave None for auto)")
    p.add_argument("--project", default="../runs/finetune", help="save project dir (relative)")
    p.add_argument("--name", default="yolo_droneRGBT_finetune", help="run name")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint if available")
    return p.parse_args()


def main(args):
    try:
        from ultralytics import YOLO
    except Exception:
        print("Install ultralytics first: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    # Resolve data path: accept absolute or cwd-relative, otherwise fall back to script-relative
    requested = Path(args.data).expanduser()
    if requested.exists():
        data_path = requested.resolve()
    else:
        data_path = Path(__file__).resolve().parent.joinpath(args.data).resolve()
    if not data_path.exists():
        print(f"Data path not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Training with weights={args.weights}, data={data_path}")
    model = YOLO(args.weights)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": str(Path(__file__).resolve().parent.joinpath(args.project)),
        "name": args.name,
        "device": args.device,
    }
    # include patience only when provided
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    if args.resume:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)

    print("Done training — attempting validation...")
    try:
        model.val(data=str(data_path), batch=args.batch, imgsz=args.imgsz)
    except Exception:
        print("Validation skipped or failed.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
