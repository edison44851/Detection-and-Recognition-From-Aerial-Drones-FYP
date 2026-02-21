#!/usr/bin/env python3
"""Run inference on the dataset `val` (test) images and save all output images.

Behavior:
- If `--source` is provided it will use that directory (or file). Otherwise it will
  try to read `val` from the provided `--data` yaml.
- Saves images with model visualizations to `--output/--name` using the Ultralytics
  `YOLO.predict(..., save=True)` mechanism.

Examples:
  python ultralytics/yolo/infer_test_save.py --weights runs/finetune/...' --data ../yolo_droneRGBT/data_rgb.yaml --device 1 --output ../runs/infer --name rgb_test
  python ultralytics/yolo/infer_test_save.py --weights yolo26s.pt --source ./yolo_droneRGBT/images/test/rgb --output ./out --name quick
"""

import argparse
import sys
import os
import shutil
from pathlib import Path
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on test set and save images")
    p.add_argument("--weights", required=True, help="model weights file or name")
    p.add_argument("--data", default="../yolo_droneRGBT/data_combined.yaml", help="dataset yaml (optional)")
    p.add_argument("--source", default=None, help="override source directory or file for inference")
    p.add_argument("--device", default=None, help="device id or cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.0, help="model confidence threshold for predictions")
    p.add_argument("--vis-thres", type=float, default=None, help="visualization threshold (overrides --conf for drawing)")
    p.add_argument("--max-boxes", type=int, default=None, help="maximum boxes to draw per image (top by score)")
    p.add_argument("--hide-labels", action="store_true", help="do not draw text labels on images")
    p.add_argument("--box-thickness", type=int, default=2, help="box line thickness for drawn boxes")
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--output", default="../runs/infer", help="output project directory (relative to script)")
    p.add_argument("--name", default="predict", help="run name inside output")
    p.add_argument("--save_txt", action="store_true", help="save prediction .txt files")
    p.add_argument("--recursive", action="store_true", help="search images recursively")
    return p.parse_args()


def resolve_data_val(data_arg):
    # Accept absolute / cwd-relative path first, else treat as script-relative
    data_p = Path(data_arg).expanduser()
    if not data_p.exists():
        data_p = Path(__file__).resolve().parent.joinpath(data_arg).resolve()
    if not data_p.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_arg}")

    # parse YAML safely
    try:
        import yaml
    except Exception:
        raise RuntimeError("PyYAML is required to read data yaml. Install with: pip install pyyaml")

    with open(data_p, "r") as f:
        data = yaml.safe_load(f)

    # Expect 'val' key
    if "val" not in data:
        raise KeyError("'val' key not found in data yaml")

    # Resolve val path relative to yaml's parent if it's relative
    val_path = Path(data["val"])
    if not val_path.is_absolute():
        val_path = data_p.parent.joinpath(val_path).resolve()
    return val_path


def collect_images_from_dir(d: Path, recursive: bool = False):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not d.exists():
        raise FileNotFoundError(f"Source directory not found: {d}")
    paths = []
    if recursive:
        for p in d.rglob("*"):
            if p.suffix.lower() in exts:
                paths.append(str(p))
    else:
        for p in d.iterdir():
            if p.suffix.lower() in exts:
                paths.append(str(p))
    paths.sort()
    return paths


def main(args):
    # Ensure requested CUDA device is used. For numeric devices (e.g. 1) we set
    # CUDA_VISIBLE_DEVICES to that index so Ultralytics / PyTorch will only see
    # that GPU and allocate there. For 'cpu', hide GPUs.
    device_arg = args.device
    device_for_predict = device_arg
    if device_arg is not None:
        ds = str(device_arg)
        if ds.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device_for_predict = "cpu"
        else:
            # accept formats: '1', 'cuda:1'
            if ds.startswith("cuda:") and ds.split(":")[1].isdigit():
                idx = ds.split(":")[1]
            elif ds.isdigit():
                idx = ds
            else:
                idx = None
            if idx is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = idx
                # inside this process the selected GPU will be mapped to index 0
                device_for_predict = "0"

    try:
        from ultralytics import YOLO
    except Exception:
        print("Install ultralytics first: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    # determine source list
    if args.source:
        src = Path(args.source)
        if src.is_dir():
            img_list = collect_images_from_dir(src, recursive=args.recursive)
        elif src.exists():
            img_list = [str(src)]
        else:
            # try resolving relative to cwd or script
            resolved = Path(args.source).resolve()
            if resolved.exists():
                if resolved.is_dir():
                    img_list = collect_images_from_dir(resolved, recursive=args.recursive)
                else:
                    img_list = [str(resolved)]
            else:
                raise FileNotFoundError(f"Source not found: {args.source}")
    else:
        val_path = resolve_data_val(args.data)
        if val_path.is_dir():
            img_list = collect_images_from_dir(val_path, recursive=args.recursive)
        else:
            # if val path is a file list or single file
            if val_path.suffix in {".txt", ".csv"}:
                with open(val_path, "r") as f:
                    img_list = [l.strip() for l in f if l.strip()]
            else:
                img_list = [str(val_path)]

    if not img_list:
        print("No images found for inference.")
        sys.exit(2)

    print(f"Running inference on {len(img_list)} images")

    model = YOLO(args.weights)

    # Use Ultralytics predict; save=True writes images to project/name
    save_dir = Path(__file__).resolve().parent.joinpath(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # We'll perform our own visualization and saving, so ask the predictor
    # to not save images itself. We still pass device/conf/iou/imgsz.
    common_kwargs = dict(
        save=False,
        save_txt=False,
        imgsz=args.imgsz,
        device=device_for_predict or None,
        conf=args.conf,
        iou=args.iou,
        batch=1,
    )

    # load class names from data yaml if present
    class_names = {}
    try:
        data_p = Path(args.data).expanduser()
        if not data_p.exists():
            data_p = Path(__file__).resolve().parent.joinpath(args.data).resolve()
        import yaml
        with open(data_p, 'r') as f:
            dd = yaml.safe_load(f)
            if 'names' in dd:
                names = dd['names']
                if isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    class_names = {i: n for i, n in enumerate(names)}
    except Exception:
        class_names = {}

    # Create target output dir for visualized images
    target_dir = Path(save_dir) / args.name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Process images one-by-one to avoid accumulating all results in RAM.
    total = len(img_list)
    print(f"Processing {total} images one-by-one to avoid RAM accumulation")

    # For evaluation: collect all prediction scores and match flags (1=TP,0=FP)
    all_scores = []
    all_matches = []
    total_gts = 0

    def load_image_size(path):
        with Image.open(path) as im:
            return im.size  # (width, height)

    def find_label_for_image(img_path: str):
        p = Path(img_path)
        # try replacing '/images/' with '/labels/'
        parts = p.parts
        try_paths = []
        if "images" in parts:
            idx = parts.index("images")
            new_parts = list(parts)
            new_parts[idx] = "labels"
            new_p = Path(*new_parts).with_suffix('.txt')
            try_paths.append(new_p)
        # fallback: sibling labels dir at same level
        lbl = p.with_suffix('.txt')
        # try searching for labels folder anywhere under repo
        try_paths.append(lbl)
        # also try replacing '/images/test'->'/labels/test' or '/images/val'->'/labels/val'
        try:
            for tp in try_paths:
                if tp.exists():
                    return tp
        except Exception:
            pass
        return None

    def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    # convert to xyxy in pixels
                    x1 = (xc - w/2.0) * img_w
                    y1 = (yc - h/2.0) * img_h
                    x2 = (xc + w/2.0) * img_w
                    y2 = (yc + h/2.0) * img_h
                    boxes.append((x1, y1, x2, y2, cls))
        except Exception:
            pass
        return boxes

    def iou(boxA, boxB):
        # boxes are (x1,y1,x2,y2)
        xa1, ya1, xa2, ya2 = boxA
        xb1, yb1, xb2, yb2 = boxB
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        areaA = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        areaB = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = areaA + areaB - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    failures = 0
    for i, img in enumerate(img_list, start=1):
        try:
            print(f"[{i}/{total}] {img}")
            # run prediction for single image
            res_list = model.predict(source=img, **common_kwargs)
            # model.predict returns a list; take first Results
            res = res_list[0] if isinstance(res_list, (list, tuple)) and len(res_list) > 0 else res_list

            # extract predicted boxes (xyxy in pixels), scores, classes
            preds = []
            try:
                boxes = res.boxes.xyxy.cpu().numpy()  # Nx4
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(boxes, scores, classes):
                    preds.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(c)))
            except Exception:
                preds = []

            # load ground truth
            img_w, img_h = load_image_size(img)
            label_path = find_label_for_image(img)
            gts = []
            if label_path is not None:
                gts = load_yolo_labels(Path(label_path), img_w, img_h)
            total_gts += len(gts)

            # match predictions to gts using greedy by score
            preds_sorted = sorted(preds, key=lambda x: x[4], reverse=True)
            matched_gt = set()
            for (x1, y1, x2, y2, score, cls) in preds_sorted:
                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(gts):
                    if j in matched_gt:
                        continue
                    gt_box = gt[0:4]
                    gt_cls = gt[4]
                    if gt_cls != cls:
                        continue
                    cur_iou = iou((x1, y1, x2, y2), gt_box)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_j = j
                if best_iou >= 0.5 and best_j >= 0:
                    # true positive
                    all_scores.append(score)
                    all_matches.append(1)
                    matched_gt.add(best_j)
                else:
                    # false positive
                    all_scores.append(score)
                    all_matches.append(0)

            # --- Visualization: filter preds for drawing ---
            vis_thres = args.vis_thres if args.vis_thres is not None else args.conf
            vis_preds = [p for p in preds_sorted if p[4] >= vis_thres]
            if args.max_boxes is not None:
                vis_preds = vis_preds[: args.max_boxes]

            # Draw filtered boxes on image and save
            try:
                im = Image.open(img).convert('RGB')
                draw = ImageDraw.Draw(im)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                for (x1, y1, x2, y2, score, cls) in vis_preds:
                    color = (0, 120, 255)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=args.box_thickness)
                    if not args.hide_labels:
                        label = f"{class_names.get(cls, 'cls'+str(cls))} {score:.2f}"
                        try:
                            text_w, text_h = draw.textsize(label, font=font)
                        except Exception:
                            text_w, text_h = (len(label) * 6, 10)
                        text_bg = (0, 120, 255)
                        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=text_bg)
                        draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)
                out_path = target_dir / Path(img).name
                im.save(out_path)
            except Exception as e:
                print(f"Failed to draw/save {img}: {e}")

            # optionally save prediction .txt for filtered preds
            if args.save_txt:
                try:
                    txt_path = target_dir / (Path(img).stem + '.txt')
                    with open(txt_path, 'w') as tf:
                        for (x1, y1, x2, y2, score, cls) in vis_preds:
                            xc = (x1 + x2) / 2.0 / img_w
                            yc = (y1 + y2) / 2.0 / img_h
                            ww = (x2 - x1) / img_w
                            hh = (y2 - y1) / img_h
                            tf.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
                except Exception as e:
                    print(f"Failed to write txt for {img}: {e}")

        except Exception as e:
            print(f"Failed on {img}: {e}")
            failures += 1

    results = None

    # Flatten saved images: move any images from nested subfolders into
    # the main run folder and remove empty subfolders.
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    target_dir = Path(save_dir) / args.name
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for p in save_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in img_exts:
            # skip files already in target_dir
            if p.parent.resolve() == target_dir.resolve():
                continue
            dest = target_dir / p.name
            if dest.exists():
                # avoid overwrite: prefix with parent folder name
                dest = target_dir / f"{p.parent.name}_{p.name}"
                idx = 1
                while dest.exists():
                    dest = target_dir / f"{p.parent.name}_{idx}_{p.name}"
                    idx += 1
            try:
                shutil.move(str(p), str(dest))
                moved += 1
            except Exception as e:
                print(f"Failed to move {p} -> {dest}: {e}")

    # Remove now-empty directories under save_dir (but keep target_dir)
    # Walk in reverse depth so children are removed before parents
    dirs = [d for d in save_dir.rglob("*") if d.is_dir()]
    dirs.sort(key=lambda p: len(str(p).split(os.sep)), reverse=True)
    for d in dirs:
        try:
            if d.resolve() == target_dir.resolve():
                continue
            d.rmdir()
        except Exception:
            pass

    print(f"Moved {moved} images into: {target_dir}")

    print(f"Saved predictions to: {target_dir}")

    # Compute precision / recall / AP from collected predictions
    try:
        if total_gts == 0:
            print("No ground-truth boxes found; skipping AP calculation.")
        elif len(all_scores) == 0:
            print("No predictions were made; skipping AP calculation.")
        else:
            scores = np.array(all_scores)
            matches = np.array(all_matches)
            # sort by score desc
            order = np.argsort(-scores)
            matches_sorted = matches[order]
            tp_cum = np.cumsum(matches_sorted == 1)
            fp_cum = np.cumsum(matches_sorted == 0)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            recalls = tp_cum / (total_gts + 1e-9)

            # AP: integrate precision-recall curve (trapezoidal)
            # ensure curve starts at recall=0 and ends at recall=1
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([1.0], precisions, [0.0]))
            # make precision monotonically decreasing
            for i in range(len(mpre)-2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i+1])
            # compute area under curve
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = 0.0
            for i in idx:
                ap += (mrec[i+1] - mrec[i]) * mpre[i+1]

            final_precision = precisions[-1] if len(precisions)>0 else 0.0
            final_recall = recalls[-1] if len(recalls)>0 else 0.0

            print("")
            print("Evaluation summary:")
            print(f"  GT boxes: {total_gts}")
            print(f"  Predictions: {len(all_scores)} (TP={int(tp_cum[-1])}, FP={int(fp_cum[-1])})")
            print(f"  Precision: {final_precision:.4f}")
            print(f"  Recall:    {final_recall:.4f}")
            print(f"  AP:        {ap:.6f}")

            # Save evaluation summary to JSON
            eval_summary = {
                "gt_boxes": int(total_gts),
                "predictions": int(len(all_scores)),
                "tp": int(tp_cum[-1]),
                "fp": int(fp_cum[-1]),
                "precision": float(final_precision),
                "recall": float(final_recall),
                "ap": float(ap),
            }
            # Include full PR curve arrays and sorted scores
            try:
                scores_sorted = scores[order]
                eval_summary["pr_curve"] = {
                    "recalls": recalls.tolist(),
                    "precisions": precisions.tolist(),
                    "scores": scores_sorted.tolist(),
                    "mrec": mrec.tolist(),
                    "mpre": mpre.tolist(),
                }
            except Exception:
                pass
            try:
                json_path = target_dir / "evaluation.json"
                with open(json_path, "w") as jf:
                    json.dump(eval_summary, jf, indent=2)
                print(f"Saved evaluation JSON to: {json_path}")
            except Exception as e:
                print(f"Failed to write evaluation JSON: {e}")
    except Exception as e:
        print(f"Failed to compute AP: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
