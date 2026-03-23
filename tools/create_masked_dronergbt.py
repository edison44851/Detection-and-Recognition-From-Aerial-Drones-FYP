#!/usr/bin/env python3
"""Create a masked DroneRGBT dataset from DroneRGBT_converted.

Workflow:
1. Copy source dataset tree to destination.
2. Read mask images from mask directory using OpenCV.
3. Apply each mask to matching test RGB/T images (white=preserve, black=remove).
4. Remove GT points that fall on masked (black) pixels.

Expected mask filenames:
    <split>_<index>.png
For example:
    test_24.png, test_50.png, test_101.png
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


MASK_NAME_PATTERN = re.compile(
    r"^(train|test)_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create masked DroneRGBT dataset")
    parser.add_argument(
        "--src",
        default=".data/DroneRGBT_converted",
        help="Source DroneRGBT_converted directory",
    )
    parser.add_argument(
        "--dst",
        default=".data/DroneRGBT_masked",
        help="Destination masked dataset directory",
    )
    parser.add_argument(
        "--mask-dir",
        default=".data/masked_image",
        help="Directory containing <split>_<index> mask images",
    )
    parser.add_argument(
        "--target-split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--only-mask-ids",
        action="store_true",
        help=(
            "Create destination with only target split and only IDs that have masks. "
            "Without this flag, the full dataset is copied first."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete destination directory if it already exists",
    )
    return parser.parse_args()


def ensure_dataset_copy(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_dir}. Use --overwrite to recreate it."
            )
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def collect_mask_entries(mask_dir: Path, target_split: str) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for mask_path in sorted(p for p in mask_dir.iterdir() if p.is_file()):
        m = MASK_NAME_PATTERN.match(mask_path.name)
        if not m:
            continue
        split_name = m.group(1).lower()
        sample_id = m.group(2)
        if split_name != target_split:
            continue
        entries.append((mask_path, sample_id))
    if not entries:
        raise RuntimeError(
            f"No valid mask files for split '{target_split}' found in {mask_dir}"
        )
    return entries


def create_minimal_dataset(src_dir: Path, dst_dir: Path, split: str, sample_ids: list[str], overwrite: bool) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_dir}. Use --overwrite to recreate it."
            )
        shutil.rmtree(dst_dir)

    src_split_dir = src_dir / split
    dst_split_dir = dst_dir / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in sorted(set(sample_ids), key=lambda x: int(x) if x.isdigit() else x):
        for suffix in ("_RGB.jpg", "_T.jpg", "_GT.npy"):
            src_file = src_split_dir / f"{sample_id}{suffix}"
            if not src_file.exists():
                raise FileNotFoundError(f"Missing source file: {src_file}")
            shutil.copy2(src_file, dst_split_dir / src_file.name)


def load_mask(mask_path: Path) -> np.ndarray:
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise RuntimeError(f"Failed to read mask with OpenCV: {mask_path}")

    # Convert to binary preserve mask: white/bright -> 255, black/dark -> 0
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def apply_mask_to_image(image_path: Path, mask_bin: np.ndarray) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False

    if image.shape[:2] != mask_bin.shape:
        resized_mask = cv2.resize(
            mask_bin,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized_mask = mask_bin

    masked_image = cv2.bitwise_and(image, image, mask=resized_mask)
    return cv2.imwrite(str(image_path), masked_image)


def filter_gt_points(gt_path: Path, mask_bin: np.ndarray) -> tuple[int, int]:
    points = np.load(gt_path)

    if points.size == 0:
        return 0, 0

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Unexpected GT shape in {gt_path}: {points.shape}")

    h, w = mask_bin.shape
    keep = []
    for pt in points:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if mask_bin[y, x] > 0:
            keep.append(pt)

    kept = np.array(keep, dtype=points.dtype) if keep else np.empty((0, points.shape[1]), dtype=points.dtype)
    np.save(gt_path, kept)
    return len(points), len(kept)


def process_masks(mask_entries: list[tuple[Path, str]], dst_dir: Path, target_split: str) -> None:
    split_dir = dst_dir / target_split
    if not split_dir.exists():
        raise FileNotFoundError(f"Destination split directory missing: {split_dir}")

    total_removed = 0
    processed = 0
    for mask_path, sample_id in mask_entries:
        mask_bin = load_mask(mask_path)

        rgb_path = split_dir / f"{sample_id}_RGB.jpg"
        t_path = split_dir / f"{sample_id}_T.jpg"
        gt_path = split_dir / f"{sample_id}_GT.npy"

        missing = [str(p) for p in (rgb_path, t_path, gt_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing expected files for {target_split}_{sample_id}: {', '.join(missing)}"
            )

        if not apply_mask_to_image(rgb_path, mask_bin):
            raise RuntimeError(f"Failed writing masked RGB image: {rgb_path}")
        if not apply_mask_to_image(t_path, mask_bin):
            raise RuntimeError(f"Failed writing masked thermal image: {t_path}")

        before, after = filter_gt_points(gt_path, mask_bin)
        removed = before - after
        total_removed += removed
        processed += 1
        print(
            f"[OK] {target_split}_{sample_id}: GT {before} -> {after} (removed {removed})"
        )

    print(f"Processed {processed} mask(s). Total removed GT points: {total_removed}")


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    mask_dir = Path(args.mask_dir)

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    target_split = args.target_split.lower()
    mask_entries = collect_mask_entries(mask_dir, target_split)

    if args.only_mask_ids:
        sample_ids = [sample_id for _, sample_id in mask_entries]
        print(
            "Creating minimal masked dataset:\n"
            f"  src={src_dir}\n"
            f"  dst={dst_dir}\n"
            f"  split={target_split}\n"
            f"  ids={sorted(set(sample_ids), key=lambda x: int(x) if x.isdigit() else x)}"
        )
        create_minimal_dataset(src_dir, dst_dir, target_split, sample_ids, args.overwrite)
    else:
        print(f"Copying dataset:\n  src={src_dir}\n  dst={dst_dir}")
        ensure_dataset_copy(src_dir, dst_dir, args.overwrite)

    print(f"Applying masks from: {mask_dir} (split={target_split})")
    process_masks(mask_entries, dst_dir, target_split)
    print("Done.")


if __name__ == "__main__":
    main()
