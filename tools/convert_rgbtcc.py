#!/usr/bin/env python3
"""
RGBT-CC dataset converter: parse JSON ground truth and copy RGB/Thermal images.

RGBT-CC-CVPR2021 structure:
  src/train/<id>_RGB.jpg, <id>_T.jpg, <id>_GT.json
  src/test/<id>_RGB.jpg, <id>_T.jpg, <id>_GT.json
  src/val/<id>_RGB.jpg, <id>_T.jpg, <id>_GT.json

Writes flat structure:
  out/train/<id>_RGB.jpg, <id>_T.jpg, <id>_GT.npy
  out/test/<id>_RGB.jpg, <id>_T.jpg, <id>_GT.npy
  out/val/<id>_RGB.jpg, <id>_T.jpg, <id>_GT.npy
"""

import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def parse_points_from_json(json_path: Path):
    """Parse point annotations from RGBT-CC JSON format."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        points = data.get('points', [])
        if not points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(points, dtype=np.float32)
    except Exception as e:
        print(f"Warning: failed to parse {json_path}: {e}")
        return np.empty((0, 2), dtype=np.float32)


def convert_split(src_root: Path, split: str, out_root: Path):
    """Convert one split (train/test/val) from RGBT-CC to flat format."""
    split_dir = src_root / split
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist, skipping")
        return 0
    
    gt_files = list(split_dir.glob('*_GT.json'))
    if not gt_files:
        print(f"Warning: no *_GT.json files found in {split_dir}")
        return 0
    
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    
    for gt_json in gt_files:
        # Extract base name (e.g., "1162_GT.json" -> "1162")
        base = gt_json.stem.replace('_GT', '')
        
        rgb_img = split_dir / f"{base}_RGB.jpg"
        thermal_img = split_dir / f"{base}_T.jpg"
        
        if not rgb_img.exists() or not thermal_img.exists():
            print(f"Warning: missing images for {base}, skipping")
            continue
        
        # Parse points from JSON
        points = parse_points_from_json(gt_json)
        
        # Copy images
        shutil.copy(rgb_img, out_dir / f"{base}_RGB.jpg")
        shutil.copy(thermal_img, out_dir / f"{base}_T.jpg")
        
        # Save points as .npy
        np.save(out_dir / f"{base}_GT.npy", points)
        
        count += 1
        if count % 100 == 0:
            print(f"  {split}: processed {count} images...")
    
    print(f"✓ {split}: converted {count} images")
    return count


def main():
    parser = argparse.ArgumentParser(description='Convert RGBT-CC dataset to flat format')
    parser.add_argument('--src', type=str, required=True,
                        help='Path to RGBT-CC-CVPR2021 directory')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for converted dataset')
    args = parser.parse_args()
    
    src_root = Path(args.src)
    out_root = Path(args.out)
    
    if not src_root.exists():
        print(f"Error: source directory {src_root} does not exist")
        return
    
    print(f"Converting RGBT-CC from {src_root} to {out_root}")
    
    total = 0
    for split in ['train', 'test', 'val']:
        count = convert_split(src_root, split, out_root)
        total += count
    
    print(f"\n✅ Total: converted {total} images")


if __name__ == '__main__':
    main()
