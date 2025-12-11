#!/usr/bin/env python3
"""Compute mean and std of thermal images in RGBT-CC dataset.

Usage:
    python tools/compute_thermal_stats.py --data-dir .data/RGBT-CC_converted --split train
    python tools/compute_thermal_stats.py --data-dir .data/RGBT-CC_converted --split val
    python tools/compute_thermal_stats.py --data-dir .data/RGBT-CC_converted --split test
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_stats(data_dir, split='train'):
    """Compute mean and std of thermal images.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        split: 'train', 'val', or 'test'
    
    Returns:
        mean: [C] mean per channel
        std: [C] std per channel
    """
    split_dir = os.path.join(data_dir, split)
    
    # Find all thermal images
    files = [f for f in os.listdir(split_dir) if f.endswith('_T.jpg')]
    print(f"Found {len(files)} thermal images in {split} split")
    
    # Accumulate statistics
    pixel_sum = None
    pixel_sum_sq = None
    num_pixels = 0
    
    for fname in tqdm(files, desc=f"Computing stats for {split}"):
        fpath = os.path.join(split_dir, fname)
        img = Image.open(fpath).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Accumulate
        if pixel_sum is None:
            pixel_sum = np.zeros(3)
            pixel_sum_sq = np.zeros(3)
        
        # Sum across spatial dims
        pixel_sum += img_np.reshape(-1, 3).sum(axis=0)
        pixel_sum_sq += (img_np.reshape(-1, 3) ** 2).sum(axis=0)
        num_pixels += img_np.reshape(-1, 3).shape[0]
    
    # Compute mean and std
    mean = pixel_sum / num_pixels
    var = (pixel_sum_sq / num_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 0.0))  # Clamp negative values from numerical errors
    
    print(f"\n{split.upper()} split statistics (normalized to [0, 1]):")
    print(f"  Mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"  Std:  [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Compute thermal image statistics')
    parser.add_argument('--data-dir', type=str, default='.data/RGBT-CC_converted',
                        help='Path to dataset root')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--all-splits', action='store_true',
                        help='Compute stats for all splits')
    args = parser.parse_args()
    
    if args.all_splits:
        all_means = []
        all_stds = []
        for split in ['train', 'val', 'test']:
            mean, std = compute_stats(args.data_dir, split)
            all_means.append(mean)
            all_stds.append(std)
        
        # Compute average
        avg_mean = np.mean(all_means, axis=0)
        avg_std = np.mean(all_stds, axis=0)
        
        print(f"\n\nAVERAGE across all splits (normalized to [0, 1]):")
        print(f"  Mean: [{avg_mean[0]:.3f}, {avg_mean[1]:.3f}, {avg_mean[2]:.3f}]")
        print(f"  Std:  [{avg_std[0]:.3f}, {avg_std[1]:.3f}, {avg_std[2]:.3f}]")
    else:
        compute_stats(args.data_dir, args.split)


if __name__ == '__main__':
    main()
