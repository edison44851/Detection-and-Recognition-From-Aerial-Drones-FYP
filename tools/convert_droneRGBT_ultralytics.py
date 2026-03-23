#!/usr/bin/env python3
"""
Convert DroneRGBT dataset to YOLO format with separate RGB and Thermal modalities.

DroneRGBT dataset structure:
  - RGB/X.jpg: RGB images (e.g., 1000.jpg)
  - Infrared/XR.jpg: Thermal images (e.g., 1000R.jpg)
  - GT_/XR.xml: Annotations for thermal (e.g., 1000R.xml)

This script:
1. Copies XR.xml to X.xml for RGB image annotations
2. Creates separate label files for RGB and Thermal images
3. Supports multiple output modes:
   - 'separate': RGB and Thermal in `/images/train/rgb/`, `/images/train/thermal/`
   - 'combined_stack': Both in same folder with RGB_XXXX.jpg and TH_XXXX.jpg naming
   - 'rgb_only': Only RGB images with annotations
   - 'thermal_only': Only Thermal images with annotations
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import shutil


def parse_xml_annotation(xml_path: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Parse XML annotation file and extract image dimensions and center points.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Tuple of (width, height, list of (x, y) center points)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract image dimensions
    size = root.find('size')
    if size is None:
        return 0, 0, []
    width_elem = size.find('width')
    height_elem = size.find('height')
    if width_elem is None or height_elem is None or width_elem.text is None or height_elem.text is None:
        return 0, 0, []
    width = int(width_elem.text)
    height = int(height_elem.text)
    
    # Extract center points
    points = []
    for obj in root.findall('object'):
        point = obj.find('point')
        if point is None:
            continue
        x_elem = point.find('x')
        y_elem = point.find('y')
        if x_elem is None or y_elem is None or x_elem.text is None or y_elem.text is None:
            continue
        x = int(x_elem.text)
        y = int(y_elem.text)
        points.append((x, y))
    
    return width, height, points


def create_bbox_from_center(center_x: int, center_y: int, bbox_size: int = 15) -> Tuple[float, float, float, float]:
    """
    Create a bounding box with specified size around a center point.
    
    Args:
        center_x: Center x coordinate
        center_y: Center y coordinate
        bbox_size: Size of bounding box (15px means 15x15 box)
        
    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    half_size = bbox_size / 2.0
    x_min = max(0, center_x - half_size)
    y_min = max(0, center_y - half_size)
    x_max = center_x + half_size
    y_max = center_y + half_size
    
    return x_min, y_min, x_max, y_max


def bbox_to_yolo_format(x_min: float, y_min: float, x_max: float, y_max: float, 
                        img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates to YOLO normalized format.
    YOLO format: (class_id, center_x_norm, center_y_norm, width_norm, height_norm)
    
    Args:
        x_min, y_min, x_max, y_max: Bounding box in pixel coordinates
        img_width, img_height: Image dimensions
        
    Returns:
        (center_x_norm, center_y_norm, width_norm, height_norm)
    """
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Clamp to [0, 1]
    center_x_norm = max(0, min(1, center_x_norm))
    center_y_norm = max(0, min(1, center_y_norm))
    width_norm = max(0, min(1, width_norm))
    height_norm = max(0, min(1, height_norm))
    
    return center_x_norm, center_y_norm, width_norm, height_norm


def copy_xml_annotations(gt_dir: Path) -> None:
    """
    Copy XR.xml files to X.xml for RGB image annotations.
    
    Args:
        gt_dir: Path to GT_ directory containing XR.xml files
    """
    xml_files = sorted(list(gt_dir.glob('*R.xml')))
    print(f"  Copying {len(xml_files)} XML files for RGB annotations...")
    
    for xml_path in tqdm(xml_files, desc="Copying annotations", leave=False):
        # Create corresponding X.xml by copying from XR.xml
        rgb_xml_path = xml_path.parent / xml_path.name.replace('R.xml', '.xml')
        if not rgb_xml_path.exists():
            shutil.copy(xml_path, rgb_xml_path)


def convert_dataset_separate(data_dir: str, output_dir: str, bbox_size: int = 15, 
                            splits: Optional[Dict[str, str]] = None):
    """
    Convert DroneRGBT dataset to YOLO format with separate RGB and Thermal modalities.
    
    Output structure:
      images/
        train/
          rgb/
          thermal/
        test/
          rgb/
          thermal/
      labels/
        train/
          rgb/
          thermal/
        test/
          rgb/
          thermal/
    
    Args:
        data_dir: Path to .data/DroneRGBT directory
        output_dir: Output directory for YOLO format dataset
        bbox_size: Size of bounding box around center points (in pixels)
        splits: Dictionary mapping split names to folders, e.g., {'train': 'Train', 'test': 'Test'}
    """
    
    if splits is None:
        splits = {'train': 'Train', 'test': 'Test'}
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create YOLO directory structure
    for split in splits.keys():
        for modality in ['rgb', 'thermal']:
            (output_path / 'images' / split / modality).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split / modality).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, split_folder in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        split_path = data_path / split_folder
        rgb_dir = split_path / 'RGB'
        thermal_dir = split_path / 'Infrared'
        gt_dir = split_path / 'GT_'
        
        if not all([rgb_dir.exists(), thermal_dir.exists(), gt_dir.exists()]):
            print(f"  Warning: Missing directories in {split_path}")
            continue
        
        # Copy XML annotations for RGB images
        copy_xml_annotations(gt_dir)
        
        # Get list of RGB annotations (both original XR.xml and copied X.xml)
        rgb_xml_files = sorted(list(gt_dir.glob('*.xml')))
        rgb_xml_files = [f for f in rgb_xml_files if not f.name.endswith('R.xml')]
        
        print(f"  Found {len(rgb_xml_files)} RGB annotations (copied from thermal)")
        
        # Process RGB images
        for gt_file in tqdm(rgb_xml_files, desc=f"Converting {split_name} RGB"):
            base_name = gt_file.stem  # e.g., "1000"
            
            # Image paths
            rgb_image_path = rgb_dir / f"{base_name}.jpg"
            
            if not rgb_image_path.exists():
                continue
            
            try:
                # Read RGB image
                rgb_img = cv2.imread(str(rgb_image_path))
                if rgb_img is None:
                    continue
                
                # Parse annotations
                width, height, points = parse_xml_annotation(str(gt_file))
                
                # Save RGB image
                output_image_name = f"{base_name}.jpg"
                output_image_path = output_path / 'images' / split_name / 'rgb' / output_image_name
                cv2.imwrite(str(output_image_path), rgb_img)
                
                # Convert annotations to YOLO format
                yolo_annotations = []
                for cx, cy in points:
                    x_min, y_min, x_max, y_max = create_bbox_from_center(cx, cy, bbox_size)
                    cx_norm, cy_norm, w_norm, h_norm = bbox_to_yolo_format(x_min, y_min, x_max, y_max, 
                                                                            width, height)
                    yolo_annotations.append(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # Save YOLO label file
                output_label_name = f"{base_name}.txt"
                output_label_path = output_path / 'labels' / split_name / 'rgb' / output_label_name
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
            except Exception as e:
                continue
        
        # Get thermal annotations (XR.xml files)
        thermal_xml_files = sorted(list(gt_dir.glob('*R.xml')))
        print(f"  Found {len(thermal_xml_files)} thermal annotations")
        
        # Process thermal images
        for gt_file in tqdm(thermal_xml_files, desc=f"Converting {split_name} Thermal"):
            base_name = gt_file.stem  # e.g., "1000R"
            
            # Image path
            thermal_image_path = thermal_dir / f"{base_name}.jpg"
            
            if not thermal_image_path.exists():
                continue
            
            try:
                # Read thermal image
                thermal_img = cv2.imread(str(thermal_image_path))
                if thermal_img is None:
                    continue
                
                # Parse annotations
                width, height, points = parse_xml_annotation(str(gt_file))
                
                # Save thermal image
                output_image_name = f"{base_name}.jpg"
                output_image_path = output_path / 'images' / split_name / 'thermal' / output_image_name
                cv2.imwrite(str(output_image_path), thermal_img)
                
                # Convert annotations to YOLO format
                yolo_annotations = []
                for cx, cy in points:
                    x_min, y_min, x_max, y_max = create_bbox_from_center(cx, cy, bbox_size)
                    cx_norm, cy_norm, w_norm, h_norm = bbox_to_yolo_format(x_min, y_min, x_max, y_max, 
                                                                            width, height)
                    yolo_annotations.append(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # Save YOLO label file
                output_label_name = f"{base_name}.txt"
                output_label_path = output_path / 'labels' / split_name / 'thermal' / output_label_name
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
            except Exception as e:
                continue
    
    print(f"\n✓ Dataset conversion complete!")
    print(f"  Output saved to: {output_path}")
    print(f"\n  Structure:")
    print(f"    RGB images: {output_path / 'images' / 'train' / 'rgb'}")
    print(f"    RGB labels: {output_path / 'labels' / 'train' / 'rgb'}")
    print(f"    Thermal images: {output_path / 'images' / 'train' / 'thermal'}")
    print(f"    Thermal labels: {output_path / 'labels' / 'train' / 'thermal'}")


def create_dataset_yaml(output_dir: str, modality: str = 'rgb'):
    """
    Create a YAML configuration file for YOLO training.
    
    Args:
        output_dir: Output directory where YOLO dataset is saved
        modality: 'rgb', 'thermal', or 'both' (for separate modality configs)
    """
    output_path = Path(output_dir)
    
    if modality == 'both':
        # Create YAML for combined training (if needed later)
        yaml_content = f"""path: {output_path.absolute()}
train: images/train
test: images/test

nc: 1
names:
  0: person
"""
        yaml_path = output_path / 'data_combined.yaml'
    else:
        yaml_content = f"""path: {output_path.absolute()}
train: images/train/{modality}
test: images/test/{modality}

nc: 1
names:
  0: person
"""
        yaml_path = output_path / f'data_{modality}.yaml'
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created YOLO configuration: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DroneRGBT dataset to YOLO format (separate modalities)")
    parser.add_argument("--data-dir", default=".data/DroneRGBT", 
                       help="Path to DroneRGBT dataset directory")
    parser.add_argument("--output-dir", default="yolo/DroneRGBT_yolo_separate", 
                       help="Output directory for YOLO format dataset")
    parser.add_argument("--bbox-size", type=int, default=15, 
                       help="Size of bounding box around center points (in pixels)")
    
    args = parser.parse_args()
    
    # Convert dataset with separate modalities
    convert_dataset_separate(args.data_dir, args.output_dir, bbox_size=args.bbox_size)
    
    # Create YAML configs for both RGB and Thermal
    create_dataset_yaml(args.output_dir, modality='rgb')
    create_dataset_yaml(args.output_dir, modality='thermal')
    create_dataset_yaml(args.output_dir, modality='both')
