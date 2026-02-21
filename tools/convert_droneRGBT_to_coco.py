"""
DroneRGBT to COCO Format Converter
Converts DroneRGBT dataset with XML annotations to COCO format
Supports RGB and Infrared (thermal) images with center-point annotations
Organizes images in COCO directory structure
"""

import os
import json
import cv2
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import argparse


def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file to extract center points.
    Expected XML format with object elements containing point info.
    
    Args:
        xml_path: Path to XML annotation file
    
    Returns:
        List of center points [(cx, cy), ...]
    """
    centers = []
    
    if not os.path.exists(xml_path):
        return centers
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find all object elements
        for obj in root.findall(".//object"):
            # Look for point element with x and y coordinates
            point = obj.find("point")
            if point is not None:
                try:
                    x_elem = point.find("x")
                    y_elem = point.find("y")
                    if x_elem is not None and y_elem is not None:
                        cx = float(x_elem.text)
                        cy = float(y_elem.text)
                        centers.append((cx, cy))
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        print(f"Warning: Error parsing {xml_path}: {e}")
    
    return centers


def center_to_bbox(cx, cy, box_size=15):
    """
    Convert center point to bounding box [x, y, w, h].
    Creates a square bounding box of size box_size x box_size around center.
    
    Args:
        cx: Center x coordinate
        cy: Center y coordinate
        box_size: Size of square box (default 15px)
    
    Returns:
        [x, y, width, height] in COCO format
    """
    x = cx - box_size / 2
    y = cy - box_size / 2
    return [x, y, box_size, box_size]


def load_image_size(image_path):
    """Get image dimensions."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height


def convert_droneRGBT_to_coco(data_dir, split="train", modality="rgb", 
                              bbox_size=15, output_dir="./coco_droneRGBT"):
    """
    Convert DroneRGBT dataset to COCO format with organized images.
    
    Args:
        data_dir: Root directory of DroneRGBT dataset
        split: "train" or "test"
        modality: "rgb" or "thermal" (infrared)
        bbox_size: Size of bounding box around center point (default 15px)
        output_dir: Output directory for COCO dataset
    
    Returns:
        COCO dataset dictionary and output structure info
    """
    
    # Build paths (handle both capitalized and lowercase)
    data_path = Path(data_dir)
    
    # Try to find the split directory (case-insensitive)
    split_candidates = [data_path / split.capitalize(), data_path / split.lower()]
    split_dir = None
    for candidate in split_candidates:
        if candidate.exists():
            split_dir = candidate
            break
    
    if split_dir is None:
        raise FileNotFoundError(f"Split directory not found for '{split}' in {data_path}")
    
    if modality.lower() == "rgb":
        image_dir = split_dir / "RGB"
        gt_dir = split_dir / "GT_"
        image_ext = ".jpg"
    elif modality.lower() in ["thermal", "infrared"]:
        image_dir = split_dir / "Infrared"
        gt_dir = split_dir / "GT_"
        image_ext = ".jpg"
    else:
        raise ValueError("modality must be 'rgb' or 'thermal'")
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    
    # Create output directory structure
    output_path = Path(output_dir) / f"{split}_{modality}"
    images_output_dir = output_path / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConverting {split} {modality} dataset")
    print(f"  Source image dir: {image_dir}")
    print(f"  Source GT dir: {gt_dir}")
    print(f"  Output dir: {output_path}")
    
    # Initialize COCO format
    coco_data = {
        "info": {
            "description": "DroneRGBT Person Detection Dataset",
            "version": "1.0",
            "year": 2024,
            "date_created": "2024",
            "split": split,
            "modality": modality
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "human"
            }
        ]
    }
    
    # Get image files
    image_files = sorted([f for f in image_dir.glob(f"*{image_ext}") if f.is_file()])
    print(f"Found {len(image_files)} images")
    
    annotation_id = 1
    image_id = 1
    copied_count = 0
    
    for idx, image_path in enumerate(image_files):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(image_files)}...")
        
        # Get image ID (filename without extension)
        stem = image_path.stem
        
        # Find corresponding XML file
        xml_path = gt_dir / f"{stem}.xml"
        
        if not xml_path.exists():
            # Try alternate naming (thermal with R suffix)
            if modality.lower() in ["thermal", "infrared"]:
                if stem.endswith("R"):
                    base_stem = stem[:-1]
                    xml_path = gt_dir / f"{base_stem}R.xml"
            
            if not xml_path.exists():
                continue
        
        # Get image dimensions
        width, height = load_image_size(str(image_path))
        if width is None or height is None:
            print(f"  Warning: Could not load image {image_path}")
            continue
        
        # Copy image to output directory
        output_image_path = images_output_dir / image_path.name
        shutil.copy2(image_path, output_image_path)
        copied_count += 1
        
        # Add image to COCO (with relative path)
        coco_data["images"].append({
            "id": image_id,
            "file_name": f"images/{image_path.name}",
            "height": height,
            "width": width
        })
        
        # Parse annotations
        centers = parse_xml_annotation(str(xml_path))
        
        # Clamp centers to valid range and create bboxes
        for cx, cy in centers:
            # Clamp to image boundaries
            cx = max(0, min(cx, width))
            cy = max(0, min(cy, height))
            
            # Convert center to bbox
            bbox = center_to_bbox(cx, cy, bbox_size)
            x, y, w, h = bbox
            
            # Clamp bbox to image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w <= 0 or h <= 0:
                continue
            
            area = w * h
            
            # Add annotation to COCO
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # person class
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    print(f"✓ Converted {len(coco_data['images'])} images")
    print(f"✓ Copied {copied_count} images")
    print(f"✓ Created {len(coco_data['annotations'])} annotations")
    
    # Save annotations JSON
    annotations_file = output_path / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✓ Saved annotations to: {annotations_file}")
    
    return coco_data, output_path


def create_coco_dataset_splits(data_dir, bbox_size=15, output_dir="./coco_droneRGBT"):
    """
    Create COCO format datasets for all splits and modalities with organized images.
    
    Args:
        data_dir: Root directory of DroneRGBT dataset
        bbox_size: Size of bounding box around center point
        output_dir: Output directory for COCO dataset
    """
    print("=" * 70)
    print("DroneRGBT to COCO Format Conversion with Image Organization")
    print("=" * 70)
    
    results = {}
    
    # Convert train RGB
    try:
        data, path = convert_droneRGBT_to_coco(
            data_dir, split="train", modality="rgb", 
            bbox_size=bbox_size, output_dir=output_dir
        )
        results["train_rgb"] = (data, path)
    except Exception as e:
        print(f"❌ Error converting train RGB: {e}")
    
    # Convert train thermal
    try:
        data, path = convert_droneRGBT_to_coco(
            data_dir, split="train", modality="thermal", 
            bbox_size=bbox_size, output_dir=output_dir
        )
        results["train_thermal"] = (data, path)
    except Exception as e:
        print(f"❌ Error converting train thermal: {e}")
    
    # Convert test RGB
    try:
        data, path = convert_droneRGBT_to_coco(
            data_dir, split="test", modality="rgb", 
            bbox_size=bbox_size, output_dir=output_dir
        )
        results["test_rgb"] = (data, path)
    except Exception as e:
        print(f"❌ Error converting test RGB: {e}")
    
    # Convert test thermal
    try:
        data, path = convert_droneRGBT_to_coco(
            data_dir, split="test", modality="thermal", 
            bbox_size=bbox_size, output_dir=output_dir
        )
        results["test_thermal"] = (data, path)
    except Exception as e:
        print(f"❌ Error converting test thermal: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    for split_name, (data, path) in results.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Location: {path}")
        print(f"  Images: {len(data['images'])}")
        print(f"  Annotations: {len(data['annotations'])}")
        if len(data['images']) > 0:
            avg_annot = len(data['annotations']) / len(data['images'])
            print(f"  Avg annotations per image: {avg_annot:.2f}")
    
    print("\n" + "=" * 70)
    print(f"✓ All datasets converted successfully!")
    print(f"✓ Output saved to: {Path(output_dir).resolve()}/")
    print("\nDataset Structure:")
    print(f"{output_dir}/")
    print("├── train_rgb/")
    print("│   ├── images/")
    print("│   └── annotations.json")
    print("├── train_thermal/")
    print("│   ├── images/")
    print("│   └── annotations.json")
    print("├── test_rgb/")
    print("│   ├── images/")
    print("│   └── annotations.json")
    print("└── test_thermal/")
    print("    ├── images/")
    print("    └── annotations.json")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DroneRGBT dataset to COCO format with organized images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./DroneRGBT",
        help="Path to DroneRGBT dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./coco_droneRGBT",
        help="Output directory for COCO dataset (default: ./coco_droneRGBT)"
    )
    parser.add_argument(
        "--bbox_size",
        type=int,
        default=15,
        help="Size of bounding box around center point (default: 15px)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Which split to convert"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["rgb", "thermal", "all"],
        default="all",
        help="Which modality to convert"
    )
    
    args = parser.parse_args()
    
    if args.split == "all" and args.modality == "all":
        create_coco_dataset_splits(args.data_dir, args.bbox_size, args.output_dir)
    else:
        split = args.split if args.split != "all" else "train"
        modality = args.modality if args.modality != "all" else "rgb"
        convert_droneRGBT_to_coco(args.data_dir, split, modality, args.bbox_size, args.output_dir)


if __name__ == "__main__":
    main()
