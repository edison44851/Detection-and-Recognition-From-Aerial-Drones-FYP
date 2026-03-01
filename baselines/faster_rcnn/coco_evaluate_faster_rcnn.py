"""
Detectron2 Model Evaluation on Test Dataset
Evaluates trained model on test set and reports metrics
"""

import os
import json
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
import argparse
import random

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer


def load_coco_dataset(dataset_json_path, dataset_root_dir):
    """
    Load COCO format dataset.
    
    Args:
        dataset_json_path: Path to annotations.json file
        dataset_root_dir: Root directory of dataset (containing 'images/' subdir)
    
    Returns:
        List of dataset dicts in Detectron2 format
    """
    dataset_dicts = []
    
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")
    
    # Load COCO JSON
    with open(dataset_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Loading COCO dataset from {dataset_json_path}")
    print(f"Found {len(coco_data['images'])} images")
    
    # Build image_id -> image_info mapping
    image_info_map = {img['id']: img for img in coco_data['images']}
    
    # Build image_id -> annotations mapping
    annotations_map = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_map[ann['image_id']].append(ann)
    
    dataset_root = Path(dataset_root_dir)
    
    for idx, (image_id, image_info) in enumerate(image_info_map.items()):
        if idx % 200 == 0:
            print(f"  Processing {idx}/{len(image_info_map)}...")
        
        # Build full image path
        image_path = dataset_root / image_info['file_name']
        
        if not image_path.exists():
            continue
        
        # Get image dimensions
        height = image_info['height']
        width = image_info['width']
        
        # Get annotations for this image
        annots = annotations_map.get(image_id, [])
        if not annots:
            continue
        
        # Convert COCO annotations to Detectron2 format
        objs = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            
            obj = {
                "bbox": [x, y, x + w, y + h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": ann['category_id'] - 1,
                "iscrowd": ann.get('iscrowd', 0)
            }
            objs.append(obj)
        
        if not objs:
            continue
        
        record = {
            "file_name": str(image_path),
            "image_id": image_id,
            "height": height,
            "width": width,
            "annotations": objs
        }
        dataset_dicts.append(record)
    
    print(f"✓ Loaded {len(dataset_dicts)} valid images with annotations")
    return dataset_dicts

def register_coco_test_dataset(coco_root_dir="./coco_droneRGBT", modality="rgb"):
    """Register COCO test dataset."""
    coco_root = Path(coco_root_dir)
    test_split_dir = coco_root / f"test_{modality}"
    test_json_path = test_split_dir / "annotations.json"
    
    if not test_json_path.exists():
        raise FileNotFoundError(f"Test annotations not found: {test_json_path}")
    
    test_dicts = load_coco_dataset(str(test_json_path), str(test_split_dir))
    
    dataset_name = f"drone_test_{modality}_eval"
    
    def make_test_loader(data=test_dicts):
        return data
    
    DatasetCatalog.register(dataset_name, make_test_loader)
    MetadataCatalog.get(dataset_name).set(thing_classes=["person"])
    
    return test_dicts, dataset_name


def load_trained_model(checkpoint_path, num_classes=1, score_threshold=0.5):
    """Load trained model from checkpoint."""
    cfg = get_cfg()
    from detectron2.model_zoo import get_config_file
    cfg.merge_from_file(get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cuda"
    
    return cfg


def evaluate_model(cfg, dataset_name, output_dir):
    """Evaluate model on test set using COCOEvaluator."""
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    
    print("\nRunning evaluation on test set...")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    return results


def nms_numpy(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression (NMS) on boxes (numpy)."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def compute_custom_metrics(cfg, dataset_dicts, nms_threshold=None):
    """Compute additional custom metrics."""
    predictor = DefaultPredictor(cfg)
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    total_pred_conf = []
    total_iou = []

    # For PR curve computation collect all predictions and GTs
    all_predictions = []  # list of dicts {image_id, box, score}
    gt_map = {}  # image_id -> list of gt boxes
    
    print("\nComputing custom metrics...")
    iou_threshold = 0.15
    
    # Debug counters
    total_images = 0
    images_with_gt = 0
    images_with_pred = 0
    debug_sample_shown = False
    
    for idx, record in enumerate(dataset_dicts):
        if idx % 200 == 0:
            print(f"  Processing {idx}/{len(dataset_dicts)}...")
        
        img = cv2.imread(record["file_name"])
        if img is None:
            continue
        
        total_images += 1
        
        # Get predictions
        outputs = predictor(img)
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy()

        if nms_threshold is not None and len(pred_boxes) > 0:
            keep = nms_numpy(pred_boxes, pred_scores, iou_threshold=nms_threshold)
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
        
        if len(pred_boxes) > 0:
            images_with_pred += 1
        
        # Get ground truth
        gt_boxes = []
        for ann in record["annotations"]:
            bbox = ann["bbox"]  # [x1, y1, x2, y2] from COCO format conversion
            gt_boxes.append(bbox)

        # store GTs for PR curve
        image_id = record.get("image_id", idx)
        gt_map[image_id] = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))
        
        if len(gt_boxes) > 0:
            images_with_gt += 1
            
        # Debug: Show first sample with predictions and GT
        if not debug_sample_shown and len(pred_boxes) > 0 and len(gt_boxes) > 0:
            print(f"\n  DEBUG SAMPLE (Image {idx}):")
            print(f"    First GT box: {gt_boxes[0]}")
            print(f"    First pred box: {pred_boxes[0]} (score: {pred_scores[0]:.4f})")
            debug_sample_shown = True
        
        gt_boxes = gt_map[image_id]

        # collect predictions for PR curve
        for pb_idx, pb in enumerate(pred_boxes):
            all_predictions.append({
                "image_id": image_id,
                "box": pb.tolist(),
                "score": float(pred_scores[pb_idx])
            })
        
        # Match predictions to ground truth  
        matched_gt = set()
        matched_preds = set()
        
        # Sort predictions by confidence score (descending) for better matching
        pred_indices = np.argsort(-pred_scores)  # Negative for descending order
        
        for pred_idx in pred_indices:
            pred_box = pred_boxes[pred_idx]
            pred_score = pred_scores[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                # Both are now in [x1, y1, x2, y2] format
                pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
                gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
                
                # Compute IoU
                inter_x_min = max(gt_x1, pred_x1)
                inter_y_min = max(gt_y1, pred_y1)
                inter_x_max = min(gt_x2, pred_x2)
                inter_y_max = min(gt_y2, pred_y2)
                
                if inter_x_max >= inter_x_min and inter_y_max >= inter_y_min:
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                else:
                    inter_area = 0
                
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                union_area = gt_area + pred_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            total_pred_conf.append(pred_score)
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_preds.add(pred_idx)
                total_iou.append(best_iou)
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(total_iou) if total_iou else 0
    mean_conf = np.mean(total_pred_conf) if total_pred_conf else 0
    
    # --- PR curve computation ---
    def compute_pr_curve(all_preds, gt_map, iou_threshold=0.5):
        # Sort predictions by descending score
        all_preds_sorted = sorted(all_preds, key=lambda x: -x["score"])

        # matched GTs per image
        matched = {img_id: set() for img_id in gt_map.keys()}

        precisions = []
        recalls = []
        thresholds = []

        tp_cum = 0
        fp_cum = 0
        total_gts = sum([len(v) for v in gt_map.values()])

        for pred in all_preds_sorted:
            img_id = pred["image_id"]
            pbox = np.array(pred["box"])
            score = pred["score"]

            best_iou = 0
            best_gt_idx = -1
            gts = gt_map.get(img_id, np.empty((0, 4)))

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched.get(img_id, set()):
                    continue
                inter_x_min = max(gt[0], pbox[0])
                inter_y_min = max(gt[1], pbox[1])
                inter_x_max = min(gt[2], pbox[2])
                inter_y_max = min(gt[3], pbox[3])
                if inter_x_max >= inter_x_min and inter_y_max >= inter_y_min:
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                else:
                    inter_area = 0
                gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                pred_area = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                union = gt_area + pred_area - inter_area
                iou = inter_area / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp_cum += 1
                matched.setdefault(img_id, set()).add(best_gt_idx)
            else:
                fp_cum += 1

            prec = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
            rec = tp_cum / total_gts if total_gts > 0 else 0

            precisions.append(prec)
            recalls.append(rec)
            thresholds.append(score)

        return {"precision": precisions, "recall": recalls, "thresholds": thresholds}

    pr_curve = compute_pr_curve(all_predictions, gt_map, iou_threshold=0.5)

    # Debug output
    print(f"\n  Debug Info:")
    print(f"    Total images processed: {total_images}")
    print(f"    Images with GT: {images_with_gt}")
    print(f"    Images with predictions: {images_with_pred}")
    print(f"    Total predictions: {tp + fp}")
    print(f"    Predictions per image: {(tp + fp) / max(total_images, 1):.2f}")
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": mean_iou,
        "mean_confidence": mean_conf,
        "pr_curve": pr_curve
    }


def print_metrics_report(coco_results, custom_metrics):
    """Print a comprehensive metrics report."""
    print("\n" + "=" * 70)
    print("DETECTRON2 MODEL EVALUATION REPORT")
    print("=" * 70)
    
    print("\n📊 COCO Metrics (from COCOEvaluator):")
    print("-" * 70)
    if "bbox" in coco_results:
        bbox_results = coco_results["bbox"]
        print(f"  AP (Average Precision):              {bbox_results.get('AP', -1):.4f}")
        print(f"  AP @ IoU=0.50:                       {bbox_results.get('AP50', -1):.4f}")
        print(f"  AP @ IoU=0.75:                       {bbox_results.get('AP75', -1):.4f}")
        print(f"  AP (small objects):                  {bbox_results.get('APs', -1):.4f}")
        print(f"  AP (medium objects):                 {bbox_results.get('APm', -1):.4f}")
        print(f"  AP (large objects):                  {bbox_results.get('APl', -1):.4f}")
        print(f"  AR (Average Recall) @ 100 boxes:     {bbox_results.get('AR100', -1):.4f}")
    else:
        print("  COCO metrics not available")
    
    print("\n🎯 Custom Metrics (at IoU=0.50):")
    print("-" * 70)
    print(f"  True Positives (TP):                 {custom_metrics['tp']}")
    print(f"  False Positives (FP):                {custom_metrics['fp']}")
    print(f"  False Negatives (FN):                {custom_metrics['fn']}")
    print(f"  Precision:                           {custom_metrics['precision']:.4f}")
    print(f"  Recall:                              {custom_metrics['recall']:.4f}")
    print(f"  F1-Score:                            {custom_metrics['f1_score']:.4f}")
    print(f"  Mean IoU (matched detections):       {custom_metrics['mean_iou']:.4f}")
    print(f"  Mean Confidence (all predictions):   {custom_metrics['mean_confidence']:.4f}")
    
    print("\n" + "=" * 70)


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics_report(coco_results, custom_metrics, output_path):
    """Save metrics report to JSON file."""
    report = {
        "coco_metrics": convert_to_native(coco_results.get("bbox", {})),
        "custom_metrics": convert_to_native(custom_metrics)
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Metrics report saved to: {output_path}")


def save_inference_visualizations(cfg, dataset_dicts, output_dir, num_images=10, random_seed=42, nms_threshold=None):
    """
    Save visualization of inference results on sample images.
    
    Args:
        cfg: Detectron2 config
        dataset_dicts: Dataset dictionary list
        output_dir: Directory to save visualizations
        num_images: Number of images to visualize
        random_seed: Random seed for reproducibility
        nms_threshold: Optional NMS IoU threshold to apply before visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Sample images
    num_to_sample = min(num_images, len(dataset_dicts))
    sampled_indices = random.sample(range(len(dataset_dicts)), num_to_sample)
    
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if cfg.DATASETS.TRAIN else None
    
    print(f"\nSaving inference visualizations ({num_to_sample} images)...")
    
    for sample_idx, data_idx in enumerate(sampled_indices):
        record = dataset_dicts[data_idx]
        img = cv2.imread(record["file_name"])
        
        if img is None:
            print(f"  Warning: Could not load image {record['file_name']}")
            continue
        
        # Run inference
        outputs = predictor(img)
        
        # Visualize predictions
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_img = v.get_image()[:, :, ::-1]
        
        # Save visualization
        image_name = Path(record["file_name"]).stem
        output_path = os.path.join(output_dir, f"{image_name}_pred.jpg")
        cv2.imwrite(output_path, vis_img)
        
        # Always create a comparison with ground truth and predictions
        img_with_gt = img.copy()
        
        # Draw ground truth boxes if available
        gt_boxes = []
        if record.get("annotations"):
            for ann in record["annotations"]:
                x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
                cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for GT
                cv2.putText(img_with_gt, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            gt_boxes = np.array([ann["bbox"] for ann in record["annotations"]])
        else:
            gt_boxes = np.array([])
        
        # Draw predictions on the image with IoU-based coloring
        pred_instances = outputs["instances"].to("cpu")
        pred_boxes = pred_instances.pred_boxes.tensor.numpy()
        pred_scores = pred_instances.scores.numpy()

        # Apply optional NMS before visualization
        if nms_threshold is not None and len(pred_boxes) > 0:
            keep = nms_numpy(pred_boxes, pred_scores, iou_threshold=nms_threshold)
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

        def compute_iou(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            inter_x_min = max(ax1, bx1)
            inter_y_min = max(ay1, by1)
            inter_x_max = min(ax2, bx2)
            inter_y_max = min(ay2, by2)
            if inter_x_max >= inter_x_min and inter_y_max >= inter_y_min:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            else:
                inter_area = 0
            a_area = (ax2 - ax1) * (ay2 - ay1)
            b_area = (bx2 - bx1) * (by2 - by1)
            union = a_area + b_area - inter_area
            return inter_area / union if union > 0 else 0

        for idx_pred, pred_box in enumerate(pred_boxes):
            best_iou = 0
            for gt_box in gt_boxes:
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= 0.15:
                color = (0, 255, 0)  # Green for TP
            else:
                color = (0, 0, 255)  # Red for FP

            x1, y1, x2, y2 = [int(v) for v in pred_box]
            cv2.rectangle(img_with_gt, (x1, y1), (x2, y2), color, 2)
            # Only show the score value
            label = f"{pred_scores[idx_pred]:.2f}"
            # Place text below the box to avoid overlap
            text_y = min(y2 + 12, img_with_gt.shape[0] - 5)
            cv2.putText(img_with_gt, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        combined_img = img_with_gt
        combined_path = os.path.join(output_dir, f"{image_name}_comparison.jpg")
        cv2.imwrite(combined_path, combined_img)
        
        print(f"  ✓ {sample_idx + 1}/{num_to_sample}: {image_name}")
    
    print(f"✓ Visualizations saved to: {output_dir}")


def main(coco_root_dir="./coco_droneRGBT", modality="rgb", checkpoint_path=None, score_threshold=0.5, 
         num_viz_images=10, random_seed=42, save_viz=False, nms_threshold=None):
    """
    Main evaluation function.
    
    Args:
        coco_root_dir: Root directory of COCO dataset
        modality: "rgb" or "thermal"
        checkpoint_path: Path to model checkpoint
        score_threshold: Score threshold for detections
        num_viz_images: Number of images to visualize
        random_seed: Random seed for reproducibility
        save_viz: Whether to save visualizations
    """
    print("Detectron2 Model Evaluation on COCO Dataset")
    print("=" * 70)
    
    # Auto-generate checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = f"./detectron2_checkpoints_coco_{modality}/model_final.pth"
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Please train the model first using:")
        print(f"  python detectron2_finetune.py --modality {modality}")
        return
    
    print(f"\n1. Loading trained model from: {checkpoint_path}")
    cfg = load_trained_model(checkpoint_path, num_classes=1, score_threshold=score_threshold)
    print("✓ Model loaded successfully")
    
    print("\n2. Registering test dataset...")
    try:
        test_dataset, dataset_name = register_coco_test_dataset(coco_root_dir, modality)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Make sure COCO dataset exists at {coco_root_dir}/test_{modality}/")
        return
    
    print("\n3. Evaluating model...")
    eval_output_dir = f"./detectron2_evaluation_coco_{modality}"
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Run COCO evaluation
    coco_results = evaluate_model(cfg, dataset_name, eval_output_dir)
    
    # Compute custom metrics
    custom_metrics = compute_custom_metrics(cfg, test_dataset, nms_threshold=nms_threshold)
    
    # Print report
    print_metrics_report(coco_results, custom_metrics)
    
    # Save report
    save_metrics_report(coco_results, custom_metrics, f"evaluation_metrics_{modality}.json")
    
    # Save visualizations if requested
    if save_viz:
        print("\n4. Saving inference visualizations...")
        viz_output_dir = f"./inference_visualizations_{modality}"
        save_inference_visualizations(cfg, test_dataset, viz_output_dir, num_viz_images, random_seed, nms_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Detectron2 model on COCO format test dataset")
    parser.add_argument("--coco_root", type=str, default="./coco_droneRGBT",
                        help="Root directory of COCO dataset (default: ./coco_droneRGBT)")
    parser.add_argument("--modality", type=str, choices=["rgb", "thermal"], default="rgb",
                        help="Which modality to evaluate (rgb or thermal)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (auto-generated if not provided)")
    parser.add_argument("--score_threshold", type=float, default=0.1429,
                        help="Score threshold for detections (default: 0.1429)")
    parser.add_argument("--nms_threshold", type=float, default=0.15,
                        help="Optional NMS IoU threshold for custom metrics (default: 0.15)")
    parser.add_argument("--save_viz", action="store_true",
                        help="Save inference visualizations")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to visualize (default: 10)")
    parser.add_argument("--vis_seed", type=int, default=42,
                        help="Random seed for visualization reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    main(
        coco_root_dir=args.coco_root,
        modality=args.modality,
        checkpoint_path=args.checkpoint,
        score_threshold=args.score_threshold,
        num_viz_images=args.num_images,
        random_seed=args.vis_seed,
        save_viz=args.save_viz,
        nms_threshold=args.nms_threshold
    )
