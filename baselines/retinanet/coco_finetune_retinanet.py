"""
Detectron2 Fine-tuning Script on Custom Dataset
Fine-tunes a pre-trained Detectron2 RetinaNet model on DroneRGBT dataset (COCO format)
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine.hooks import HookBase


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

    # Build category_id -> category_name mapping
    category_map = {cat['id']: cat for cat in coco_data['categories']}

    dataset_root = Path(dataset_root_dir)

    for idx, (image_id, image_info) in enumerate(image_info_map.items()):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(image_info_map)}...")

        # Build full image path
        image_path = dataset_root / image_info['file_name']

        if not image_path.exists():
            print(f"  Warning: Image not found {image_path}")
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
                "bbox": [x, y, x + w, y + h],  # Convert [x, y, w, h] to [x1, y1, x2, y2]
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": ann['category_id'] - 1,  # COCO uses 1-indexed, Detectron2 uses 0-indexed
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

    print(f"OK Loaded {len(dataset_dicts)} valid images with annotations")
    return dataset_dicts


def register_coco_datasets(coco_root_dir="./coco_droneRGBT", modality="rgb"):
    """Register COCO format datasets with Detectron2."""
    coco_root = Path(coco_root_dir)

    # Train dataset
    train_split_dir = coco_root / f"train_{modality}"
    train_json_path = train_split_dir / "annotations.json"

    if train_json_path.exists():
        # Load training data
        train_dicts = load_coco_dataset(str(train_json_path), str(train_split_dir))

        train_dataset_name = f"drone_train_{modality}"

        # Register with a factory function that preserves the data
        def make_train_loader(data=train_dicts):
            return data

        DatasetCatalog.register(train_dataset_name, make_train_loader)
        MetadataCatalog.get(train_dataset_name).set(thing_classes=["person"])
        print(f"OK Registered {train_dataset_name}: {len(train_dicts)} images")
    else:
        raise FileNotFoundError(f"Train annotations not found: {train_json_path}")

    # Test dataset (optional)
    test_split_dir = coco_root / f"test_{modality}"
    test_json_path = test_split_dir / "annotations.json"

    if test_json_path.exists():
        test_dicts = load_coco_dataset(str(test_json_path), str(test_split_dir))
        test_dataset_name = f"drone_test_{modality}"

        def make_test_loader(data=test_dicts):
            return data

        DatasetCatalog.register(test_dataset_name, make_test_loader)
        MetadataCatalog.get(test_dataset_name).set(thing_classes=["person"])
        print(f"OK Registered {test_dataset_name}: {len(test_dicts)} images")

    # Return both train and test dataset names (test may be None)
    return train_dataset_name, (test_dataset_name if test_json_path.exists() else None)


def setup_cfg(output_dir, train_dataset, num_classes=1,
              num_epochs=10, base_lr=0.001, batch_size=16):
    """
    Setup Detectron2 training config.

    Args:
        output_dir: Directory to save checkpoints
        train_dataset: Name of registered training dataset
        num_classes: Number of object classes
        num_epochs: Number of training epochs
        base_lr: Base learning rate
        batch_size: Training batch size

    Returns:
        cfg: Detectron2 config object
    """
    cfg = get_cfg()

    # Use RetinaNet with R50 backbone
    cfg.merge_from_file(get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))

    # Training data
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ()  # No validation during training

    # Model
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    # Solver (optimizer)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 1e-4

    # Learning rate scheduler
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    # NOTE: milestones must be iteration numbers and strictly increasing.
    # We'll compute STEPS after MAX_ITER is set below.
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"

    # Max iterations (approximate)
    cfg.SOLVER.MAX_ITER = num_epochs * 500  # Adjust based on your dataset size
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    # Set LR milestones as fractions of MAX_ITER (ensure strictly increasing)
    s1 = int(cfg.SOLVER.MAX_ITER * 0.6)
    s2 = int(cfg.SOLVER.MAX_ITER * 0.8)
    if s1 < 1:
        s1 = 1
    if s2 <= s1:
        s2 = s1 + 1
    cfg.SOLVER.STEPS = (s1, s2)
    # Early stopping defaults (can be tuned)
    cfg.SOLVER.EARLY_STOP_PATIENCE = 5
    cfg.SOLVER.EARLY_STOP_METRIC = "custom_f1"
    cfg.SOLVER.EARLY_STOP_MAXIMIZE = True

    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1000
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 1000

    # Output
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = "cuda"

    # Misc
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.TEST.EVAL_PERIOD = 500

    return cfg


def get_config_file(model_name):
    """Get path to model config file."""
    from detectron2.model_zoo import get_config_file as get_cfg_file
    return get_cfg_file(model_name)


class CustomTrainer(DefaultTrainer):
    """Custom trainer with custom evaluator."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        # Read early-stopping config from cfg.SOLVER
        patience = getattr(self.cfg.SOLVER, "EARLY_STOP_PATIENCE", None)
        metric_key = getattr(self.cfg.SOLVER, "EARLY_STOP_METRIC", None)
        maximize = getattr(self.cfg.SOLVER, "EARLY_STOP_MAXIMIZE", True)
        if patience is not None and metric_key is not None:
            es_hook = EarlyStoppingHook(patience=patience, metric_key=metric_key, maximize=maximize)
            # insert before the last hook (usually Checkpointer)
            hooks.insert(-1, es_hook)
        return hooks


class EarlyStoppingHook(HookBase):
    """Early stopping hook that monitors a metric and stops training when it plateaus.

    It also saves the best checkpoint as `model_best.pth` in the trainer's output dir.
    """
    def __init__(self, patience=5, metric_key="bbox/AP50", maximize=True, min_delta=1e-4):
        self.patience = int(patience)
        self.metric_key = metric_key
        self.maximize = bool(maximize)
        self.min_delta = float(min_delta)
        self.best = None
        self.counter = 0
        self._last_iter_processed = -1

    def after_step(self):
        latest = self.trainer.storage.latest()
        if self.metric_key not in latest:
            return

        # Only check metrics on evaluation iterations (when new metrics are actually computed)
        current_iter = int(getattr(self.trainer, "iter", -1))
        eval_period = self.trainer.cfg.TEST.EVAL_PERIOD
        if eval_period > 0 and (current_iter % eval_period != 0):
            return

        # Ensure we only process the metric once per evaluation iteration
        if current_iter == self._last_iter_processed:
            return

        val_raw = latest[self.metric_key]
        # Normalize value to a Python float when possible
        if isinstance(val_raw, (list, tuple)):
            if len(val_raw) == 0:
                return
            val_raw = val_raw[0]
        try:
            val = float(val_raw)
        except Exception:
            return
        if val is None:
            return

        if self.best is None:
            self.best = float(val)
            self.counter = 0
            # Save initial best
            try:
                self.trainer.checkpointer.save("model_best")
            except Exception:
                pass
            print(f"[EarlyStoppingHook] Initial {self.metric_key} = {self.best:.6f}; patience counter reset to 0")
            return

        improved = (val > self.best + self.min_delta) if self.maximize else (val < self.best - self.min_delta)
        if improved:
            self.best = val
            self.counter = 0
            try:
                self.trainer.checkpointer.save("model_best")
            except Exception:
                pass
            print(f"[EarlyStoppingHook] Improved {self.metric_key}: {val:.6f} (best -> {self.best:.6f}); counter reset to 0")
        else:
            self.counter += 1
            print(f"[EarlyStoppingHook] No improvement: {self.metric_key}={val:.6f}; counter -> {self.counter}/{self.patience} (best={self.best:.6f})")

        if self.counter >= self.patience:
            print(f"[EarlyStoppingHook] No improvement in '{self.metric_key}' for {self.patience} evaluations. Stopping training.")
            # force end of training loop
            self.trainer.iter = self.trainer.max_iter
        # mark we've processed this iteration's metric
        self._last_iter_processed = current_iter


def main(coco_root_dir="./coco_droneRGBT", modality="rgb", num_epochs=30, base_lr=0.001, batch_size=4,
         early_stop_patience=5, early_stop_metric="custom_f1", early_stop_maximize=True):
    """
    Main training function.

    Args:
        coco_root_dir: Root directory of COCO format dataset
        modality: "rgb" or "thermal"
        num_epochs: Number of training epochs
        base_lr: Base learning rate
        batch_size: Batch size for training
    """
    print("Detectron2 Fine-tuning on DroneRGBT Dataset (COCO Format)")
    print("=" * 60)

    # Register datasets
    print("\n1. Registering datasets...")
    try:
        train_dataset_name, test_dataset_name = register_coco_datasets(coco_root_dir, modality)
    except Exception as e:
        print(f"Error registering datasets: {e}")
        print(f"Make sure COCO dataset exists at {coco_root_dir}")
        return

    # Setup training config
    print("\n2. Setting up training config...")
    output_dir = f"./detectron2_checkpoints_coco_{modality}"
    os.makedirs(output_dir, exist_ok=True)

    cfg = setup_cfg(
        output_dir=output_dir,
        train_dataset=train_dataset_name,
        num_classes=1,  # Single class: person
        num_epochs=num_epochs,
        base_lr=base_lr,
        batch_size=batch_size
    )

    # If a test dataset was registered, enable evaluation during training
    if 'test_dataset_name' in locals() and test_dataset_name is not None:
        cfg.DATASETS.TEST = (test_dataset_name,)

    # Apply early-stopping settings from CLI/runtime args
    cfg.SOLVER.EARLY_STOP_PATIENCE = early_stop_patience
    cfg.SOLVER.EARLY_STOP_METRIC = early_stop_metric
    cfg.SOLVER.EARLY_STOP_MAXIMIZE = early_stop_maximize

    print("Config created:")
    print(f"  Output dir: {output_dir}")
    print(f"  Training dataset: {cfg.DATASETS.TRAIN}")
    print(f"  Num classes: {cfg.MODEL.RETINANET.NUM_CLASSES}")
    print(f"  Base LR: {cfg.SOLVER.BASE_LR}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")

    # Start training
    print("\n3. Starting training...")
    print("-" * 60)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("-" * 60)
    print("OK Training completed")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"  Final model: {output_dir}/model_final.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Detectron2 RetinaNet on COCO format DroneRGBT dataset")
    parser.add_argument("--coco_root", type=str, default="./coco_droneRGBT",
                        help="Root directory of COCO format dataset")
    parser.add_argument("--modality", type=str, choices=["rgb", "thermal"], default="rgb",
                        help="Which modality to train on (rgb or thermal)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (reduce if GPU memory is limited)")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early-stop patience (how many eval steps of no improvement before stopping)")
    parser.add_argument("--early_stop_metric", type=str, default="custom_f1",
                        help="Metric key to monitor for early stopping (e.g. 'custom_f1', 'bbox/AP50' or 'total_loss')")
    parser.add_argument("--early_stop_maximize", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether higher metric is better")

    args = parser.parse_args()

    main(
        coco_root_dir=args.coco_root,
        modality=args.modality,
        num_epochs=args.epochs,
        base_lr=args.lr,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_maximize=args.early_stop_maximize
    )
