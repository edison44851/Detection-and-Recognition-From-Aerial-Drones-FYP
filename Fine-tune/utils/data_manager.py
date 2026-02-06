"""Data management: dataset and dataloader setup."""

from typing import Any, List, Tuple
import logging
import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from datasets.dm_crowd import Crowd
from datasets.crowd import Crowd as Test_Crowd
from datasets.dm_detection import DetectionDataset


def detection_collate(batch: List[dict]) -> dict:
    """Collate function for detection dataset with variable-length annotations."""
    rgb = torch.stack([s['rgb'] for s in batch], 0)
    t = torch.stack([s['t'] for s in batch], 0)
    heatmap = torch.stack([s['heatmap'] for s in batch], 0)
    size = torch.stack([s['size'] for s in batch], 0)
    offset = torch.stack([s['offset'] for s in batch], 0)
    ids = [s['id'] for s in batch]
    points = [s['points'] for s in batch]
    
    return {
        'rgb': rgb,
        't': t,
        'heatmap': heatmap,
        'size': size,
        'offset': offset,
        'id': ids,
        'points': points
    }


def train_collate(batch: List) -> Tuple:
    """Collate function for counting task."""
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        images = [torch.stack(rgb_list, 0), torch.stack(t_list, 0)]
    else:
        images = torch.stack(transposed_batch[0], 0)
    
    points = transposed_batch[1]
    gt_discretes = torch.stack(transposed_batch[2], 0)
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, gt_discretes, st_sizes


class DataManager:
    """Manages dataset and dataloader creation."""
    
    def __init__(self, world_size: int = 1):
        """
        Initialize data manager.
        
        Args:
            world_size: Number of distributed processes
        """
        self.world_size = world_size
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None
    
    def setup_detection_data(self, args: Any, downsample_ratio: int) -> None:
        """
        Setup detection task datasets.
        
        Args:
            args: Configuration
            downsample_ratio: Downsampling ratio for heatmap generation
        """
        # Augmentation parameters
        aug_scale = None
        if hasattr(args, 'aug_scale_min') and hasattr(args, 'aug_scale_max'):
            if args.aug_scale_min != 1.0 or args.aug_scale_max != 1.0:
                aug_scale = (args.aug_scale_min, args.aug_scale_max)
        
        aug_flip = getattr(args, 'aug_flip', False)
        aug_crop_size = getattr(args, 'aug_crop_size', 0)
        thermal_clahe = getattr(args, 'thermal_clahe', True)
        thermal_clahe_clip = getattr(args, 'thermal_clahe_clip', 2.0)
        det_sigma = float(getattr(args, 'det_sigma', 0.8) or 0.8)
        
        # Training dataset
        train_dataset = DetectionDataset(
            args.data_dir,
            split='train',
            output_stride=downsample_ratio,
            sigma=det_sigma,
            aug_scale=aug_scale,
            aug_flip=aug_flip,
            aug_crop_size=aug_crop_size,
            thermal_clahe=thermal_clahe,
            thermal_clahe_clip=thermal_clahe_clip
        )
        
        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset)
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=detection_collate
            )
        else:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=detection_collate
            )
        
        # Validation dataset (optional)
        try:
            val_dataset = DetectionDataset(
                args.data_dir,
                split='val',
                output_stride=downsample_ratio,
                sigma=det_sigma,
                thermal_clahe=thermal_clahe,
                thermal_clahe_clip=thermal_clahe_clip
            )
            if self.world_size > 1:
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
                self.val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    sampler=val_sampler,
                    num_workers=8,
                    pin_memory=True,
                    collate_fn=detection_collate
                )
            else:
                self.val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    collate_fn=detection_collate
                )
        except FileNotFoundError:
            logging.warning('Validation split not found at %s', os.path.join(args.data_dir, 'val'))
            self.val_dataloader = []
        
        # Test dataset (required)
        try:
            test_dataset = DetectionDataset(
                args.data_dir,
                split='test',
                output_stride=downsample_ratio,
                sigma=det_sigma,
                thermal_clahe=thermal_clahe,
                thermal_clahe_clip=thermal_clahe_clip
            )
            if self.world_size > 1:
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
                self.test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=1,
                    sampler=test_sampler,
                    num_workers=8,
                    pin_memory=True,
                    collate_fn=detection_collate
                )
            else:
                self.test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    collate_fn=detection_collate
                )
        except FileNotFoundError as e:
            logging.error('Test split not found at %s', os.path.join(args.data_dir, 'test'))
            raise
    
    def setup_counting_data(self, args: Any, device_count: int) -> None:
        """
        Setup counting task datasets.
        
        Args:
            args: Configuration
            device_count: Number of GPUs
        """
        self.train_dataloader = DataLoader(
            Crowd(os.path.join(args.data_dir, 'train'), args.crop_size, args.downsample_ratio, 'train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers * device_count,
            pin_memory=True,
            collate_fn=train_collate
        )
        
        self.val_dataloader = DataLoader(
            Test_Crowd(os.path.join(args.data_dir, 'val'), method='val'),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        self.test_dataloader = DataLoader(
            Test_Crowd(os.path.join(args.data_dir, 'test'), method='test'),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
