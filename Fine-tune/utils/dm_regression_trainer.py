from utils.evaluation import eval_game, eval_relative
from utils.detection_eval import heatmap_peaks, compute_ap
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
from models.counting.swin_unet import Swin_BM_RGBT, count_parameters

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.dm_crowd import Crowd
from datasets.crowd import Crowd as Test_Crowd
from datasets.dm_detection import DetectionDataset
from losses.ot_loss import OT_Loss
from losses.LRD import CL1
import torch.distributed as dist


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    gt_discretes = torch.stack(transposed_batch[2], 0)
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, gt_discretes, st_sizes


def detection_collate(batch):
    """Collate function for detection dataset that handles variable-sized points"""
    # batch is a list of dicts from DetectionDataset
    rgb = torch.stack([s['rgb'] for s in batch], 0)
    t = torch.stack([s['t'] for s in batch], 0)
    heatmap = torch.stack([s['heatmap'] for s in batch], 0)
    size = torch.stack([s['size'] for s in batch], 0)
    offset = torch.stack([s['offset'] for s in batch], 0)
    ids = [s['id'] for s in batch]
    # points are variable-sized - keep as list
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


class RegTrainer(Trainer):
    def _setup_distributed(self):
        """Initialize distributed training environment"""
        self.local_rank = getattr(self.args, 'local_rank', int(os.environ.get('LOCAL_RANK', 0)))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus (world_size {})'.format(self.device_count, self.world_size))
            logging.info(f"Current torch seed: {torch.initial_seed()}, Current torch.cuda seed: {torch.cuda.initial_seed()}")
        else:
            raise Exception("gpu is not available")

        self.is_distributed = self.world_size > 1
        if self.is_distributed:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def _setup_datasets(self):
        """Initialize datasets and dataloaders"""
        args = self.args
        self.downsample_ratio = args.downsample_ratio
        
        if args.task == 'detection':
            self._setup_detection_datasets()
        else:
            self._setup_counting_datasets()

    def _setup_detection_datasets(self):
        """Setup datasets for detection task"""
        args = self.args
        
        # Prepare augmentation parameters
        aug_scale = (args.aug_scale_min, args.aug_scale_max) if args.aug_scale_min != 1.0 or args.aug_scale_max != 1.0 else None
        aug_flip = args.aug_flip
        aug_crop_size = args.aug_crop_size
        
        # Prepare thermal preprocessing parameters
        thermal_clahe = getattr(args, 'thermal_clahe', True)
        thermal_clahe_clip = getattr(args, 'thermal_clahe_clip', 2.0)
        
        # Training dataset
        ds_sigma = float(getattr(args, 'det_sigma', 0) or 0)
        if ds_sigma and ds_sigma > 0:
            self.datasets = DetectionDataset(args.data_dir, split='train', output_stride=args.downsample_ratio, sigma=ds_sigma,
                                           aug_scale=aug_scale, aug_flip=aug_flip, aug_crop_size=aug_crop_size,
                                           thermal_clahe=thermal_clahe, thermal_clahe_clip=thermal_clahe_clip)
        else:
            self.datasets = DetectionDataset(args.data_dir, split='train', output_stride=args.downsample_ratio,
                                           aug_scale=aug_scale, aug_flip=aug_flip, aug_crop_size=aug_crop_size,
                                           thermal_clahe=thermal_clahe, thermal_clahe_clip=thermal_clahe_clip)
        if self.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(self.datasets)
            self.dataloader = DataLoader(self.datasets, batch_size=args.batch_size, sampler=train_sampler,
                                         num_workers=args.num_workers, pin_memory=True, collate_fn=detection_collate)
        else:
            self.dataloader = DataLoader(self.datasets, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, pin_memory=True, collate_fn=detection_collate)

        # Validation dataset (optional, no augmentation)
        try:
            if ds_sigma and ds_sigma > 0:
                self.val_dataset = DetectionDataset(args.data_dir, split='val', output_stride=args.downsample_ratio, sigma=ds_sigma,
                                                   thermal_clahe=thermal_clahe, thermal_clahe_clip=thermal_clahe_clip)
            else:
                self.val_dataset = DetectionDataset(args.data_dir, split='val', output_stride=args.downsample_ratio,
                                                   thermal_clahe=thermal_clahe, thermal_clahe_clip=thermal_clahe_clip)
            if self.world_size > 1:
                val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
                self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, sampler=val_sampler, num_workers=8,
                                                 pin_memory=True, collate_fn=detection_collate)
            else:
                self.val_dataloader = DataLoader(self.val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True,
                                                 collate_fn=detection_collate)
        except FileNotFoundError:
            logging.warning(f"Validation split not found at {os.path.join(args.data_dir, 'val')} - will skip validation and use test set for evaluation.")
            self.val_dataset = []
            self.val_dataloader = []

        # Test dataset (required, no augmentation)
        try:
            if ds_sigma and ds_sigma > 0:
                self.test_dataset = DetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio, sigma=ds_sigma,
                                                    thermal_clahe=thermal_clahe, thermal_clahe_clip=thermal_clahe_clip)
            else:
                self.test_dataset = DetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio,
                                                    thermal_clahe=thermal_clahe, thermal_clahe_clip=thermal_clahe_clip)
            if self.world_size > 1:
                test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
                self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, sampler=test_sampler, num_workers=8,
                                                  pin_memory=True, collate_fn=detection_collate)
            else:
                self.test_dataloader = DataLoader(self.test_dataset, 1, shuffle=False, num_workers=8, pin_memory=True,
                                                  collate_fn=detection_collate)
        except FileNotFoundError:
            logging.error(f"Test split not found at {os.path.join(args.data_dir, 'test')} - cannot proceed with detection training.")
            raise

    def _setup_counting_datasets(self):
        """Setup datasets for counting task"""
        args = self.args
        
        self.datasets = Crowd(os.path.join(args.data_dir, 'train'), args.crop_size, args.downsample_ratio, 'train')
        self.dataloader = DataLoader(self.datasets, collate_fn=train_collate, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers*self.device_count, pin_memory=True)

        self.val_dataset = Test_Crowd(os.path.join(args.data_dir, 'val'), method='val')
        self.val_dataloader = DataLoader(self.val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

        self.test_dataset = Test_Crowd(os.path.join(args.data_dir, 'test'), method='test')
        self.test_dataloader = DataLoader(self.test_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

    def _create_model(self):
        """Create model instance based on task"""
        args = self.args
        
        if args.task == 'detection':
            from models.detection.det_model import DetectionHeadWrapper
            self.model = Swin_BM_RGBT(pre_train=False)
            # Optionally add a lightweight adaptor to map fused features to head input
            if getattr(args, 'use_det_adaptor', False):
                try:
                    # Allow using GroupNorm in adaptor when requested
                    in_ch = 768
                    if getattr(args, 'det_use_gn', False):
                        # pick a dividing group count
                        for g in (32, 16, 8, 4, 2, 1):
                            if in_ch % g == 0:
                                gn_groups = g
                                break
                        bn_layer = nn.GroupNorm(gn_groups, in_ch)
                    else:
                        bn_layer = nn.BatchNorm2d(in_ch)

                    self.model.det_adaptor = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=1),
                        bn_layer,
                        nn.ReLU(inplace=True)
                    )
                    logging.info('Initialized det_adaptor (1x1 conv + %s + ReLU) for detection head.',
                                 'GroupNorm' if getattr(args, 'det_use_gn', False) else 'BatchNorm')
                except Exception as e:
                    logging.warning('Failed to initialize det_adaptor: %s', repr(e))
                except Exception as e:
                    logging.warning('Failed to initialize det_adaptor: %s', repr(e))
            # Defer detection head attachment until after checkpoint loading
            head_conv = getattr(args, 'head_conv', 256)
            use_deconv = getattr(args, 'use_deconv', False)
            keypoint_only = getattr(args, 'keypoint_mode', False)
            self._deferred_det_head = DetectionHeadWrapper(
                in_channels=768, hidden=256, 
                head_conv=head_conv, use_deconv=use_deconv, keypoint_only=keypoint_only,
                use_fpn=getattr(args, 'use_fpn', False)
            )
            # Pass use_logits and use_gn to the underlying CenterHead
            self._deferred_det_head.head.use_logits = getattr(args, 'use_bce_logits', False)
            self._deferred_det_head.head.use_gn = getattr(args, 'det_use_gn', False)
        else:
            self.model = Swin_BM_RGBT()
        
        self.model.to(self.device)

    def _remap_checkpoint_keys(self, state_dict):
        """Remap checkpoint keys to match model architecture
        
        Tries three variants (original, prefixed, stripped) and returns
        the one with maximum overlap with current model keys.
        """
        try:
            model_keys = set(self.model.state_dict().keys())
            sd_orig = dict(state_dict)
            sd_keys = set(sd_orig.keys())
            
            # Variant 1: original keys
            best_sd = sd_orig
            best_common = len(model_keys & sd_keys)

            # Variant 2: add 'backbone.' prefix
            prefixed = {('backbone.' + k): v for k, v in sd_orig.items()}
            pref_common = len(model_keys & set(prefixed.keys()))
            if pref_common > best_common:
                best_common = pref_common
                best_sd = prefixed

            # Variant 3: strip 'backbone.' prefix
            stripped = {}
            for k, v in sd_orig.items():
                if k.startswith('backbone.'):
                    stripped[k.replace('backbone.', '', 1)] = v
                else:
                    stripped[k] = v
            strip_common = len(model_keys & set(stripped.keys()))
            if strip_common > best_common:
                best_common = strip_common
                best_sd = stripped

            return best_sd
        except Exception:
            return state_dict

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint from file and return optimizer state and start epoch"""
        logging.info(f'Loading checkpoint from {checkpoint_path}')
        suf = checkpoint_path.rsplit('.', 1)[-1]
        
        saved_optimizer_state = None
        saved_start_epoch = 0
        
        if suf == 'tar':
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            sd = checkpoint.get('model_state_dict', checkpoint)
            sd = self._remap_checkpoint_keys(sd)
            
            self._load_state_dict(sd, '.tar')
            saved_optimizer_state = checkpoint.get('optimizer_state_dict', None)
            saved_start_epoch = checkpoint.get('epoch', -1) + 1
            
        elif suf == 'pth':
            sd = torch.load(checkpoint_path, map_location=self.device)
            sd = self._remap_checkpoint_keys(sd)
            self._load_state_dict(sd, '.pth')
        
        return saved_optimizer_state, saved_start_epoch

    def _load_state_dict(self, state_dict, file_type):
        """Load state dict into model with error handling"""
        try:
            if self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                result = self.model.module.load_state_dict(state_dict, strict=False)
            else:
                result = self.model.load_state_dict(state_dict, strict=False)
            logging.info(f'Checkpoint ({file_type}) loaded successfully. Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}')
        except Exception as e:
            logging.warning(f'Warning: checkpoint keys did not exactly match model keys; loaded partial state_dict. Error: {e}')

    def _attach_detection_head(self):
        """Attach detection head to model (before DDP wrapping)"""
        if hasattr(self, '_deferred_det_head'):
            try:
                target_model = self.model
                target_model.attach_det_head(self._deferred_det_head)
                logging.info('Attached deferred detection head to model before DDP wrapping.')
            except Exception:
                # Fallback for older model versions
                target_model = self.model
                target_model.det_adaptor = getattr(target_model, 'det_adaptor', nn.Identity())
                target_model.det_head = self._deferred_det_head
                logging.info('Attached deferred detection head (fallback assignment) before DDP wrapping.')
            
            try:
                delattr(self, '_deferred_det_head')
            except Exception:
                pass
            
            # Move new submodules to device
            try:
                self.model.to(self.device)
            except Exception:
                logging.warning('Failed to move attached detection head to device with `model.to(device)`; continuing.')

    def _freeze_model_components(self):
        """Apply freezing to model components based on CLI flags"""
        args = self.args
        target_model = self.model.module if (self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) else self.model
        
        # Freeze backbone
        if getattr(args, 'freeze_backbone', False):
            if hasattr(target_model, 'backbone'):
                for name, p in target_model.backbone.named_parameters():
                    p.requires_grad = False
                logging.info('Freezing backbone parameters.')
            else:
                logging.warning('Requested --freeze-backbone but model has no `backbone` attribute.')

        # Freeze U-Net
        if getattr(args, 'freeze_unet', False):
            if hasattr(target_model, 'unet'):
                for name, p in target_model.unet.named_parameters():
                    p.requires_grad = False
                logging.info('Freezing U-Net parameters.')
            else:
                logging.warning('Requested --freeze-unet but model has no `unet` attribute.')

        # Freeze counting head
        if getattr(args, 'freeze_counter', False):
            if hasattr(target_model, 'reg_layer'):
                for name, p in target_model.reg_layer.named_parameters():
                    p.requires_grad = False
                logging.info('Freezing counter/regression (reg_layer) parameters.')
            else:
                logging.warning('Requested --freeze-counter but model has no `reg_layer` attribute.')

        # Ensure reg_layer is trainable unless explicitly frozen
        if not getattr(args, 'freeze_counter', False):
            if hasattr(target_model, 'reg_layer'):
                for p in target_model.reg_layer.parameters():
                    p.requires_grad = True
        
        # Fix DDP gradient stride warning by ensuring gradients are contiguous
        if self.is_distributed:
            def make_grad_contiguous(grad):
                if grad is not None and not grad.is_contiguous():
                    return grad.contiguous()
                return grad
            
            for param in target_model.parameters():
                if param.requires_grad:
                    param.register_hook(make_grad_contiguous)

    def _create_optimizer(self):
        """Create optimizer from trainable parameters"""
        args = self.args
        target_model = self.model.module if (self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) else self.model
        
        # If a head-specific LR is provided, create optimizer param groups so
        # detection head/adaptor params can use a higher lr while keeping other
        # params at the base lr.
        head_lr = getattr(args, 'head_lr', None)
        if head_lr is not None:
            head_params = []
            other_params = []
            for name, p in target_model.named_parameters():
                if not p.requires_grad:
                    continue
                if 'det_head' in name or 'det_adaptor' in name:
                    head_params.append(p)
                else:
                    other_params.append(p)

            total_trainable = len(head_params) + len(other_params)
            if total_trainable == 0:
                logging.warning('No trainable parameters remain after applying freeze flags. Check flags.')
                self.optimizer = optim.Adam([], lr=args.lr, weight_decay=args.weight_decay)
                return

            logging.info(f'Training {total_trainable} parameter tensors (head: {len(head_params)}, other: {len(other_params)})')
            param_groups = []
            if len(head_params) > 0:
                param_groups.append({'params': head_params, 'lr': head_lr})
            if len(other_params) > 0:
                param_groups.append({'params': other_params, 'lr': args.lr})

            self.optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
        else:
            params = [p for p in target_model.parameters() if p.requires_grad]
            if len(params) == 0:
                logging.warning('No trainable parameters remain after applying freeze flags. Check flags.')
            else:
                logging.info(f'Training {len(params)} parameter tensors (total elements: {sum(p.numel() for p in params)})')
            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        
        # Log trainable parameters
        self._log_trainable_parameters(target_model)

    def _log_trainable_parameters(self, model):
        """Log names and counts of trainable parameters"""
        try:
            trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
            total = sum([n for _, n in trainable])
            if self.rank == 0:
                logging.info(f"Trainable parameters: total elements={total}, tensors={len(trainable)}")
                # Log up to first 80 param names
                for name, num in trainable[:80]:
                    logging.info(f"  {name}: {num}")
        except Exception:
            pass

    def _initialize_losses(self):
        """Initialize loss functions"""
        args = self.args
        
        # Counting losses
        self.ot_loss = OT_Loss(args.crop_size, self.downsample_ratio, args.norm_cood, self.device,
                               args.num_of_iter_in_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.count_loss = nn.L1Loss(reduction='sum').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.rd_loss = CL1()

        # Detection losses
        # Heatmap: BCE or focal (both reduction='none' element-wise). Focal requires logits.
        self.use_focal = bool(getattr(args, 'use_focal_heatmap', False))
        self.focal_alpha = float(getattr(args, 'focal_alpha', 0.25))
        self.focal_gamma = float(getattr(args, 'focal_gamma', 2.0))
        if self.use_focal:
            # focal implemented manually on logits; use BCEWithLogits for CE term
            self.heatmap_bce_logits = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        else:
            if getattr(args, 'use_bce_logits', False):
                self.heatmap_loss = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
            else:
                self.heatmap_loss = nn.BCELoss(reduction='none').to(self.device)
        self.l1 = nn.L1Loss(reduction='sum')
        
        # Background suppression loss
        self.use_bg_suppress = getattr(args, 'use_bg_suppress', True)
        self.bg_suppress_weight = getattr(args, 'bg_suppress_weight', 0.01)  # Reduced for stability

    def _initialize_metrics(self):
        """Initialize metric tracking variables"""
        args = self.args
        
        # Counting metrics
        self.val_best_mae = np.inf
        self.val_best_mse = np.inf
        self.best_game = [np.inf, np.inf, np.inf, np.inf]
        self.best_mse = np.inf
        self.best_count = 0
        
        # Detection metrics
        self.best_ap = -np.inf
        
        # Early stopping
        self.det_patience = int(getattr(args, 'det_patience', 10))
        self.no_improve_ap = 0
        self.should_stop = False
        
        # Save handle
        self.save_list = Save_Handle(max_num=args.max_model_num)

    def setup(self):
        """Initialize the datasets, model, loss and optimizer"""
        args = self.args
        
        # Step 1: Setup distributed environment
        self._setup_distributed()
        
        # Step 2: Setup datasets
        self._setup_datasets()
        
        # Step 3: Create model
        self._create_model()
        
        # Step 4: Load checkpoint if resuming
        saved_optimizer_state = None
        saved_start_epoch = 0
        if args.resume:
            saved_optimizer_state, saved_start_epoch = self._load_checkpoint(args.resume)
        
        # Step 5: Attach detection head (before DDP)
        self._attach_detection_head()

        # Step 6: Freeze model components BEFORE DDP wrapping so that
        # DistributedDataParallel sees the correct `requires_grad` flags.
        # This avoids subtle issues where parameters are frozen after DDP
        # wrapping and become unused or excluded from the optimizer.
        self._freeze_model_components()

        # Step 7: Wrap with DDP if distributed
        if self.is_distributed:
            # Set find_unused_parameters=False since we now properly freeze components before DDP
            # and all active parameters are used in forward pass (detection head is always attached)
            find_unused = False
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=find_unused)
        
        # Step 8: Create optimizer
        self._create_optimizer()
        
        # Step 9: Restore optimizer state if available
        self.start_epoch = 0
        if saved_optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(saved_optimizer_state)
                self.start_epoch = saved_start_epoch
            except Exception:
                logging.warning('Failed to load optimizer state from checkpoint; continuing with fresh optimizer.')
        
        # Step 10: Initialize losses and metrics
        self._initialize_losses()
        self._initialize_metrics()

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            # handle unfreeze epoch
            if args.task in ('detection', 'multi') and args.unfreeze_epoch >= 0 and epoch == args.unfreeze_epoch:
                logging.info('Unfreezing backbone parameters at epoch {}'.format(epoch))
                # handle DDP-wrapped model
                backbone = self.model.module.backbone if self.is_distributed else self.model.backbone
                for name, p in backbone.named_parameters():
                    p.requires_grad = True
                # rebuild optimizer to include all params (use smaller lr)
                params = self.model.parameters() if not self.is_distributed else self.model.module.parameters()
                self.optimizer = optim.Adam(params, lr=args.lr * 0.2, weight_decay=args.weight_decay)
            self.train_eopch()
            mae_is_best = False
            mse_is_best = False
            
            # Check if validation set exists
            try:
                has_val = len(self.val_dataloader) > 0
            except Exception:
                has_val = True
            
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                if has_val:
                    # Use validation set for evaluation and best model selection
                    mae_is_best, mse_is_best = self.val_epoch()
                    # If validation signalled early stop (e.g., detection AP plateau), break training loop
                    if getattr(self, 'should_stop', False):
                        logging.info('Early stopping triggered at epoch %d', epoch)
                        break
                    # Run test epoch when validation shows improvement
                    if epoch >= args.val_start and (mse_is_best or mae_is_best):
                        self.test_epoch()
                else:
                    # No validation set - use test set for both evaluation and best model selection
                    logging.info('No validation data found; running test epoch for evaluation and model selection.')
                    self.test_epoch(save_best=True)
                    # If test-based early stopping flagged, break the main training loop
                    if getattr(self, 'should_stop', False):
                        logging.info('Early stopping triggered by test-set AP at epoch %d', epoch)
                        break
        # Clean up distributed resources if used
        if self.is_distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def _prepare_batch(self, batch):
        """Prepare batch data for training/evaluation
        
        Returns:
            Tuple of (inputs, points, gt_discrete, gd_count, detection_targets)
            detection_targets is dict with heat_target, size_target, offset_target or None
        """
        if self.args.task == 'detection':
            sample = batch
            inputs = [sample['rgb'].to(self.device), sample['t'].to(self.device)]
            heat_target = sample['heatmap'].to(self.device)
            size_target = sample['size'].to(self.device)
            offset_target = sample['offset'].to(self.device)
            
            # points is a list of tensors (variable size per image)
            points = sample.get('points', [])
            # For counting ground-truth, use actual point counts
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            gt_discrete = None
            
            detection_targets = {
                'heat_target': heat_target,
                'size_target': size_target,
                'offset_target': offset_target
            }
        else:
            inputs, points, gt_discrete, st_sizes = batch
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            
            if gt_discrete is not None:
                gt_discrete = gt_discrete.to(self.device)
            
            detection_targets = None
        
        return inputs, points, gt_discrete, gd_count, detection_targets

    def _compute_counting_losses(self, outputs, outputs_normed, points, gd_count, gt_discrete, img_size):
        """Compute all counting-related losses
        
        Returns:
            Dict with keys: ot_loss, ot_obj_value, wd, count_loss, tv_loss, rd_loss, features
        """
        N = outputs.size(0)
        img_h, img_w = img_size
        
        # Reinitialize OT loss if grid size changed
        out_h, out_w = outputs.size(2), outputs.size(3)
        if getattr(self.ot_loss, 'output_h', None) != out_h or getattr(self.ot_loss, 'output_w', None) != out_w:
            logging.info('Reinitializing OT_Loss grid to match density size (%d,%d) -> image size (%d,%d)', 
                        out_h, out_w, img_h, img_w)
            self.ot_loss = OT_Loss((img_h, img_w), self.downsample_ratio, self.args.norm_cood, 
                                   self.device, self.args.num_of_iter_in_ot, self.args.reg)
        
        # OT loss
        ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points, image_size=img_size)
        ot_loss = ot_loss * self.args.wot
        ot_obj_value = ot_obj_value * self.args.wot
        
        # Count loss
        count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                             torch.from_numpy(gd_count).float().to(self.device))
        
        # TV loss
        gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
        tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1) * 
                  torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
        
        return {
            'ot_loss': ot_loss,
            'ot_obj_value': ot_obj_value,
            'wd': wd,
            'count_loss': count_loss,
            'tv_loss': tv_loss,
            'N': N
        }

    def _compute_detection_losses(self, heat_pred, size_pred, offset_pred, detection_targets):
        """Compute all detection-related losses - FIXED for numerical stability"""
        heat_target = detection_targets['heat_target']
        size_target = detection_targets['size_target']
        offset_target = detection_targets['offset_target']
        
        keypoint_mode = getattr(self.args, 'keypoint_mode', False)
        
        # Heatmap loss: BCE or focal-on-logits
        # Apply gradient clipping to predictions for stability
        heat_pred_clipped = torch.clamp(heat_pred, -10, 10)  # Prevent extreme values
        
        pos_mask = (heat_target > 0).float()
        if self.use_focal:
            # FIXED: Use clipped predictions for focal loss
            ce = self.heatmap_bce_logits(heat_pred_clipped, heat_target)
            # Sigmoid with clamping for numerical stability
            p = torch.sigmoid(heat_pred_clipped)
            p_t = p * pos_mask + (1.0 - p) * (1.0 - pos_mask)
            focal_factor = (self.focal_alpha * torch.pow(1.0 - p_t, self.focal_gamma)).detach()
            loss_map = focal_factor * ce
        else:
            loss_map = self.heatmap_loss(heat_pred_clipped, heat_target)

        pos_loss = (loss_map * pos_mask).sum()
        
        # Negative mining with gradient clipping for stability
        if getattr(self.args, 'det_neg_topk_ratio', None):
            ratio = float(self.args.det_neg_topk_ratio)
            neg_map = (loss_map * (1.0 - pos_mask)).view(loss_map.size(0), -1)
            k = max(1, int(ratio * neg_map.size(1)))
            topk_vals, _ = torch.topk(neg_map, k, dim=1)
            neg_loss = topk_vals.sum()
        else:
            neg_loss = (loss_map * (1.0 - pos_mask)).sum()
        
        # FIXED: Background suppression with proper scaling
        if self.use_bg_suppress:
            # Only apply to high-confidence background predictions
            bg_mask = (heat_target < 0.01).float()
            # Use sigmoid for probability scale (0-1)
            bg_prob = torch.sigmoid(heat_pred_clipped)
            # Gentle penalty for high background predictions
            bg_penalty = (bg_prob * bg_mask).sum() * self.bg_suppress_weight * 0.01  # Reduced scale
            neg_loss = neg_loss + bg_penalty
        
        # Normalize by number of positive pixels + small constant for stability
        num_pixels = float(loss_map.numel())
        num_pos_pixels = pos_mask.sum().clamp(min=1.0)
        pos_w = float(getattr(self.args, 'det_pos_weight', 1.0))
        
        # FIXED: Proper normalization to prevent exploding loss
        hm_loss = (pos_loss * pos_w) / (num_pos_pixels + 1e-6) + neg_loss / (num_pixels + 1e-6)

        # Size/offset only at positive locations
        num_pos = pos_mask.sum().clamp(min=1.0)
        mask2 = pos_mask.repeat(1, 2, 1, 1)
        
        # PHASE 1: Skip size loss in keypoint mode
        if keypoint_mode or size_pred is None:
            size_l = torch.tensor(0.0, device=self.device)
            iou_l = torch.tensor(0.0, device=self.device)
        else:
            size_l = self.l1(size_pred * mask2, size_target * mask2) / num_pos
            if bool(getattr(self.args, 'use_iou_size', False)):
                wp = (size_pred[:, 0:1] * pos_mask).view(size_pred.size(0), -1)
                hp = (size_pred[:, 1:2] * pos_mask).view(size_pred.size(0), -1)
                wg = (size_target[:, 0:1] * pos_mask).view(size_target.size(0), -1)
                hg = (size_target[:, 1:2] * pos_mask).view(size_target.size(0), -1)
                inter = torch.minimum(wp, wg) * torch.minimum(hp, hg)
                union = (wp * hp + wg * hg - inter).clamp(min=1e-6)
                iou = (inter / union)
                iou_l = (1.0 - iou).sum() / num_pos
                size_l = size_l + float(getattr(self.args, 'iou_weight', 0.5)) * iou_l
        
        off_l = self.l1(offset_pred * mask2, offset_target * mask2) / num_pos

        # FIXED: Scale detection loss properly
        det_loss = (hm_loss * 0.1 + size_l * 0.1 + off_l * 0.1) * self.args.det_weight

        # Return detailed components for logging/diagnostics
        return {
            'det_loss': det_loss,
            'hm_loss': hm_loss,
            'pos_loss': pos_loss,
            'neg_loss': neg_loss,
            'size_l': size_l,
            'off_l': off_l,
            'iou_l': iou_l,
            'num_pixels': num_pixels,
            'num_pos': num_pos
        }

    def train_eopch(self):
        """Training loop for one epoch"""
        # Initialize meters
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_det_loss = AverageMeter()
        epoch_hm_pos = AverageMeter()
        epoch_hm_neg = AverageMeter()
        epoch_size_loss = AverageMeter()
        epoch_off_loss = AverageMeter()
        epoch_grad_norm = AverageMeter()
        epoch_total_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_rd_loss = AverageMeter()
        epoch_bg_suppress = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        dataloader = tqdm(self.dataloader, desc="Training", leave=False, dynamic_ncols=True) if self.rank == 0 else self.dataloader
        for step, batch in enumerate(dataloader):
            # Prepare batch data
            inputs, points, gt_discrete, gd_count, detection_targets = self._prepare_batch(batch)
            
            if type(inputs) == list:
                N = inputs[0].size(0)
                rgb, t = inputs
            else:
                N = inputs.size(0)
                rgb, t = inputs, inputs

            with torch.set_grad_enabled(True):
                # Forward pass based on task
                if self.args.task == 'detection':
                    outputs, (heat_pred, size_pred, offset_pred) = self.model(rgb, t)
                else:
                    # Counting task - request features for RD loss
                    res = self.model(rgb, t, return_feats=True)
                    if isinstance(res, tuple) and len(res) == 2:
                        outputs, features = res
                    else:
                        outputs = res
                        features = None
                
                # Normalize outputs
                outputs_sum = outputs.view([outputs.size(0), -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                outputs_normed = outputs / (outputs_sum + 1e-6)

                # Compute task-specific losses
                if self.args.task == 'detection':
                    # Detection task - only detection losses
                    det_losses = self._compute_detection_losses(heat_pred, size_pred, offset_pred, detection_targets)
                    total_loss = det_losses['det_loss']
                    
                    # FIXED: Check for NaN/Inf in loss
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        logging.warning(f"NaN/Inf detected in loss at step {step}, skipping batch")
                        continue
                    
                    # Log detailed components
                    epoch_det_loss.update(total_loss.item(), N)
                    epoch_hm_pos.update(float(det_losses['pos_loss']) / max(1.0, float(det_losses['num_pixels'])), N)
                    epoch_hm_neg.update(float(det_losses['neg_loss']) / max(1.0, float(det_losses['num_pixels'])), N)
                    epoch_size_loss.update(float(det_losses['size_l']), N)
                    epoch_off_loss.update(float(det_losses['off_l']), N)
                    
                    # Track background suppression
                    if self.use_bg_suppress:
                        bg_mask = (detection_targets['heat_target'] < 0.01).float()
                        bg_prob = torch.sigmoid(torch.clamp(heat_pred, -10, 10))
                        bg_penalty = (bg_prob * bg_mask).mean().item()
                        epoch_bg_suppress.update(bg_penalty, N)
                else:
                    # Counting task - only counting losses
                    img_h = int(outputs.size(2) * self.downsample_ratio)
                    img_w = int(outputs.size(3) * self.downsample_ratio)
                    
                    counting_losses = self._compute_counting_losses(
                        outputs, outputs_normed, points, gd_count, gt_discrete, (img_h, img_w))
                    
                    ot_loss = counting_losses['ot_loss']
                    count_loss = counting_losses['count_loss']
                    tv_loss = counting_losses['tv_loss']
                    
                    epoch_ot_loss.update(ot_loss.item(), N)
                    epoch_ot_obj_value.update(counting_losses['ot_obj_value'].item(), N)
                    epoch_wd.update(counting_losses['wd'], N)
                    epoch_count_loss.update(count_loss.item(), N)
                    epoch_tv_loss.update(tv_loss.item(), N)
                    
                    # RD loss
                    rd_loss = self.rd_loss(features, points, image_size=(img_h, img_w))
                    epoch_rd_loss.update(rd_loss.item(), N)
                    
                    total_loss = ot_loss + count_loss + tv_loss + rd_loss * self.args.wrd

                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # FIXED: Stronger gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()

                # Compute gradient norms for detection head/adaptor (diagnostics)
                if self.rank == 0:
                    # determine the (possibly DDP-wrapped) target model
                    target_model = self.model.module if (self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) else self.model

                    def grad_norm_for_params(named_parameters, substrings):
                        total_sq = 0.0
                        count = 0
                        for n, p in named_parameters:
                            if not p.requires_grad or p.grad is None:
                                continue
                            if any(s in n for s in substrings):
                                total_sq += float((p.grad.data ** 2).sum().cpu().item())
                                count += 1
                        return (total_sq ** 0.5, count)

                    gn, gc = grad_norm_for_params(target_model.named_parameters(), ['det_head', 'det_adaptor'])
                    if gc > 0:
                        # store grad norm for epoch-level reporting instead of printing every batch
                        epoch_grad_norm.update(gn, 1)

                # Update metrics
                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_total_loss.update(total_loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_game.update(np.mean(abs(pred_err)), N)
        
        # Close tqdm
        if self.rank == 0 and hasattr(dataloader, 'close'):
            dataloader.close()

        # Log epoch summary
        if self.rank == 0:
            logging.info('Epoch {} Train, Count Loss: {:.2f}, Det Loss: {:.2f}, Total Loss: {:.2f}, RD Loss: {:.4f}, GAME0: {:.2f} MSE: {:.2f}, Cost {:.1f} sec'
                 .format(self.epoch, epoch_count_loss.get_avg(), epoch_det_loss.get_avg(), 
                        epoch_total_loss.get_avg(), epoch_rd_loss.get_avg(), epoch_game.get_avg(), 
                        np.sqrt(epoch_mse.get_avg()), time.time() - epoch_start))
            # additional diagnostics for detection
            logging.info('Epoch {} Train Detection diagnostics: HM_pos {:.6f}, HM_neg {:.6f}, Size {:.6f}, Off {:.6f}'.format(
                self.epoch, epoch_hm_pos.get_avg(), epoch_hm_neg.get_avg(), epoch_size_loss.get_avg(), epoch_off_loss.get_avg()))
            # Gradient norm diagnostic (averaged over epoch) to avoid per-batch printing
            try:
                logging.info('Epoch {} Avg Grad Norm (det_head/adaptor): {:.6f}'.format(self.epoch, epoch_grad_norm.get_avg()))
            except Exception:
                pass
        
        # Save checkpoint
        model_state_dic = self.model.state_dict()
        if self.rank == 0:
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)

    def _evaluate_counting_sample(self, outputs, target):
        """Evaluate a single counting sample
        
        Returns:
            Tuple of (residual, relative_error)
        """
        res = torch.sum(target).item() - torch.sum(outputs).item()
        relative_error = eval_relative(outputs, target)
        return res, relative_error

    def _evaluate_detection_sample(self, heat_pred, offset_pred, sample=None):
        """Extract detections from prediction heatmaps
        
        Returns:
            List of detection lists (one per image in batch): [[(cx, cy, score), ...], ...]
        """
        B = heat_pred.shape[0]
        all_preds = []
        
        for idx in range(B):
            hm = heat_pred[idx, 0]
            # heat_pred may be raw logits if model was configured with `use_bce_logits`.
            # Convert to probabilities for peak extraction if values are outside [0,1].
            if np.nanmin(hm) < 0.0 or np.nanmax(hm) > 1.0:
                # Numerically stable sigmoid with overflow protection
                with np.errstate(over='ignore', invalid='ignore'):
                    hm_pos = np.where(hm >= 0, 1.0 / (1.0 + np.exp(-np.clip(hm, -500, 500))), 0)
                    hm_neg = np.where(hm < 0, np.exp(np.clip(hm, -500, 0)) / (1.0 + np.exp(np.clip(hm, -500, 0))), 0)
                    hm = hm_pos + hm_neg
            
            # Apply NMS during peak extraction (CenterNet-style)
            use_nms = True  # Enable by default for CenterNet-style heads
            nms_kernel = getattr(self.args, 'nms_kernel', 3)
            peaks = heatmap_peaks(hm, min_score=0.01, use_nms=use_nms, nms_kernel=nms_kernel)
            
            preds_px = []
            for x_out, y_out, score in peaks:
                offx = float(offset_pred[idx, 0, int(y_out), int(x_out)]) if offset_pred is not None else 0.0
                offy = float(offset_pred[idx, 1, int(y_out), int(x_out)]) if offset_pred is not None else 0.0
                cx = (x_out + offx) * self.downsample_ratio
                cy = (y_out + offy) * self.downsample_ratio
                preds_px.append((cx, cy, float(score)))
            # optional eval-time NMS/top-K filtering
            nms_type = getattr(self.args, 'eval_nms', None)
            if nms_type:
                if nms_type == 'radius':
                    rad = float(getattr(self.args, 'eval_nms_radius', 4.0))
                    preds_px = self._radius_nms(preds_px, rad)
                elif nms_type == 'soft':
                    sigma = float(getattr(self.args, 'eval_soft_nms_sigma', 0.5))
                    preds_px = self._soft_nms_points(preds_px, sigma)
            # cap top-200 to avoid huge lists
            preds_px = sorted(preds_px, key=lambda x: x[2], reverse=True)[:200]
            all_preds.append(preds_px)
        
        return all_preds

    @staticmethod
    def _radius_nms(preds, radius):
        if not preds:
            return []
        pts = sorted(preds, key=lambda x: x[2], reverse=True)
        keep = []
        taken = [False] * len(pts)
        r2 = radius * radius
        for i, (x, y, s) in enumerate(pts):
            if taken[i]:
                continue
            keep.append((x, y, s))
            for j in range(i + 1, len(pts)):
                if taken[j]:
                    continue
                x2, y2, _ = pts[j]
                dx = x - x2
                dy = y - y2
                if dx * dx + dy * dy <= r2:
                    taken[j] = True
        return keep

    @staticmethod
    def _soft_nms_points(preds, sigma):
        if not preds:
            return []
        pts = sorted(preds, key=lambda x: x[2], reverse=True)
        result = []
        while pts:
            x, y, s = pts.pop(0)
            result.append((x, y, s))
            new_pts = []
            for x2, y2, s2 in pts:
                dist2 = (x - x2) ** 2 + (y - y2) ** 2
                # gaussian penalty on score
                s2_new = s2 * np.exp(-dist2 / (2.0 * (sigma ** 2)))
                if s2_new > 0.0:
                    new_pts.append((x2, y2, s2_new))
            pts = sorted(new_pts, key=lambda x: x[2], reverse=True)
        return result

    def _compute_game_metrics(self, outputs, target):
        """Compute GAME metrics at all levels
        
        Returns:
            Tuple of (game_list, mse_list) with 4 levels each
        """
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        
        for L in range(4):
            abs_error, square_error = eval_game(outputs, target, L)
            game[L] = abs_error
            mse[L] = square_error
        
        return game, mse

    def _run_eval_split(self, dataloader, split_name, save_best=False):
        """Shared evaluation routine for val/test to keep logic identical."""
        epoch_start = time.time()
        args = self.args
        self.model.eval()

        # Metric accumulators
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        ap = None

        # Progress bar label matches prior behaviour
        desc = "Testing" if split_name.lower() == "test" else "Validating"
        loader = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True) if self.rank == 0 else dataloader

        if args.task == 'detection':
            preds_per_image = []
            gts_per_image = []
            for sample in loader:
                rgb = sample['rgb'].to(self.device)
                t = sample['t'].to(self.device)
                target_heatmap = sample['heatmap'].to(self.device)
                points_list = sample.get('points', [])

                if len(rgb.shape) == 3:
                    rgb = rgb.unsqueeze(0)
                if len(t.shape) == 3:
                    t = t.unsqueeze(0)

                with torch.set_grad_enabled(False):
                    outputs, dets = self.model(rgb, t)
                    heat_pred = dets[0].detach().cpu().numpy()
                    size_pred = dets[1].detach().cpu().numpy() if dets[1] is not None else None
                    offset_pred = dets[2].detach().cpu().numpy() if dets[2] is not None else None

                    # Extract detections - returns list of detection lists
                    batch_preds = self._evaluate_detection_sample(heat_pred, offset_pred, sample)
                    preds_per_image.extend(batch_preds)

                    # Convert points tensors to numpy arrays
                    for pts in points_list:
                        gts_per_image.append(pts.cpu().numpy() if isinstance(pts, torch.Tensor) else pts)

                    # Counting GAME metrics
                    game_scores, mse_scores = self._compute_game_metrics(outputs, target_heatmap[0])
                    for L in range(4):
                        game[L] += game_scores[L]
                        mse[L] += mse_scores[L]

            # Compute and log counting GAME metrics
            N = len(dataloader)
            split_game = [m / N for m in game]
            split_mse = [torch.sqrt(m / N) for m in mse]
            logging.info('Epoch {} {} Counting: GAME0 {:.2f} GAME1 {:.2f} GAME2 {:.2f} GAME3 {:.2f} MSE {:.2f}'.format(
                self.epoch, split_name, split_game[0], split_game[1], split_game[2], split_game[3], split_mse[0]))

            # Compute and log detection AP (use configurable dist threshold in pixels)
            ap, precisions, recalls = compute_ap(preds_per_image, gts_per_image, dist_thresh=float(getattr(self.args, 'ap_dist_thresh', 8.0)))
            logging.info('Epoch {} {} Detection AP (dist {:.1f}px): {:.4f}'.format(self.epoch, split_name, float(getattr(self.args, 'ap_dist_thresh', 8.0)), ap))

            # For downstream logic, reassign game/mse to averaged versions
            game = split_game
            mse = split_mse
        else:
            # Counting-only task
            for inputs, target, name in loader:
                if type(inputs) == list:
                    inputs[0] = inputs[0].to(self.device)
                    inputs[1] = inputs[1].to(self.device)
                else:
                    inputs = inputs.to(self.device)

                if len(inputs[0].shape) == 5:
                    inputs[0] = inputs[0].squeeze(0)
                    inputs[1] = inputs[1].squeeze(0)
                if len(inputs[0].shape) == 3:
                    inputs[0] = inputs[0].unsqueeze(0)
                    inputs[1] = inputs[1].unsqueeze(0)

                with torch.set_grad_enabled(False):
                    rgb, t = inputs
                    res = self.model(rgb, t)
                    outputs = res[0] if isinstance(res, tuple) else res

                    game_scores, mse_scores = self._compute_game_metrics(outputs, target)
                    for L in range(4):
                        game[L] += game_scores[L]
                        mse[L] += mse_scores[L]

            N = len(dataloader)
            game = [m / N for m in game]
            mse = [torch.sqrt(m / N) for m in mse]

        if self.rank == 0 and hasattr(loader, 'close'):
            loader.close()

        # Log split summary
        log_str = '{} {}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f}, Time cost {time_cost:.1f}s'.format(
                      split_name, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3],
                      mse=mse[0], time_cost=time.time() - epoch_start)
        logging.info(log_str)

        # Determine if this is the best model and save if needed
        prev_best_ap = float(self.best_ap) if hasattr(self, 'best_ap') else -np.inf
        should_save, save_msg = self._should_save_best_model(game, ap)

        if should_save or save_best:
            if should_save:
                logging.info(save_msg)
            if self.rank == 0:
                model_state_dic = self.model.state_dict()
                torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))
        elif save_msg:
            logging.info(save_msg)

        # Update early-stopping counters based on detection AP (useful when validation is absent)
        if ap is not None:
            try:
                if ap > prev_best_ap:
                    self.no_improve_ap = 0
                else:
                    self.no_improve_ap += 1
                if self.no_improve_ap >= self.det_patience:
                    logging.info('Detection AP did not improve for %d epochs (patience=%d). Stopping early.',
                                 self.no_improve_ap, self.det_patience)
                    self.should_stop = True
            except Exception:
                pass

        # For callers that expect improvement flags, mirror best-model decision
        mae_is_best = should_save
        mse_is_best = should_save

        return mae_is_best, mse_is_best

    def val_epoch(self):
        """Validation epoch - evaluate on validation set"""
        return self._run_eval_split(self.val_dataloader, 'Val')

    def _should_save_best_model(self, game, ap=None):
        """Determine if current model is best based on task type
        
        Detection task saves by AP improvement, counting task saves by GAME0 improvement.
        
        Returns:
            Tuple of (should_save: bool, log_message: str)
        """
        game0 = float(game[0]) if len(game) > 0 else float('inf')
        
        if self.args.task == 'detection':
            # Detection task: save by AP improvement
            if ap is not None and ap > self.best_ap:
                self.best_ap = ap
                self.best_epoch = self.epoch
                msg = '*****Save Best Detection AP {:.4f} Model Epoch {}'.format(self.best_ap, self.best_epoch)
                return True, msg
        
        else:
            # Counting task: save by GAME0 improvement
            if game0 < self.best_game[0]:
                self.best_mse = self.best_mse if hasattr(self, 'best_mse') else game[0]
                self.best_game = game
                self.best_epoch = self.epoch
                msg = '*****Save Best GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                      'MSE {mse:.2f} Model Epoch {e}'.format(
                          game0=self.best_game[0], game1=self.best_game[1],
                          game2=self.best_game[2], game3=self.best_game[3],
                          mse=self.best_mse, e=self.best_epoch)
                return True, msg
            else:
                msg = 'Best GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                      'MSE {mse:.2f} Epoch {e}'.format(game0=self.best_game[0], game1=self.best_game[1],
                                                       game2=self.best_game[2], game3=self.best_game[3],
                                                       mse=self.best_mse, e=self.best_epoch)
                return False, msg
        
        return False, ""

    def test_epoch(self, save_best=False):
        """Test epoch - evaluate on test set
        
        Args:
            save_best: If True, save best model based on test performance (used when no validation set)
        """
        self._run_eval_split(self.test_dataloader, 'Test', save_best=save_best)
