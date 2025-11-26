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
        
        # Training dataset
        self.datasets = DetectionDataset(args.data_dir, split='train', output_stride=args.downsample_ratio)
        if self.world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(self.datasets)
            self.dataloader = DataLoader(self.datasets, batch_size=args.batch_size, sampler=train_sampler,
                                         num_workers=args.num_workers, pin_memory=True, collate_fn=detection_collate)
        else:
            self.dataloader = DataLoader(self.datasets, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, pin_memory=True, collate_fn=detection_collate)

        # Validation dataset (optional)
        try:
            self.val_dataset = DetectionDataset(args.data_dir, split='val', output_stride=args.downsample_ratio)
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

        # Test dataset (required)
        try:
            self.test_dataset = DetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio)
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
            from models.detection.center_head import CenterHead
            self.model = Swin_BM_RGBT(pre_train=False)
            # Defer detection head attachment until after checkpoint loading
            self._deferred_det_head = CenterHead(in_channels=768)
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
        self.heatmap_loss = nn.BCELoss(reduction='sum')
        self.l1 = nn.L1Loss(reduction='sum')

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
        
        # Step 6: Wrap with DDP if distributed
        if self.is_distributed:
            find_unused = bool(getattr(args, 'freeze_backbone', False) or 
                             getattr(args, 'freeze_counter', False) or 
                             getattr(args, 'freeze_unet', False))
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=find_unused)
        
        # Step 7: Freeze model components
        self._freeze_model_components()
        
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
        """Compute all detection-related losses
        
        Returns:
            Total detection loss (weighted)
        """
        heat_target = detection_targets['heat_target']
        size_target = detection_targets['size_target']
        offset_target = detection_targets['offset_target']
        
        # Heatmap BCE loss
        hm_loss = self.heatmap_loss(heat_pred, heat_target)
        
        # Size/offset only at positive locations
        pos_mask = (heat_target > 0).float()
        num_pos = pos_mask.sum().clamp(min=1.0)
        mask2 = pos_mask.repeat(1, 2, 1, 1)
        
        size_l = self.l1(size_pred * mask2, size_target * mask2) / num_pos
        off_l = self.l1(offset_pred * mask2, offset_target * mask2) / num_pos
        
        det_loss = (hm_loss + size_l + off_l) * self.args.det_weight
        return det_loss

    def train_eopch(self):
        """Training loop for one epoch"""
        # Initialize meters
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_det_loss = AverageMeter()
        epoch_total_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_rd_loss = AverageMeter()
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
                    total_loss = self._compute_detection_losses(heat_pred, size_pred, offset_pred, detection_targets)
                    epoch_det_loss.update(total_loss.item(), N)
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

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

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
            peaks = heatmap_peaks(hm, min_score=0.01)
            
            preds_px = []
            for x_out, y_out, score in peaks:
                offx = float(offset_pred[idx, 0, int(y_out), int(x_out)]) if offset_pred is not None else 0.0
                offy = float(offset_pred[idx, 1, int(y_out), int(x_out)]) if offset_pred is not None else 0.0
                cx = (x_out + offx) * self.downsample_ratio
                cy = (y_out + offy) * self.downsample_ratio
                preds_px.append((cx, cy, float(score)))
            all_preds.append(preds_px)
        
        return all_preds

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

    def val_epoch(self):
        """Validation epoch - evaluate on validation set"""
        self.model.eval()
        epoch_start = time.time()
        total_relative_error = 0
        epoch_res = []

        dataloader = tqdm(self.val_dataloader, desc="Validating", leave=False, dynamic_ncols=True) if self.rank == 0 else self.val_dataloader
        
        if self.args.task == 'detection':
            preds_per_image = []
            gts_per_image = []
            val_game = [0, 0, 0, 0]
            val_mse = [0, 0, 0, 0]
            
            for sample in dataloader:
                rgb = sample['rgb'].to(self.device)
                t = sample['t'].to(self.device)
                target = sample['heatmap'].to(self.device)
                # points is a list of tensors (one per image in batch)
                points_list = sample.get('points', [])
                
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
                    game, mse = self._compute_game_metrics(outputs, target)
                    for L in range(4):
                        val_game[L] += game[L]
                        val_mse[L] += mse[L]
                    
                    # Standard counting metrics
                    res, relative_error = self._evaluate_counting_sample(outputs, target)
                    epoch_res.append(res)
                    total_relative_error += relative_error
            
            # Log counting GAME metrics
            N_val = len(self.val_dataloader)
            if N_val > 0:
                val_game = [m / N_val for m in val_game]
                val_mse = [torch.sqrt(torch.tensor(m / N_val)) for m in val_mse]
                logging.info('Epoch {} Val Counting: GAME0 {:.2f} GAME1 {:.2f} GAME2 {:.2f} GAME3 {:.2f} MSE {:.2f}'.format(
                    self.epoch, val_game[0], val_game[1], val_game[2], val_game[3], val_mse[0]))
            
            # Compute and log detection AP
            ap, precisions, recalls = compute_ap(preds_per_image, gts_per_image, dist_thresh=4.0)
            logging.info('Epoch {} Val Detection AP: {:.4f}'.format(self.epoch, ap))
            
            # Early stopping check - only if we have actual validation data
            if N_val > 0:
                if ap > self.best_ap:
                    self.best_ap = ap
                    self.no_improve_ap = 0
                else:
                    self.no_improve_ap += 1
                if self.no_improve_ap >= self.det_patience:
                    logging.info('Detection AP did not improve for %d epochs (patience=%d). Stopping early.', 
                               self.no_improve_ap, self.det_patience)
                    self.should_stop = True
        else:
            # Counting-only task
            for inputs, target, name in dataloader:
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
                    outputs, _ = self.model(rgb, t)
                    res, relative_error = self._evaluate_counting_sample(outputs, target)
                    epoch_res.append(res)
                    total_relative_error += relative_error

        if self.rank == 0 and hasattr(dataloader, 'close'):
            dataloader.close()

        # Compute final validation metrics
        N = len(self.val_dataloader)
        if len(epoch_res) == 0 or N == 0:
            logging.warning('Validation set is empty (N=%d, collected_samples=%d). Skipping val metric computation.', N, len(epoch_res))
            mse = float('inf')
            mae = float('inf')
            mae_is_best = False
            mse_is_best = False
            total_relative_error = float('nan')
        else:
            epoch_res = np.array(epoch_res)
            mse = np.sqrt(np.mean(np.square(epoch_res)))
            mae = np.mean(np.abs(epoch_res))
            mae_is_best = mae < self.val_best_mae
            mse_is_best = mse < self.val_best_mse
            total_relative_error = total_relative_error / N
        
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Re: {:.4f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, total_relative_error, time.time() - epoch_start))

        if mae_is_best or mse_is_best:
            self.val_best_mse = mse
            self.val_best_mae = mae
            logging.info("*** Best mse {:.2f} mae {:.2f} model epoch {}".format(self.val_best_mse,
                                                                                 self.val_best_mae,
                                                                                 self.epoch))

        return mae_is_best, mse_is_best

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
        epoch_start = time.time()
        args = self.args
        self.model.eval()
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]

        dataloader = tqdm(self.test_dataloader, desc="Testing", leave=False, dynamic_ncols=True) if self.rank == 0 else self.test_dataloader
        
        if args.task == 'detection':
            preds_per_image = []
            gts_per_image = []
            
            for sample in dataloader:
                rgb = sample['rgb'].to(self.device)
                t = sample['t'].to(self.device)
                target_heatmap = sample['heatmap'].to(self.device)
                # points is a list of tensors
                points_list = sample.get('points', [])
                
                if len(rgb.shape) == 3:
                    rgb = rgb.unsqueeze(0)
                    t = t.unsqueeze(0)

                with torch.set_grad_enabled(False):
                    outputs, dets = self.model(rgb, t)
                    heat_pred = dets[0].detach().cpu().numpy()
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
            N = len(self.test_dataloader)
            test_game = [m / N for m in game]
            test_mse = [torch.sqrt(m / N) for m in mse]
            logging.info('Epoch {} Test Counting: GAME0 {:.2f} GAME1 {:.2f} GAME2 {:.2f} GAME3 {:.2f} MSE {:.2f}'.format(
                self.epoch, test_game[0], test_game[1], test_game[2], test_game[3], test_mse[0]))
            
            # Compute and log detection AP
            ap, precisions, recalls = compute_ap(preds_per_image, gts_per_image, dist_thresh=4.0)
            logging.info('Epoch {} Test Detection AP: {:.4f}'.format(self.epoch, ap))
            
            game = test_game
            mse = test_mse
        else:
            # Counting-only task
            for inputs, target, name in dataloader:
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
                    if isinstance(res, tuple):
                        outputs = res[0]
                    else:
                        outputs = res
                    
                    game_scores, mse_scores = self._compute_game_metrics(outputs, target)
                    for L in range(4):
                        game[L] += game_scores[L]
                        mse[L] += mse_scores[L]

            N = len(self.test_dataloader)
            game = [m / N for m in game]
            mse = [torch.sqrt(m / N) for m in mse]

        if self.rank == 0 and hasattr(dataloader, 'close'):
            dataloader.close()

        # Log test summary
        log_str = 'Test {}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f}, Time cost {time_cost:.1f}s'.format(
                      N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], 
                      mse=mse[0], time_cost=time.time() - epoch_start)
        logging.info(log_str)

        # Determine if this is the best model and save if needed
        # When save_best=True (no validation set), always check and potentially save based on test performance
        ap = locals().get('ap', None)
        should_save, save_msg = self._should_save_best_model(game, ap)
        
        if should_save or save_best:
            if should_save:
                logging.info(save_msg)
            if self.rank == 0:
                model_state_dic = self.model.state_dict()
                torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))
        elif save_msg:
            logging.info(save_msg)
