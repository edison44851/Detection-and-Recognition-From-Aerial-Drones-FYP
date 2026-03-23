"""Detection and counting trainer with distributed training support.

Provides training loop, evaluation, and loss computation for both detection
and counting tasks with proper DDP support.
"""

from typing import Any, Dict, List, Optional, Tuple
from utils.evaluation import eval_game, eval_relative
from utils.detection_eval import heatmap_peaks, compute_ap
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
from utils.model_manager import ModelManager
from utils.data_manager import DataManager
from utils.evaluation_manager import EvaluationManager
from utils.loss_manager import LossManager
from utils.optimizer_builder import OptimizerBuilder
import os
import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
            raise RuntimeError("CUDA is not available but required for training")

        self.is_distributed = self.world_size > 1
        if self.is_distributed:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def _setup_datasets(self) -> None:
        """Initialize datasets via DataManager."""
        args = self.args
        self.downsample_ratio = args.downsample_ratio
        
        # Delegate to DataManager
        if args.task == 'detection':
            self.data_manager.setup_detection_data(args, self.downsample_ratio)
        else:
            self.data_manager.setup_counting_data(args, self.device_count)
        
        # Copy references from manager to trainer
        self.dataloader = self.data_manager.train_dataloader
        self.val_dataloader = getattr(self.data_manager, 'val_dataloader', [])
        self.test_dataloader = getattr(self.data_manager, 'test_dataloader', None)

    def _create_optimizer(self) -> None:
        """Create optimizer via OptimizerBuilder."""
        assert self.model is not None
        self.optimizer = OptimizerBuilder.create_optimizer(self.model, self.args, self.rank)
        self.scheduler = None  # Will be initialized in setup()

    def _initialize_losses(self) -> None:
        """Initialize loss functions via LossManager."""
        self.loss_manager = LossManager(self.device, self.args)

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

    def setup(self) -> None:
        """Initialize datasets, model, losses, and optimizer using manager classes."""
        args = self.args
        
        # Step 1: Setup distributed environment
        self._setup_distributed()
        
        # Step 2: Create manager instances
        self.model_manager = ModelManager(self.device, self.is_distributed, self.local_rank)
        self.data_manager = DataManager(self.world_size)
        self.eval_manager = EvaluationManager(self.device, self, self.is_distributed, self.rank)
        
        # Step 3: Setup datasets via DataManager
        self._setup_datasets()
        
        # Step 4: Create model via ModelManager
        self.model_manager.create_model(args)
        self.model = self.model_manager.model
        
        # Step 5: Load checkpoint if resuming
        saved_optimizer_state = None
        saved_scheduler_state = None
        saved_start_epoch = 0
        if args.resume:
            saved_optimizer_state, saved_start_epoch, saved_scheduler_state = self.model_manager.load_checkpoint(args.resume)
        
        # Step 6: Attach detection head before freezing/DDP
        if args.task == 'detection':
            self.model_manager.attach_detection_head()
            self.model = self.model_manager.model

        # Step 7: Freeze model components BEFORE DDP wrapping
        self.model_manager.freeze_components(args)
        self.model = self.model_manager.model

        # Step 8: Wrap with DDP if distributed
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True, gradient_as_bucket_view=True)
        
        # Step 9: Create optimizer
        self._create_optimizer()
        
        # Step 10: Restore optimizer state if available
        self.start_epoch = 0
        if saved_optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(saved_optimizer_state)
                self.start_epoch = saved_start_epoch
            except RuntimeError as e:
                logging.warning('Failed to load optimizer state: %s. Continuing with fresh optimizer.', repr(e))
        
        # Step 10b: Create learning rate scheduler (Cosine annealing)
        total_epochs = args.max_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=1e-7)
        # Restore scheduler state if available
        if saved_scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(saved_scheduler_state)
                logging.info(f'Loaded scheduler state from checkpoint')
            except Exception as e:
                logging.warning('Failed to load scheduler state: %s. Continuing with fresh scheduler.', repr(e))
        else:
            # Advance scheduler to current epoch if resuming without scheduler state
            for _ in range(self.start_epoch):
                self.scheduler.step()
        logging.info(f'Created CosineAnnealingLR scheduler (T_max={total_epochs}, start_epoch={self.start_epoch})')
        
        # Step 11: Initialize losses and metrics
        self._initialize_losses()
        self._initialize_metrics()

    def train(self):
        """training process"""
        args = self.args
        assert self.model is not None
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            # handle unfreeze epoch
            if args.task in ('detection', 'multi') and args.unfreeze_epoch >= 0 and epoch == args.unfreeze_epoch:
                logging.info('Unfreezing backbone parameters at epoch {}'.format(epoch))
                # handle DDP-wrapped model
                _unwrapped = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
                backbone = getattr(_unwrapped, 'backbone')
                for name, p in backbone.named_parameters():
                    p.requires_grad = True
                # rebuild optimizer to include all params (use smaller lr)
                params = _unwrapped.parameters()
                self.optimizer = optim.Adam(params, lr=args.lr * 0.2, weight_decay=args.weight_decay)
            self.train_epoch()
            mae_is_best = False
            mse_is_best = False
            
            # Check if validation set exists
            has_val = True
            try:
                has_val = len(self.val_dataloader) > 0
            except (TypeError, AttributeError):
                pass
            
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                if has_val:
                    # Use validation set for evaluation and best model selection
                    mae_is_best, mse_is_best = self.val_epoch()
                    # If validation signalled early stop (e.g., detection AP plateau), break training loop
                    if getattr(self, 'should_stop', False):
                        logging.info('Early stopping triggered at epoch %d', epoch)
                        break
                    # Always evaluate test to track best test model even when val split exists
                    if epoch >= args.val_start:
                        self.test_epoch(save_best=True)
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
            except RuntimeError:
                logging.debug('Process group already destroyed or not initialized')

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





    def train_epoch(self) -> None:
        """Training loop for one epoch."""
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
        epoch_start = time.time()
        nan_inf_count = 0  # Track NaN/Inf occurrences
        assert self.model is not None
        self.model.train()

        assert self.dataloader is not None
        dataloader = tqdm(self.dataloader, desc="Training", leave=False, dynamic_ncols=True) if self.rank == 0 else self.dataloader
        for step, batch in enumerate(dataloader):
            # Prepare batch data
            inputs, points, gt_discrete, gd_count, detection_targets = self._prepare_batch(batch)
            
            if isinstance(inputs, list):
                N = inputs[0].size(0)
                rgb, t = inputs[0], inputs[1]
            else:
                assert isinstance(inputs, torch.Tensor)
                N = inputs.size(0)
                rgb, t = inputs, inputs

            with torch.set_grad_enabled(True):
                heat_pred: Optional[torch.Tensor] = None
                size_pred: Optional[torch.Tensor] = None
                offset_pred: Optional[torch.Tensor] = None
                features: Optional[torch.Tensor] = None
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
                
                assert isinstance(outputs, torch.Tensor)
                # Normalize outputs
                outputs_sum = outputs.view([outputs.size(0), -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                outputs_normed = outputs / (outputs_sum + 1e-6)

                # Compute task-specific losses
                if self.args.task == 'detection':
                    # Detection task - only detection losses
                    assert heat_pred is not None and offset_pred is not None and detection_targets is not None
                    det_losses = self.loss_manager.compute_detection_losses(heat_pred, size_pred, offset_pred, detection_targets)
                    total_loss = det_losses['det_loss']
                    
                    # Check for NaN/Inf in loss with diagnostics
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        nan_inf_count += 1
                        logging.warning(f"[Epoch {self.epoch}] NaN/Inf detected at step {step} (total occurrences: {nan_inf_count}). "
                                      f"heat_pred range: [{heat_pred.min():.4f}, {heat_pred.max():.4f}]. Skipping batch.")
                        continue
                    
                    # Log detailed components
                    epoch_det_loss.update(total_loss.item(), N)
                    epoch_hm_pos.update(float(det_losses['pos_loss']) / max(1.0, float(det_losses['num_pixels'])), N)
                    epoch_hm_neg.update(float(det_losses['neg_loss']) / max(1.0, float(det_losses['num_pixels'])), N)
                    epoch_size_loss.update(float(det_losses['size_l']), N)
                    epoch_off_loss.update(float(det_losses['off_l']), N)
                    
                else:
                    # Counting task - only counting losses
                    img_h = int(outputs.size(2) * self.downsample_ratio)
                    img_w = int(outputs.size(3) * self.downsample_ratio)
                    
                    counting_losses = self.loss_manager.compute_counting_losses(
                        outputs, outputs_normed, points, gd_count, gt_discrete, (img_h, img_w), self.downsample_ratio)
                    
                    ot_loss = counting_losses['ot_loss']
                    count_loss = counting_losses['count_loss']
                    tv_loss = counting_losses['tv_loss']
                    
                    epoch_ot_loss.update(ot_loss.item(), N)
                    epoch_ot_obj_value.update(counting_losses['ot_obj_value'].item(), N)
                    epoch_wd.update(counting_losses['wd'], N)
                    epoch_count_loss.update(count_loss.item(), N)
                    epoch_tv_loss.update(tv_loss.item(), N)
                    
                    # RD loss
                    assert features is not None and self.loss_manager.rd_loss is not None
                    rd_loss = self.loss_manager.rd_loss(features, points, image_size=(img_h, img_w))
                    epoch_rd_loss.update(rd_loss.item(), N)
                    
                    total_loss = ot_loss + count_loss + tv_loss + rd_loss * self.args.wrd

                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability (increased from 0.5 to allow sufficient signal)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

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
        if self.rank == 0:
            _dl_close = getattr(dataloader, 'close', None)
            if _dl_close is not None:
                _dl_close()

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
            # Log NaN/Inf statistics
            if nan_inf_count > 0:
                logging.warning(f'Epoch {self.epoch} encountered {nan_inf_count} batches with NaN/Inf loss (skipped)')
            # Log learning rate
            if self.scheduler is not None:
                lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'Epoch {self.epoch} Learning rate: {lr:.2e}')
        
        # Save checkpoint
        model_state_dic = self.model.state_dict()
        if self.rank == 0:
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            checkpoint = {
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }
            # Save scheduler state if available
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(checkpoint, save_path)
            self.save_list.append(save_path)

    def val_epoch(self) -> Tuple[bool, bool]:
        """Validation epoch - delegate to EvaluationManager.
        
        Returns:
            Tuple of (mae_is_best, mse_is_best)
        """
        return self.eval_manager.evaluate_split(self.val_dataloader, 'Val')

    def test_epoch(self, save_best: bool = False) -> None:
        """Test epoch - delegate to EvaluationManager.
        
        Args:
            save_best: If True, save best model based on test performance
        """
        self.eval_manager.evaluate_split(self.test_dataloader, 'Test', save_best=save_best)


# ============================================================================
# Type Hints & Architecture Summary
# ============================================================================
# This trainer is now lean and focused: coordinates data loading, model setup,
# and training. Evaluation is delegated to EvaluationManager for 2000+ lines
# of separation of concerns.
