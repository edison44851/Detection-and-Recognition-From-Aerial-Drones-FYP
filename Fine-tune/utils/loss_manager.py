"""Loss function management and computation for counting and detection tasks.

Handles initialization of all loss functions and computation of both counting
and detection losses with proper numerical stability.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import logging
import numpy as np
from losses.ot_loss import OT_Loss
from losses.LRD import CL1


class LossManager:
    """Manages loss functions and computation for multimodal crowd counting/detection.
    
    Handles initialization of all loss functions (OT, TV, focal, etc.) and provides
    unified methods for computing counting and detection losses.
    """
    
    def __init__(self, device: torch.device, args: Any) -> None:
        """Initialize loss manager with task-specific loss functions.
        
        Args:
            device: torch.device for loss tensors
            args: Configuration object with loss parameters
        """
        self.device = device
        self.args = args
        
        # Counting loss attributes
        self.ot_loss = None
        self.tv_loss = None
        self.count_loss = None
        self.mse = None
        self.mae = None
        self.rd_loss = None
        
        # Detection loss attributes
        self.use_focal = False
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.heatmap_bce_logits = None
        self.heatmap_loss = None
        self.l1 = None
        self.use_bg_suppress = True
        self.bg_suppress_weight = 0.01
        
        # Initialize losses
        self._initialize_losses()
    
    def _initialize_losses(self) -> None:
        """Initialize all loss functions based on configuration."""
        args = self.args
        
        # Get downsample ratio from args if available
        downsample_ratio = getattr(args, 'downsample_ratio', 1)
        crop_size = getattr(args, 'crop_size', 64)
        
        # Counting losses
        self.ot_loss = OT_Loss(
            crop_size, downsample_ratio, args.norm_cood, self.device,
            args.num_of_iter_in_ot, args.reg
        )
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
        self.bg_suppress_weight = getattr(args, 'bg_suppress_weight', 0.01)
    
    def compute_counting_losses(
        self,
        outputs: torch.Tensor,
        outputs_normed: torch.Tensor,
        points: list,
        gd_count: np.ndarray,
        gt_discrete: Optional[torch.Tensor],
        img_size: tuple,
        downsample_ratio: int
    ) -> Dict[str, torch.Tensor]:
        """Compute all counting-related losses.
        
        Args:
            outputs: Density map predictions
            outputs_normed: Normalized density map
            points: List of point annotations per image
            gd_count: Ground truth count per image
            gt_discrete: Ground truth discrete density
            img_size: Image (height, width)
            downsample_ratio: Downsampling ratio
        
        Returns:
            Dict with keys: ot_loss, ot_obj_value, wd, count_loss, tv_loss, N
        """
        N = outputs.size(0)
        img_h, img_w = img_size
        
        # Reinitialize OT loss if grid size changed
        out_h, out_w = outputs.size(2), outputs.size(3)
        if getattr(self.ot_loss, 'output_h', None) != out_h or getattr(self.ot_loss, 'output_w', None) != out_w:
            logging.info('Reinitializing OT_Loss grid to match density size (%d,%d) -> image size (%d,%d)',
                        out_h, out_w, img_h, img_w)
            self.ot_loss = OT_Loss(
                (img_h, img_w), downsample_ratio, self.args.norm_cood,
                self.device, self.args.num_of_iter_in_ot, self.args.reg
            )
        
        # OT loss
        ot_loss, wd, ot_obj_value = self.ot_loss(
            outputs_normed, outputs, points, image_size=img_size
        )
        ot_loss = ot_loss * self.args.wot
        ot_obj_value = ot_obj_value * self.args.wot
        
        # Count loss
        count_loss = self.mae(
            outputs.sum(1).sum(1).sum(1),
            torch.from_numpy(gd_count).float().to(self.device)
        )
        
        # TV loss
        gd_count_tensor = (
            torch.from_numpy(gd_count).float().to(self.device)
            .unsqueeze(1).unsqueeze(2).unsqueeze(3)
        )
        gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
        tv_loss = (
            (self.tv_loss(outputs_normed, gt_discrete_normed)
             .sum(1).sum(1).sum(1) *
             torch.from_numpy(gd_count).float().to(self.device))
            .mean(0) * self.args.wtv
        )
        
        return {
            'ot_loss': ot_loss,
            'ot_obj_value': ot_obj_value,
            'wd': wd,
            'count_loss': count_loss,
            'tv_loss': tv_loss,
            'N': N
        }
    
    def compute_detection_losses(
        self,
        heat_pred: torch.Tensor,
        size_pred: Optional[torch.Tensor],
        offset_pred: torch.Tensor,
        detection_targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all detection-related losses with numerical stability.
        
        Args:
            heat_pred: Heatmap predictions
            size_pred: Size predictions (or None in keypoint mode)
            offset_pred: Offset predictions
            detection_targets: Dict with heat_target, size_target, offset_target
        
        Returns:
            Dict with loss components: det_loss, hm_loss, pos_loss, neg_loss, size_l, off_l, iou_l
        """
        heat_target = detection_targets['heat_target']
        size_target = detection_targets['size_target']
        offset_target = detection_targets['offset_target']
        
        keypoint_mode = getattr(self.args, 'keypoint_mode', False)
        
        # Heatmap loss: BCE or focal-on-logits
        # Apply gradient clipping to predictions for stability
        heat_pred_clipped = torch.clamp(heat_pred, -10, 10)  # Prevent extreme values
        
        pos_mask = (heat_target > 0).float()
        if self.use_focal:
            # Use clipped predictions for focal loss
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
        
        # Background suppression with proper scaling
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
        
        # Proper normalization to prevent exploding loss
        hm_loss = (pos_loss * pos_w) / (num_pos_pixels + 1e-6) + neg_loss / (num_pixels + 1e-6)

        # Size/offset only at positive locations
        num_pos = pos_mask.sum().clamp(min=1.0)
        mask2 = pos_mask.repeat(1, 2, 1, 1)
        
        # Skip size loss in keypoint mode
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

        # Scale detection loss properly
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
