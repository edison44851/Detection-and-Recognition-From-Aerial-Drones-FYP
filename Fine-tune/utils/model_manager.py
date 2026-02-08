"""Model management: creation, checkpoint loading, and component freezing."""

from typing import Any, Dict, Optional, Tuple
import logging
import torch
import torch.nn as nn
from torch.distributed import get_rank, is_available as dist_available

from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.det_model import DetectionHeadWrapper


class ModelManager:
    """Manages model creation, checkpoint loading, and component freezing."""
    
    def __init__(self, device: torch.device, is_distributed: bool, local_rank: int = 0):
        """
        Initialize model manager.
        
        Args:
            device: Torch device (cuda:X or cpu)
            is_distributed: Whether using DDP
            local_rank: Local rank in distributed training
        """
        self.device = device
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.model: Optional[nn.Module] = None
        self._deferred_det_head: Optional[DetectionHeadWrapper] = None
    
    def create_model(self, args: Any) -> nn.Module:
        """
        Create model based on task type.
        
        Args:
            args: Configuration arguments containing task, pretrain flags, etc.
            
        Returns:
            Model instance
        """
        if args.task == 'detection':
            self.model = Swin_BM_RGBT(pre_train=False)
            self._create_detection_head(args)
        else:
            self.model = Swin_BM_RGBT(pre_train=True)
        
        self.model.to(self.device)
        return self.model
    
    def _create_detection_head(self, args: Any) -> None:
        """
        Create detection head adaptor and defer head attachment.
        
        Args:
            args: Configuration arguments
        """
        # Optionally create adaptor
        if getattr(args, 'use_det_adaptor', False):
            self._create_adaptor(args)
        
        # Defer detection head creation until after checkpoint loading
        head_conv = getattr(args, 'head_conv', 256)
        use_deconv = getattr(args, 'use_deconv', False)
        keypoint_only = getattr(args, 'keypoint_mode', False)
        
        self._deferred_det_head = DetectionHeadWrapper(
            in_channels=768,
            hidden=256,
            head_conv=head_conv,
            use_deconv=use_deconv,
            keypoint_only=keypoint_only,
            use_fpn=getattr(args, 'use_fpn', False),
            use_gn=getattr(args, 'det_use_gn', False),
            use_logits=getattr(args, 'use_bce_logits', False)
        )
    
    def _create_adaptor(self, args: Any) -> None:
        """Create detection adaptor (1x1 conv + normalization + ReLU)."""
        try:
            in_ch = 768
            if getattr(args, 'det_use_gn', False):
                # Find appropriate group count
                for g in (32, 16, 8, 4, 2, 1):
                    if in_ch % g == 0:
                        bn_layer = nn.GroupNorm(g, in_ch)
                        break
            else:
                bn_layer = nn.BatchNorm2d(in_ch)
            
            self.model.det_adaptor = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=1),
                bn_layer,
                nn.ReLU(inplace=True)
            )
            logging.info('Created det_adaptor (1x1 conv + %s + ReLU)',
                        'GroupNorm' if getattr(args, 'det_use_gn', False) else 'BatchNorm')
        except Exception as e:
            logging.warning('Failed to create det_adaptor: %s. Continuing without adaptor.', repr(e))
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Optional[Dict], int, Optional[Dict]]:
        """
        Load model checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint (.pth or .tar)
            
        Returns:
            Tuple of (optimizer_state, start_epoch, scheduler_state) or (None, 0, None) if not found
        """
        logging.info('Loading checkpoint from %s', checkpoint_path)
        file_ext = checkpoint_path.rsplit('.', 1)[-1]
        
        try:
            if file_ext == 'tar':
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                state_dict = self._remap_checkpoint_keys(state_dict)
                self._load_state_dict(state_dict)
                optimizer_state = checkpoint.get('optimizer_state_dict', None)
                scheduler_state = checkpoint.get('scheduler_state_dict', None)
                start_epoch = checkpoint.get('epoch', -1) + 1
                return optimizer_state, start_epoch, scheduler_state
            elif file_ext == 'pth':
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                state_dict = self._remap_checkpoint_keys(state_dict)
                self._load_state_dict(state_dict)
                return None, 0, None
        except OSError as e:
            logging.error('Failed to load checkpoint: %s', str(e))
            raise
        except Exception as e:
            logging.warning('Error loading checkpoint, continuing with current weights: %s', repr(e))
            return None, 0
    
    def _remap_checkpoint_keys(self, state_dict: Dict) -> Dict:
        """
        Remap checkpoint keys to match model.
        
        Tries: (1) original, (2) with 'backbone.' prefix, (3) without 'backbone.' prefix
        Returns the variant with best key overlap.
        """
        if not isinstance(self.model, nn.Module):
            return state_dict
        
        try:
            model_keys = set(self.model.state_dict().keys())
            
            # Variant 1: original
            variants = [
                ('original', state_dict),
                ('prefixed', {f'backbone.{k}': v for k, v in state_dict.items()}),
                ('stripped', {k.replace('backbone.', '', 1) if k.startswith('backbone.') else k: v 
                             for k, v in state_dict.items()})
            ]
            
            # Find best match
            best_name, best_dict = 'original', state_dict
            best_overlap = len(model_keys & set(state_dict.keys()))
            
            for name, variant_dict in variants[1:]:
                overlap = len(model_keys & set(variant_dict.keys()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_dict = variant_dict
                    best_name = name
            
            if best_name != 'original':
                logging.info('Using %s key mapping (overlap: %d/%d)', best_name, best_overlap, len(model_keys))
            
            return best_dict
        except Exception as e:
            logging.warning('Key remapping failed: %s. Using original keys.', repr(e))
            return state_dict
    
    def _load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict with error handling."""
        try:
            target_model = self.model
            if self.is_distributed and isinstance(self.model, nn.parallel.DistributedDataParallel):
                target_model = self.model.module
            
            result = target_model.load_state_dict(state_dict, strict=False)
            logging.info('Checkpoint loaded. Missing: %d, Unexpected: %d',
                        len(result.missing_keys), len(result.unexpected_keys))
        except RuntimeError as e:
            logging.warning('Failed to load checkpoint keys: %s', str(e))
        except Exception as e:
            logging.warning('Checkpoint loading error: %s', repr(e))
    
    def attach_detection_head(self) -> None:
        """Attach deferred detection head to model."""
        if not self._deferred_det_head:
            return
        
        try:
            self.model.attach_det_head(self._deferred_det_head)
            logging.info('Attached detection head to model')
            self._deferred_det_head = None
        except AttributeError:
            # Fallback for older model versions
            self.model.det_adaptor = getattr(self.model, 'det_adaptor', nn.Identity())
            self.model.det_head = self._deferred_det_head
            logging.info('Attached detection head (fallback mode)')
        except Exception as e:
            logging.error('Failed to attach detection head: %s', repr(e))
            raise
        
        # Move to device
        try:
            self.model.to(self.device)
        except Exception as e:
            logging.warning('Failed to move model to device: %s', repr(e))
    
    def freeze_components(self, args: Any) -> None:
        """
        Freeze model components based on arguments.
        
        Args:
            args: Configuration with freeze_backbone, freeze_unet, freeze_counter flags
        """
        target_model = self._get_target_model()
        
        # Freeze backbone
        if getattr(args, 'freeze_backbone', False):
            self._freeze_component(target_model, 'backbone', 'backbone parameters')
        
        # Freeze U-Net
        if getattr(args, 'freeze_unet', False):
            self._freeze_component(target_model, 'unet', 'U-Net parameters')
        
        # Freeze counter/regression head
        if getattr(args, 'freeze_counter', False):
            self._freeze_component(target_model, 'reg_layer', 'counting/regression parameters')
        
        # Ensure reg_layer is trainable unless frozen
        if not getattr(args, 'freeze_counter', False):
            if hasattr(target_model, 'reg_layer'):
                for p in target_model.reg_layer.parameters():
                    p.requires_grad = True
    
    def _freeze_component(self, model: nn.Module, attr_name: str, label: str) -> None:
        """Freeze a model component."""
        if not hasattr(model, attr_name):
            logging.warning('Cannot freeze %s: model has no %s attribute', label, attr_name)
            return
        
        component = getattr(model, attr_name)
        for param in component.parameters():
            param.requires_grad = False
        logging.info('Froze %s', label)
    
    def _get_target_model(self) -> nn.Module:
        """Get the actual model (unwrap DDP if needed)."""
        if self.is_distributed and isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model
    
    def wrap_with_ddp(self) -> None:
        """Wrap model with DistributedDataParallel."""
        if not self.is_distributed:
            return
        
        try:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            logging.info('Wrapped model with DDP')
        except Exception as e:
            logging.error('Failed to wrap with DDP: %s', str(e))
            raise
