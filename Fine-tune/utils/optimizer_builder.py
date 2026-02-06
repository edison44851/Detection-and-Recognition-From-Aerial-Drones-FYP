"""Optimizer creation and parameter management.

Handles building optimizers with custom parameter groups and logging
trainable parameters for diagnostics.
"""

from typing import Any, List, Tuple
import torch
from torch import optim
import logging


class OptimizerBuilder:
    """Builder for creating optimizers with custom parameter groups.
    
    Supports differential learning rates (e.g., higher LR for detection head)
    and comprehensive parameter logging.
    """
    
    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        args: Any,
        rank: int = 0
    ) -> optim.Optimizer:
        """Create Adam optimizer with optional parameter groups.
        
        Supports head-specific learning rate (via args.head_lr) for differential
        training of detection head vs backbone.
        
        Args:
            model: PyTorch model (should be unwrapped from DDP)
            args: Configuration with lr, weight_decay, head_lr (optional)
            rank: Distributed rank for logging (only rank 0 logs)
        
        Returns:
            Configured Adam optimizer
        """
        # Handle DDP-wrapped model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            target_model = model.module
        else:
            target_model = model
        
        # If a head-specific LR is provided, create optimizer param groups
        head_lr = getattr(args, 'head_lr', None)
        if head_lr is not None:
            return OptimizerBuilder._create_with_param_groups(
                target_model, args, head_lr, rank
            )
        else:
            return OptimizerBuilder._create_standard_optimizer(
                target_model, args, rank
            )
    
    @staticmethod
    def _create_with_param_groups(
        model: torch.nn.Module,
        args: Any,
        head_lr: float,
        rank: int
    ) -> optim.Optimizer:
        """Create optimizer with separate param groups for head and backbone.
        
        Args:
            model: Unwrapped model
            args: Configuration
            head_lr: Learning rate for detection head
            rank: Distributed rank
        
        Returns:
            Optimizer with param groups
        """
        head_params = []
        other_params = []
        
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'det_head' in name or 'det_adaptor' in name:
                head_params.append(p)
            else:
                other_params.append(p)

        total_trainable = len(head_params) + len(other_params)
        if total_trainable == 0:
            if rank == 0:
                logging.warning('No trainable parameters remain after applying freeze flags. Check flags.')
            return optim.Adam([], lr=args.lr, weight_decay=args.weight_decay)

        if rank == 0:
            logging.info(f'Training {total_trainable} parameter tensors (head: {len(head_params)}, other: {len(other_params)})')
        
        param_groups = []
        if len(head_params) > 0:
            param_groups.append({'params': head_params, 'lr': head_lr})
        if len(other_params) > 0:
            param_groups.append({'params': other_params, 'lr': args.lr})

        optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
        
        # Log trainable parameters
        OptimizerBuilder.log_trainable_parameters(model, rank)
        
        return optimizer
    
    @staticmethod
    def _create_standard_optimizer(
        model: torch.nn.Module,
        args: Any,
        rank: int
    ) -> optim.Optimizer:
        """Create standard optimizer with single learning rate.
        
        Args:
            model: Unwrapped model
            args: Configuration
            rank: Distributed rank
        
        Returns:
            Standard Adam optimizer
        """
        params = [p for p in model.parameters() if p.requires_grad]
        
        if len(params) == 0:
            if rank == 0:
                logging.warning('No trainable parameters remain after applying freeze flags. Check flags.')
        else:
            if rank == 0:
                total_elements = sum(p.numel() for p in params)
                logging.info(f'Training {len(params)} parameter tensors (total elements: {total_elements})')
        
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        
        # Log trainable parameters
        OptimizerBuilder.log_trainable_parameters(model, rank)
        
        return optimizer
    
    @staticmethod
    def log_trainable_parameters(model: torch.nn.Module, rank: int = 0) -> None:
        """Log names and counts of trainable parameters.
        
        Args:
            model: PyTorch model
            rank: Distributed rank (only rank 0 logs)
        """
        if rank != 0:
            return
        
        try:
            trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
            total = sum([n for _, n in trainable])
            logging.info(f"Trainable parameters: total elements={total}, tensors={len(trainable)}")
            # Log up to first 80 param names
            for name, num in trainable[:80]:
                logging.info(f"  {name}: {num}")
        except Exception as e:
            logging.debug(f"Failed to log trainable parameters: {e}")
