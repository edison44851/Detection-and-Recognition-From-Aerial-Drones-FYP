"""Evaluation manager for handling validation and testing logic.

Encapsulates all evaluation routines for both detection and counting tasks,
including metric computation, model selection, and early stopping.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from utils.detection_eval import heatmap_peaks, compute_ap
from utils.evaluation import eval_game, eval_relative


class EvaluationManager:
    """Manages evaluation for detection and counting tasks.
    
    Handles:
    - Metric computation (GAME, MSE, AP)
    - Model selection (best model saving)
    - Early stopping logic
    - Both detection and counting evaluation
    """
    
    def __init__(self, device: torch.device, trainer_instance: Any, is_distributed: bool = False, rank: int = 0):
        """Initialize evaluation manager.
        
        Args:
            device: PyTorch device (cuda or cpu)
            trainer_instance: Reference to trainer for accessing model, losses, etc.
            is_distributed: Whether using distributed training
            rank: Distributed rank (0 for single-GPU)
        """
        self.device = device
        self.trainer = trainer_instance
        self.is_distributed = is_distributed
        self.rank = rank
    
    def evaluate_split(self, dataloader, split_name: str, save_best: bool = False) -> Tuple[bool, bool]:
        """Run evaluation on a dataset split.
        
        Args:
            dataloader: DataLoader for the split
            split_name: Name of split ('Val', 'Test', etc.)
            save_best: Whether to save model if it's best so far
            
        Returns:
            Tuple of (mae_is_best, mse_is_best) for compatibility
        """
        epoch_start = time.time()
        args = self.trainer.args
        self.trainer.model.eval()

        # Metric accumulators
        game: List[float] = [0.0, 0.0, 0.0, 0.0]
        mse: List[float] = [0.0, 0.0, 0.0, 0.0]
        ap = None

        # Progress bar
        desc = "Testing" if split_name.lower() == "test" else "Validating"
        loader = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True) if self.rank == 0 else dataloader

        if args.task == 'detection':
            # Detection evaluation
            all_preds = []
            all_gts = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    sample = batch
                    inputs = [sample['rgb'].to(self.device), sample['t'].to(self.device)]
                    
                    # Forward pass - unpack inputs as separate rgb, t arguments
                    output_dict = self.trainer.model(*inputs)
                    if isinstance(output_dict, dict):
                        heat_pred = output_dict.get('heatmap', output_dict.get('heat', None))
                        offset_pred = output_dict.get('offset', None)
                    else:
                        # Model returns (density, (heat, size, offset)) for detection mode
                        if len(output_dict) >= 2 and isinstance(output_dict[1], tuple) and len(output_dict[1]) >= 3:
                            heat_pred, size_pred, offset_pred = output_dict[1]
                        else:
                            heat_pred, offset_pred = output_dict[:2] if len(output_dict) >= 2 else (output_dict, None)
                    
                    # Extract detections
                    if heat_pred is None:
                        continue
                    preds = self._evaluate_detection_sample(heat_pred, offset_pred, sample)
                    all_preds.extend(preds)
                    all_gts.extend(sample['points'])
            
            # Compute AP
            if len(all_gts) > 0:
                ap_result = compute_ap(all_preds, all_gts, dist_thresh=getattr(args, 'ap_dist_thresh', 8.0))
                if isinstance(ap_result, tuple):
                    ap, precisions, recalls = ap_result
                else:
                    ap, precisions, recalls = ap_result, np.zeros((0,)), np.zeros((0,))
                if self.rank == 0:
                    logging.info(f'{split_name} AP: {ap:.4f}')
                    if precisions.size > 0 and recalls.size > 0:
                        f1 = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
                        best_idx = int(np.argmax(f1))
                        logging.info(
                            f'{split_name} PR: P@bestF1={precisions[best_idx]:.4f} '
                            f'R@bestF1={recalls[best_idx]:.4f} F1={f1[best_idx]:.4f}'
                        )
                        logging.info(
                            f'{split_name} PR (all): P={precisions[-1]:.4f} R={recalls[-1]:.4f}'
                        )
        else:
            # Counting evaluation
            N = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    images, points, gt_discrete, st_sizes = batch
                    if type(images) == list:
                        inputs = [img.to(self.device) for img in images]
                        # Forward pass - unpack as separate rgb, t arguments
                        outputs = self.trainer.model(*inputs)
                    else:
                        inputs = images.to(self.device)
                        outputs = self.trainer.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('density', outputs.get('output', outputs.get('pred', outputs)))
                    
                    if gt_discrete is not None:
                        gt_discrete = gt_discrete.to(self.device)
                    
                    # Compute metrics
                    for i in range(outputs.size(0)):
                        res, relative_error = self._evaluate_counting_sample(outputs[i:i+1], gt_discrete[i:i+1] if gt_discrete is not None else None)
                        game_batch, mse_batch = self._compute_game_metrics(outputs[i:i+1], gt_discrete[i:i+1] if gt_discrete is not None else None)
                        for j in range(4):
                            game[j] += game_batch[j]
                            mse[j] += mse_batch[j]
                        N += 1
            
            if N > 0:
                for j in range(4):
                    game[j] /= N
                    mse[j] /= N

        if self.rank == 0 and hasattr(loader, 'close'):
            loader.close()

        # Log split summary
        log_str = '{} {}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f}, Time cost {time_cost:.1f}s'.format(
                      split_name, 'N/A', game0=game[0], game1=game[1], game2=game[2], game3=game[3],
                      mse=mse[0], time_cost=time.time() - epoch_start)
        logging.info(log_str)

        # Model selection and saving
        should_save, save_msg = self._should_save_best_model(game, ap)

        if should_save or save_best:
            if self.rank == 0:
                save_path = os.path.join(self.trainer.save_dir, f'best_model_epoch_{self.trainer.epoch}.pth')
                torch.save(self.trainer.model.state_dict(), save_path)
                self.trainer.save_list.append(save_path)
                if save_msg:
                    logging.info(save_msg)
        elif save_msg and self.rank == 0:
            logging.info(save_msg)

        # Update early-stopping counters based on detection AP
        if ap is not None:
            if should_save:
                self.trainer.best_ap = ap
                self.trainer.no_improve_ap = 0
            else:
                self.trainer.no_improve_ap += 1
                if self.rank == 0:
                    logging.info(f'AP did not improve ({self.trainer.no_improve_ap}/{self.trainer.det_patience})')

                if self.trainer.no_improve_ap >= self.trainer.det_patience:
                    self.trainer.should_stop = True
                    if self.rank == 0:
                        logging.info(f'Early stopping: AP plateau for {self.trainer.det_patience} evals')

        # Return improvement flags for compatibility
        mae_is_best = should_save
        mse_is_best = should_save

        return mae_is_best, mse_is_best

    def _evaluate_counting_sample(self, outputs: torch.Tensor, target: Optional[torch.Tensor]) -> Tuple[float, float]:
        """Evaluate a single counting sample.
        
        Returns:
            Tuple of (residual, relative_error)
        """
        res = torch.sum(outputs).item() - (torch.sum(target).item() if target is not None else 0)
        relative_error = eval_relative(outputs, target) if target is not None else 0.0
        return float(res), float(relative_error)

    def _evaluate_detection_sample(self, heat_pred: torch.Tensor, offset_pred: Optional[torch.Tensor], 
                                   sample: Optional[Dict] = None) -> List[List[Tuple[float, float, float]]]:
        """Extract detections from prediction heatmaps.
        
        Returns:
            List of detection lists: [[(cx, cy, score), ...], ...]
        """
        B = heat_pred.shape[0]
        all_preds = []
        
        for idx in range(B):
            hm = heat_pred[idx, 0].detach().cpu().numpy()
            
            # Convert logits to probabilities if needed
            if np.nanmin(hm) < 0.0 or np.nanmax(hm) > 1.0:
                hm = 1.0 / (1.0 + np.exp(-hm))
            
            # Extract peaks with NMS
            use_nms = True
            nms_kernel = getattr(self.trainer.args, 'nms_kernel', 3)
            min_score = getattr(self.trainer.args, 'det_score_threshold', 0.5)

            peaks = heatmap_peaks(
                hm,
                min_score=min_score,
                use_nms=use_nms,
                nms_kernel=nms_kernel
            )
            
            preds_px = []
            off_map = None
            if offset_pred is not None:
                off_map = offset_pred[idx].detach().cpu().numpy()
            for x_out, y_out, score in peaks:
                ix = int(x_out)
                iy = int(y_out)
                offx = float(off_map[0, iy, ix]) if off_map is not None else 0.0
                offy = float(off_map[1, iy, ix]) if off_map is not None else 0.0
                # Convert output space to input space (with sub-pixel offsets)
                x_in = (x_out + offx) * self.trainer.downsample_ratio
                y_in = (y_out + offy) * self.trainer.downsample_ratio
                preds_px.append((x_in, y_in, score))

            # Optional NMS/filtering at eval time
            nms_type = getattr(self.trainer.args, 'eval_nms', None)
            if nms_type == 'radius':
                nms_radius = getattr(self.trainer.args, 'eval_nms_radius', 2.0)
                preds_px = self._radius_nms(preds_px, nms_radius)
            elif nms_type == 'soft':
                nms_sigma = getattr(self.trainer.args, 'eval_nms_sigma', 1.0)
                preds_px = self._soft_nms_points(preds_px, nms_sigma)
            
            max_dets = getattr(self.trainer.args, 'max_detections', 300)
            preds_px = sorted(preds_px, key=lambda x: x[2], reverse=True)[:max_dets]
            
            all_preds.append(preds_px)
        
        return all_preds

    @staticmethod
    def _radius_nms(preds: List[Tuple], radius: float) -> List[Tuple]:
        """Apply radius-based NMS to detections."""
        if not preds:
            return preds
        
        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
        keep = []
        for i, (x, y, s) in enumerate(preds_sorted):
            if i == 0:
                keep.append((x, y, s))
            else:
                suppressed = False
                for kx, ky, ks in keep:
                    dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                    if dist < radius:
                        suppressed = True
                        break
                if not suppressed:
                    keep.append((x, y, s))
        return keep

    @staticmethod
    def _soft_nms_points(preds: List[Tuple], sigma: float) -> List[Tuple]:
        """Apply soft-NMS to detections."""
        if not preds:
            return preds
        
        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
        keep = []
        for i, (x, y, s) in enumerate(preds_sorted):
            # Reduce score based on distance to kept detections
            for kx, ky, ks in keep:
                dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                s *= np.exp(-(dist ** 2) / (2 * sigma ** 2))
            
            if s > 0.01:  # Keep if score still above threshold
                keep.append((x, y, s))
        
        return keep

    def _compute_game_metrics(self, outputs: torch.Tensor, target: Optional[torch.Tensor]) -> Tuple[List[float], List[float]]:
        """Compute GAME metrics at all levels.
        
        Returns:
            Tuple of (game_list, mse_list) with 4 levels each
        """
        game: List[float] = [0.0, 0.0, 0.0, 0.0]
        mse: List[float] = [0.0, 0.0, 0.0, 0.0]
        
        if target is None:
            return game, mse
        
        for L in range(4):
            game[L], mse[L] = eval_game(outputs, target, L)
        
        return game, mse

    def _should_save_best_model(self, game: List[float], ap: Optional[float] = None) -> Tuple[bool, str]:
        """Determine if current model is best based on task type.
        
        Detection task saves by AP improvement, counting task saves by GAME0 improvement.
        
        Returns:
            Tuple of (should_save: bool, log_message: str)
        """
        game0 = float(game[0]) if len(game) > 0 else float('inf')
        
        if self.trainer.args.task == 'detection':
            if ap is not None:
                if ap > self.trainer.best_ap:
                    msg = f'Best AP updated: {ap:.4f}'
                    return True, msg
                else:
                    msg = f'AP {ap:.4f} <= best {self.trainer.best_ap:.4f}'
                    return False, msg
            return False, ''
        
        else:
            # Counting task - save by GAME0
            if game0 < self.trainer.best_game[0]:
                self.trainer.best_game = game
                msg = f'Best GAME0 updated: {game0:.2f}'
                return True, msg
            else:
                msg = f'GAME0 {game0:.2f} >= best {self.trainer.best_game[0]:.2f}'
                return False, msg
