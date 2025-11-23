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
from datasets.dm_detection import DroneRGBTDetectionDataset
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


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        # Distributed setup
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

        # initialize distributed process group early so DistributedSampler can query world size
        self.is_distributed = self.world_size > 1
        if self.is_distributed:
            dist.init_process_group(backend='nccl', init_method='env://')
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        self.downsample_ratio = args.downsample_ratio
        if args.task in ('detection', 'multi'):
            # use detection dataset which expects DroneRGBT_counting layout
            self.datasets = DroneRGBTDetectionDataset(args.data_dir, split='train', output_stride=args.downsample_ratio)
            if self.world_size > 1:
                from torch.utils.data.distributed import DistributedSampler
                train_sampler = DistributedSampler(self.datasets)
                self.dataloader = DataLoader(self.datasets, batch_size=args.batch_size, sampler=train_sampler,
                                             num_workers=args.num_workers, pin_memory=True)
            else:
                self.dataloader = DataLoader(self.datasets, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=True)

            self.val_dataset = DroneRGBTDetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio)
            if self.world_size > 1:
                val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
                self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, sampler=val_sampler, num_workers=8,
                                                 pin_memory=True)
            else:
                self.val_dataloader = DataLoader(self.val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

            self.test_dataset = DroneRGBTDetectionDataset(args.data_dir, split='test', output_stride=args.downsample_ratio)
            if self.world_size > 1:
                test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
                self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, sampler=test_sampler, num_workers=8,
                                                  pin_memory=True)
            else:
                self.test_dataloader = DataLoader(self.test_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)
        else:
            self.datasets = Crowd(os.path.join(args.data_dir, 'train'), args.crop_size, args.downsample_ratio, 'train')
            self.dataloader = DataLoader(self.datasets, collate_fn=train_collate, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers*self.device_count, pin_memory=True)

            self.val_dataset = Test_Crowd(os.path.join(args.data_dir, 'val'), method='val')
            self.val_dataloader = DataLoader(self.val_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

            self.test_dataset = Test_Crowd(os.path.join(args.data_dir, 'test'), method='test')
            self.test_dataloader = DataLoader(self.test_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

        # instantiate appropriate model for the requested task
        if args.task in ('detection', 'multi'):
            from models.detection.det_model import DetectionModel
            self.model = DetectionModel(backbone_pretrained=False)
        else:
            self.model = Swin_BM_RGBT()
        self.model.to(self.device)

        # process group already initialized above (if applicable)

        # If resuming from a checkpoint, load model weights now so freezing decisions
        # can be applied before constructing the optimizer.
        saved_optimizer_state = None
        saved_start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, map_location=self.device)
                sd = checkpoint.get('model_state_dict', checkpoint)
                # remap key prefixes if checkpoint keys lack the 'backbone.' prefix
                try:
                    model_keys = set(self.model.state_dict().keys())
                    sd_keys = set(sd.keys())
                    # if checkpoint has 'unet.' keys but model expects 'backbone.unet.' keys, prefix them
                    if any(k.startswith('unet.') for k in sd_keys) and any(k.startswith('backbone.unet.') for k in model_keys):
                        sd = {('backbone.' + k): v for k, v in sd.items()}
                    # if checkpoint has 'backbone.' but model keys don't, strip the prefix
                    elif any(k.startswith('backbone.') for k in sd_keys) and any(not k.startswith('backbone.') for k in model_keys):
                        new_sd = {}
                        for k, v in sd.items():
                            if k.startswith('backbone.'):
                                new_sd[k.replace('backbone.', '', 1)] = v
                            else:
                                new_sd[k] = v
                        sd = new_sd
                except Exception:
                    pass
                # load into module when distributed; allow partial loads
                try:
                    if self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        self.model.module.load_state_dict(sd, strict=False)
                    else:
                        self.model.load_state_dict(sd, strict=False)
                except Exception:
                    # fallback to strict=False already used; warn
                    logging.warning('Warning: checkpoint keys did not exactly match model keys; loaded partial state_dict')
                saved_optimizer_state = checkpoint.get('optimizer_state_dict', None)
                saved_start_epoch = checkpoint.get('epoch', -1) + 1
            elif suf == 'pth':
                sd = torch.load(args.resume, map_location=self.device)
                try:
                    model_keys = set(self.model.state_dict().keys())
                    sd_keys = set(sd.keys())
                    if any(k.startswith('unet.') for k in sd_keys) and any(k.startswith('backbone.unet.') for k in model_keys):
                        sd = {('backbone.' + k): v for k, v in sd.items()}
                    elif any(k.startswith('backbone.') for k in sd_keys) and any(not k.startswith('backbone.') for k in model_keys):
                        new_sd = {}
                        for k, v in sd.items():
                            if k.startswith('backbone.'):
                                new_sd[k.replace('backbone.', '', 1)] = v
                            else:
                                new_sd[k] = v
                        sd = new_sd
                except Exception:
                    pass
                try:
                    if self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        self.model.module.load_state_dict(sd, strict=False)
                    else:
                        self.model.load_state_dict(sd, strict=False)
                except Exception:
                    logging.warning('Warning: checkpoint keys did not exactly match model keys; loaded partial state_dict')

        # wrap model with DDP when distributed
        if self.is_distributed:
            # If freezing parts of the model, some parameters may be unused in backward;
            # enable find_unused_parameters when any freeze flag is set to avoid NCCL reduction errors.
            find_unused = bool(getattr(args, 'freeze_backbone', False) or getattr(args, 'freeze_counter', False) or getattr(args, 'freeze_unet', False))
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=find_unused)

        # Optionally freeze backbone or counter (handle wrapped model when DDP)
        # We construct the optimizer after applying freezing so optimizer state matches trainable params.
        if args.task in ('detection', 'multi') and args.freeze_backbone:
            # model may be DDP-wrapped
            backbone = self.model.module.backbone if self.is_distributed else self.model.backbone
            for name, p in backbone.named_parameters():
                p.requires_grad = False
            # optimizer only for head (train detection head when backbone frozen)
            head = self.model.module.head if self.is_distributed else self.model.head
            params = [p for p in head.parameters() if p.requires_grad]
            logging.info('Freezing backbone parameters; training head only.')
            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif getattr(args, 'freeze_unet', False):
            # Freeze only U-Net parameters
            target_model = self.model.module if (self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) else self.model
            if hasattr(target_model, 'unet'):
                for name, p in target_model.unet.named_parameters():
                    p.requires_grad = False
            # build optimizer for remaining trainable params
            params = [p for p in (self.model.parameters() if not self.is_distributed else self.model.module.parameters()) if p.requires_grad]
            if len(params) == 0:
                logging.warning('No trainable parameters remain after freezing U-Net. Check flags.')
            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            logging.info('Freezing U-Net parameters; training remaining params only.')
        elif getattr(args, 'freeze_counter', False):
            # freeze only the counting/regression head (reg_layer)
            target_model = self.model.module if (self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) else self.model
            if hasattr(target_model, 'reg_layer'):
                for name, p in target_model.reg_layer.named_parameters():
                    p.requires_grad = False
            # build optimizer for remaining trainable params
            params = [p for p in (self.model.parameters() if not self.is_distributed else self.model.module.parameters()) if p.requires_grad]
            if len(params) == 0:
                logging.warning('No trainable parameters remain after freezing counter. Check flags.')
            self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            logging.info('Freezing counter/regression parameters; training remaining params only.')
        else:
            logging.info('Training all model parameters')
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Log trainable parameter names and counts (helpful for debugging freeze flags)
        try:
            model_for_params = self.model.module if (self.is_distributed and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)) else self.model
            trainable = [(name, p.numel()) for name, p in model_for_params.named_parameters() if p.requires_grad]
            total = sum([n for _, n in trainable])
            if self.rank == 0:
                logging.info(f"Trainable parameters: total elements={total}, tensors={len(trainable)}")
                # log up to first 80 param names to avoid huge logs
                for name, num in trainable[:80]:
                    logging.info(f"  {name}: {num}")
        except Exception:
            # don't fail setup on logging
            pass

        # If we loaded a checkpoint earlier, restore optimizer state and start epoch
        self.start_epoch = 0
        if saved_optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(saved_optimizer_state)
                self.start_epoch = saved_start_epoch
            except Exception:
                logging.warning('Failed to load optimizer state from checkpoint; continuing with fresh optimizer.')

        self.ot_loss = OT_Loss(args.crop_size, self.downsample_ratio, args.norm_cood, self.device,
                               args.num_of_iter_in_ot,
                               args.reg)
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.count_loss = nn.L1Loss(reduction='sum').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.rd_loss = CL1()

        # detection losses
        self.heatmap_loss = nn.BCELoss(reduction='sum')
        self.l1 = nn.L1Loss(reduction='sum')

        self.val_best_mae = np.inf
        self.val_best_mse = np.inf
        self.best_game = [np.inf, np.inf, np.inf, np.inf]
        self.best_mse = np.inf
        self.best_count = 0
        # detection metric tracking
        self.best_ap = -np.inf
        # combined score tracking for saving strategy
        self.best_combined = -np.inf
        # saving strategy from CLI (default: 'count')
        self.save_by = getattr(args, 'save_by', 'count')
        self.combined_alpha = float(getattr(args, 'combined_alpha', 1.0))
        self.combined_beta = float(getattr(args, 'combined_beta', 1.0))
        # early stopping for detection AP
        self.det_patience = int(getattr(args, 'det_patience', 10))
        self.no_improve_ap = 0
        self.should_stop = False

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
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                mae_is_best, mse_is_best = self.val_epoch()
            # If validation signalled early stop (e.g., detection AP plateau), break training loop
            if getattr(self, 'should_stop', False):
                logging.info('Early stopping triggered at epoch %d', epoch)
                break
            if epoch >= args.val_start and (mse_is_best or mae_is_best):  # or (epoch > 200 and epoch % 5 == 0)):
                self.test_epoch()
        # Clean up distributed resources if used
        if self.is_distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def train_eopch(self):
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        # separate meters for detection loss and total loss to avoid mixing them
        epoch_det_loss = AverageMeter()
        epoch_total_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_rd_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        # Iterate over data.
        dataloader = tqdm(self.dataloader, desc="Training", leave=False, dynamic_ncols=True) if self.rank == 0 else self.dataloader
        for step, batch in enumerate(dataloader):
            # support both counting and detection dataloaders
            if self.args.task in ('detection', 'multi'):
                sample = batch
                inputs = [sample['rgb'].to(self.device), sample['t'].to(self.device)]
                heat_target = sample['heatmap'].to(self.device)
                size_target = sample['size'].to(self.device)
                offset_target = sample['offset'].to(self.device)
                ids = sample.get('id', None)
                # for counting ground-truth, convert heatmap to point-counts
                gd_count = heat_target.view(heat_target.size(0), -1).sum(1).cpu().numpy()
                points = None
                gt_discrete = None
            else:
                inputs, points, gt_discrete, st_sizes = batch

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            # If this is a counting batch, move point tensors to device and compute gd_count
            if points is not None:
                gd_count = np.array([len(p) for p in points], dtype=np.float32)
                points = [p.to(self.device) for p in points]
                gt_discrete = gt_discrete.to(self.device)
            if type(inputs) == list:
                N = inputs[0].size(0)
            else:
                N = inputs.size(0)

            with torch.set_grad_enabled(True):
                rgb, t = inputs
                # For multi-task we need backbone features for RD loss (even if backbone frozen),
                # so request features from the model when task == 'multi'.
                if self.args.task == 'multi':
                    res = self.model(rgb, t, return_feats=True)
                    # DetectionModel returns (density, (heat,size,offset), feats) when return_feats=True
                    if isinstance(res, tuple) and len(res) == 3:
                        outputs, (heat_pred, size_pred, offset_pred), features = res
                    else:
                        # fallback: older models may return (density, dets)
                        outputs, (heat_pred, size_pred, offset_pred) = res
                        features = None
                elif self.args.task == 'detection':
                    outputs, (heat_pred, size_pred, offset_pred) = self.model(rgb, t)
                    # features variable for RD loss compatibility
                    features = None
                else:
                    outputs, features = self.model(rgb, t)
                outputs_sum = outputs.view([outputs.size(0), -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                outputs_normed = outputs / (outputs_sum + 1e-6)

                # Counting losses (when applicable)
                ot_loss = torch.tensor(0.0, device=self.device)
                ot_obj_value = torch.tensor(0.0, device=self.device)
                wd = 0.0
                count_loss = torch.tensor(0.0, device=self.device)
                tv_loss = torch.tensor(0.0, device=self.device)
                rd_loss = torch.tensor(0.0, device=self.device)

                if self.args.task in ('counting', 'multi'):
                    # Compute OT loss.
                    ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
                    ot_loss = ot_loss * self.args.wot
                    ot_obj_value = ot_obj_value * self.args.wot
                    epoch_ot_loss.update(ot_loss.item(), N)
                    epoch_ot_obj_value.update(ot_obj_value.item(), N)
                    epoch_wd.update(wd, N)

                    # Compute counting loss.
                    count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                          torch.from_numpy(gd_count).float().to(self.device))
                    epoch_count_loss.update(count_loss.item(), N)

                    # Compute TV loss.
                    gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(
                        2).unsqueeze(3)
                    gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
                    tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(
                        1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
                    epoch_tv_loss.update(tv_loss.item(), N)

                    # Compute RD loss.
                    rd_loss = self.rd_loss(features, points)

                total_loss = ot_loss + count_loss + tv_loss + rd_loss * self.args.wrd

                # Detection losses (when applicable)
                det_loss = torch.tensor(0.0, device=self.device)
                if self.args.task in ('detection', 'multi'):
                    # heatmap BCE
                    hm_loss = self.heatmap_loss(heat_pred, heat_target)
                    # size/offset only at positive locations
                    pos_mask = (heat_target > 0).float()
                    num_pos = pos_mask.sum().clamp(min=1.0)
                    # expand mask for channels: (B,1,H,W) -> (B,2,H,W)
                    mask2 = pos_mask.repeat(1, 2, 1, 1)
                    size_l = self.l1(size_pred * mask2, size_target * mask2) / num_pos
                    off_l = self.l1(offset_pred * mask2, offset_target * mask2) / num_pos
                    det_loss = hm_loss + size_l + off_l
                    det_loss = det_loss * self.args.det_weight
                    epoch_det_loss.update(det_loss.item(), N)

                    total_loss = total_loss + det_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_total_loss.update(total_loss.item(), N)
                epoch_rd_loss.update(rd_loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_game.update(np.mean(abs(pred_err)), N)
                # ensure count_loss meter is updated when applicable
                try:
                    epoch_count_loss.update(count_loss.item(), N)
                except Exception:
                    pass
        # close tqdm if it was used
        if self.rank == 0 and hasattr(dataloader, 'close'):
            dataloader.close()

        if self.rank == 0:
                logging.info('Epoch {} Train, Count Loss: {:.2f}, Det Loss: {:.2f}, Total Loss: {:.2f}, RD Loss: {:.4f}, GAME0: {:.2f} MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_count_loss.get_avg(), epoch_det_loss.get_avg(), epoch_total_loss.get_avg(), epoch_rd_loss.get_avg(), epoch_game.get_avg(), np.sqrt(epoch_mse.get_avg()),
                         time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        # only rank 0 saves checkpoints
        if self.rank == 0:
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)

    def val_epoch(self):
        self.model.eval()
        epoch_start = time.time()
        total_relative_error = 0
        epoch_res = []

        dataloader = tqdm(self.val_dataloader, desc="Validating", leave=False, dynamic_ncols=True) if self.rank == 0 else self.val_dataloader
        if self.args.task in ('detection', 'multi'):
            preds_per_image = []
            gts_per_image = []
            for sample in dataloader:
                rgb = sample['rgb'].to(self.device)
                t = sample['t'].to(self.device)
                target = sample['heatmap'].to(self.device)
                points = sample['points'].cpu().numpy() if 'points' in sample else np.zeros((0, 2))
                with torch.set_grad_enabled(False):
                    outputs, dets = self.model(rgb, t)
                    # dets: (heat_pred, size_pred, offset_pred)
                    heat_pred = dets[0].detach().cpu().numpy()
                    size_pred = dets[1].detach().cpu().numpy() if dets[1] is not None else None
                    offset_pred = dets[2].detach().cpu().numpy() if dets[2] is not None else None

                    # batch handling: ensure shapes
                    B = heat_pred.shape[0]
                    for i in range(B):
                        hm = heat_pred[i, 0]
                        # extract peaks in output grid coords
                        peaks = heatmap_peaks(hm, min_score=0.01)
                        # convert to pixel coords using output_stride
                        preds_px = []
                        for x_out, y_out, score in peaks:
                            offx = float(offset_pred[i, 0, int(y_out), int(x_out)]) if offset_pred is not None else 0.0
                            offy = float(offset_pred[i, 1, int(y_out), int(x_out)]) if offset_pred is not None else 0.0
                            cx = (x_out + offx) * self.downsample_ratio
                            cy = (y_out + offy) * self.downsample_ratio
                            preds_px.append((cx, cy, float(score)))
                        preds_per_image.append(preds_px)
                        gts_per_image.append(points if points is not None else np.zeros((0, 2)))
                    # counting style metrics for compatibility
                    res = torch.sum(target).item() - torch.sum(outputs).item()
                    epoch_res.append(res)
                    relative_error = eval_relative(outputs, target)
                    total_relative_error += relative_error
        else:
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
                    res = torch.sum(target).item() - torch.sum(outputs).item()
                    epoch_res.append(res)

                    relative_error = eval_relative(outputs, target)
                    total_relative_error += relative_error

        if self.rank == 0 and hasattr(dataloader, 'close'):
            dataloader.close()

        if self.args.task in ('detection', 'multi'):
            # compute AP over dataset (use default thresholds and 4px matching)
            ap, precisions, recalls = compute_ap(preds_per_image, gts_per_image, dist_thresh=4.0)
            logging.info('Epoch {} Val Detection AP: {:.4f}'.format(self.epoch, ap))
            # early-stopping: track AP improvements and stop if no improvement for `det_patience` epochs
            try:
                if ap > self.best_ap:
                    self.best_ap = ap
                    self.no_improve_ap = 0
                else:
                    self.no_improve_ap += 1
                if self.no_improve_ap >= self.det_patience:
                    logging.info('Detection AP did not improve for %d epochs (patience=%d). Stopping early.', self.no_improve_ap, self.det_patience)
                    self.should_stop = True
            except Exception:
                pass
            # Additionally compute detection loss on validation set for diagnostics
            try:
                det_val_loss = 0.0
                cnt = 0
                for sample in self.val_dataloader:
                    rgb = sample['rgb'].to(self.device)
                    t = sample['t'].to(self.device)
                    heat_target = sample['heatmap'].to(self.device)
                    size_target = sample['size'].to(self.device)
                    offset_target = sample['offset'].to(self.device)
                    with torch.set_grad_enabled(False):
                        _, (heat_pred, size_pred, offset_pred) = self.model(rgb, t)
                        hm_loss = self.heatmap_loss(heat_pred, heat_target)
                        pos_mask = (heat_target > 0).float()
                        num_pos = pos_mask.sum().clamp(min=1.0)
                        mask2 = pos_mask.repeat(1, 2, 1, 1)
                        size_l = self.l1(size_pred * mask2, size_target * mask2) / num_pos
                        off_l = self.l1(offset_pred * mask2, offset_target * mask2) / num_pos
                        det_loss = (hm_loss + size_l + off_l) * self.args.det_weight
                        det_val_loss += float(det_loss)
                        cnt += 1
                if cnt > 0:
                    logging.info('Epoch {} Val Detection Loss (avg over samples): {:.4f}'.format(self.epoch, det_val_loss / cnt))
            except Exception:
                pass

        N = len(self.val_dataloader)
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

    def test_epoch(self):
        epoch_start = time.time()
        args = self.args
        self.model.eval()
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]

        dataloader = tqdm(self.test_dataloader, desc="Testing", leave=False, dynamic_ncols=True) if self.rank == 0 else self.test_dataloader
        i = 0
        if args.task in ('detection', 'multi'):
            preds_per_image = []
            gts_per_image = []
            for sample in dataloader:
                i += 1
                rgb = sample['rgb'].to(self.device)
                t = sample['t'].to(self.device)
                if len(rgb.shape) == 3:
                    rgb = rgb.unsqueeze(0)
                    t = t.unsqueeze(0)

                with torch.set_grad_enabled(False):
                    outputs, dets = self.model(rgb, t)
                    heat_pred = dets[0].detach().cpu().numpy()
                    offset_pred = dets[2].detach().cpu().numpy() if dets[2] is not None else None
                    B = heat_pred.shape[0]
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
                        preds_per_image.append(preds_px)
                        points = sample['points'].cpu().numpy() if 'points' in sample else np.zeros((0, 2))
                        gts_per_image.append(points if points is not None else np.zeros((0, 2)))

                    for L in range(4):
                        abs_error, square_error = eval_game(outputs, sample['heatmap'][0], L)
                        game[L] += abs_error
                        mse[L] += square_error

            # compute AP for test set
            ap, precisions, recalls = compute_ap(preds_per_image, gts_per_image, dist_thresh=4.0)
            logging.info('Epoch {} Test Detection AP: {:.4f}'.format(self.epoch, ap))
            # compute detection loss on test set as diagnostic
            try:
                det_test_loss = 0.0
                cnt = 0
                for sample in self.test_dataloader:
                    rgb = sample['rgb'].to(self.device)
                    t = sample['t'].to(self.device)
                    heat_target = sample['heatmap'].to(self.device)
                    size_target = sample['size'].to(self.device)
                    offset_target = sample['offset'].to(self.device)
                    with torch.set_grad_enabled(False):
                        _, (heat_pred, size_pred, offset_pred) = self.model(rgb, t)
                        hm_loss = self.heatmap_loss(heat_pred, heat_target)
                        pos_mask = (heat_target > 0).float()
                        num_pos = pos_mask.sum().clamp(min=1.0)
                        mask2 = pos_mask.repeat(1, 2, 1, 1)
                        size_l = self.l1(size_pred * mask2, size_target * mask2) / num_pos
                        off_l = self.l1(offset_pred * mask2, offset_target * mask2) / num_pos
                        det_loss = (hm_loss + size_l + off_l) * self.args.det_weight
                        det_test_loss += float(det_loss)
                        cnt += 1
                if cnt > 0:
                    logging.info('Epoch {} Test Detection Loss (avg over samples): {:.4f}'.format(self.epoch, det_test_loss / cnt))
            except Exception:
                pass
        else:
            for inputs, target, name in dataloader:
                i += 1
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
                    for L in range(4):
                        abs_error, square_error = eval_game(outputs, target, L)
                        game[L] += abs_error
                        mse[L] += square_error

        if self.rank == 0 and hasattr(dataloader, 'close'):
            dataloader.close()

        N = len(self.test_dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        model_state_dic = self.model.state_dict()

        log_str = 'Test {}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                  'MSE {mse:.2f}, Time cost {time_cost:.1f}s'. \
            format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],
                   time_cost=time.time() - epoch_start)
        logging.info(log_str)

        # Decide whether to save best model according to `--save-by` strategy.
        # Options:
        #  - 'count': preserve old behaviour (lower GAME0 is better)
        #  - 'det': save by detection AP (higher is better)
        #  - 'multi': save if either AP improves or GAME0 improves
        #  - 'combined': save by combined score = alpha * AP - beta * GAME0_norm
        saved = False
        save_by = getattr(self.args, 'save_by', getattr(self, 'save_by', 'count'))

        # helper: normalize GAME0 into (0,1) with GAME0_norm = game0 / (game0 + 1)
        game0 = float(game[0]) if len(game) > 0 else float('inf')
        game0_norm = game0 / (game0 + 1.0 + 1e-12)

        if save_by == 'det':
            # save by AP (requires ap to be defined)
            try:
                if ap > self.best_ap:
                    self.best_ap = ap
                    self.best_epoch = self.epoch
                    logging.info('*****Save Best Detection AP {:.4f} Model Epoch {}'.format(self.best_ap, self.best_epoch))
                    torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))
                    saved = True
            except Exception:
                pass
        elif save_by == 'multi':
            # save if either detection AP improves or counting GAME0 improves
            saved_flag = False
            try:
                if ap > self.best_ap:
                    self.best_ap = ap
                    saved_flag = True
            except Exception:
                pass
            if game0 < self.best_game[0]:
                self.best_mse = mse[0]
                self.best_game = game
                saved_flag = True
            if saved_flag:
                self.best_epoch = self.epoch
                logging.info('*****Save Best Multi-task Model: AP {:.4f} GAME0 {:.2f} Epoch {}'.format(getattr(self, 'best_ap', float('nan')), self.best_game[0], self.best_epoch))
                torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))
                saved = True
        elif save_by == 'combined':
            # combined scoring: alpha * AP - beta * GAME0_norm
            try:
                ap_val = float(ap)
            except Exception:
                ap_val = 0.0
            alpha = float(getattr(self, 'combined_alpha', getattr(self.args, 'combined_alpha', 1.0)))
            beta = float(getattr(self, 'combined_beta', getattr(self.args, 'combined_beta', 1.0)))
            combined = alpha * ap_val - beta * game0_norm
            if combined > self.best_combined:
                self.best_combined = combined
                self.best_epoch = self.epoch
                logging.info('*****Save Best Combined Score {:.6f} (AP {:.4f}, GAME0 {:.4f}) Epoch {}'.format(self.best_combined, ap_val, game0, self.best_epoch))
                torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))
                saved = True
        else:
            # default: count
            if game0 < self.best_game[0]:
                self.best_mse = mse[0]
                self.best_game = game
                self.best_epoch = self.epoch
                logging.info(
                    '*****Save Best GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                    'MSE {mse:.2f} Model Epoch {e}'.format(
                        game0=self.best_game[0], game1=self.best_game[1],
                        game2=self.best_game[2], game3=self.best_game[3],
                        mse=self.best_mse, e=self.best_epoch))
                torch.save(model_state_dic, os.path.join(self.save_dir, "best_model.pth"))
                saved = True
            else:
                logging.info('Best GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
                             'MSE {mse:.2f} Epoch {e}'.format(game0=self.best_game[0], game1=self.best_game[1],
                                                              game2=self.best_game[2], game3=self.best_game[3],
                                                              mse=self.best_mse, e=self.best_epoch))
