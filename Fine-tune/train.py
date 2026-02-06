from utils.dm_regression_trainer import RegTrainer
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train - FIXED for drone dataset stability')

    # ====== DATA & PATHS ======
    parser.add_argument('--data-dir', default=r'',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='directory to save models')
    parser.add_argument('--resume', default='',
                        help='path to resume training model')

    # ====== DEVICE & DISTRIBUTED ======
    parser.add_argument('--device', default='0',
                        help='GPU device IDs to use (default: 0)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training (set by torchrun)')

    # ====== TASK SELECTION & FREEZING ======
    parser.add_argument('--task', type=str, default='counting',
                        help='task to run: counting | detection')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='freeze backbone at start of training')
    parser.add_argument('--freeze-counter', action='store_true',
                        help='freeze counting/regression head at start of training')
    parser.add_argument('--freeze-unet', action='store_true',
                        help='freeze U-Net weights at start of training')
    parser.add_argument('--unfreeze-epoch', type=int, default=-1,
                        help='epoch to unfreeze backbone (-1 to never)')

    # ====== TRAINING GENERAL ======
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop size for input images')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay for optimizer')
    parser.add_argument('--max-epoch', type=int, default=100,
                        help='maximum training epochs')
    parser.add_argument('--val-epoch', type=int, default=2,
                        help='validation frequency (every N epochs)')
    parser.add_argument('--val-start', type=int, default=10,
                        help='epoch to start validation')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='maximum number of checkpoints to keep')

    # ====== DETECTION ARCHITECTURE ======
    parser.add_argument('--downsample-ratio', type=int, default=4,
                        help='network downsample ratio')
    parser.add_argument('--output-stride', type=int, default=None,
                        help='(alias) output stride; overrides downsample-ratio when provided')
    parser.add_argument('--head-conv', type=int, default=256,
                        help='detection head conv channels (CenterNet default: 256)')
    parser.add_argument('--use-deconv', action='store_true',
                        help='use deconv upsampling (CenterNet style); else use bilinear+conv')
    parser.add_argument('--nms-kernel', type=int, default=3,
                        help='NMS kernel size for heatmap decode')
    parser.add_argument('--use-fpn', action='store_true',
                        help='enable lightweight FPN neck')
    parser.add_argument('--use-det-adaptor', action='store_true',
                        help='create 1x1 conv adaptor before detection head')
    parser.add_argument('--det-use-gn', action='store_true',
                        help='use GroupNorm instead of BatchNorm in detection head')

    # ====== KEYPOINT MODE (PHASE 1) ======
    parser.add_argument('--keypoint-mode', action='store_true',
                        help='keypoint-only mode: use heatmap + offset, skip size head')

    # ====== DETECTION LOSS & HEATMAP ======
    parser.add_argument('--det-weight', type=float, default=0.1,
                        help='weight for detection loss in multi-task learning')
    parser.add_argument('--det-pos-weight', type=float, default=7.0,
                        help='positive weight for detection heatmap BCE')
    parser.add_argument('--det-sigma', type=float, default=0.6,
                        help='gaussian sigma for heatmap generation')
    parser.add_argument('--use-bce-logits', action='store_true',
                        help='use BCEWithLogitsLoss (head outputs logits)')
    parser.add_argument('--det-neg-topk-ratio', type=float, default=0.2,
                        help='ratio of hardest negatives to include in heatmap loss')

    # ====== FOCAL LOSS (DETECTION) ======
    parser.add_argument('--use-focal-heatmap', action='store_true',
                        help='use focal loss on heatmap')
    parser.add_argument('--focal-alpha', type=float, default=0.85,
                        help='focal loss alpha (class balance)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='focal loss gamma (hard example focus)')

    # ====== SIZE LOSS & IoU ======
    parser.add_argument('--use-iou-size', action='store_true',
                        help='add IoU-based loss for size regression')
    parser.add_argument('--iou-weight', type=float, default=0.3,
                        help='weight for IoU size loss')

    # ====== FALSE POSITIVE REDUCTION (STABILITY) ======
    parser.add_argument('--boundary-suppress', action='store_true', default=True,
                        help='apply boundary suppression to reduce false positives')
    parser.add_argument('--suppress-margin', type=int, default=4,
                        help='margin size for boundary suppression')
    parser.add_argument('--use-bg-suppress', action='store_true', default=True,
                        help='use background suppression loss')
    parser.add_argument('--bg-suppress-weight', type=float, default=0.01,
                        help='weight for background suppression loss')
    parser.add_argument('--adaptive-threshold', action='store_true', default=True,
                        help='use adaptive threshold based on image statistics')
    parser.add_argument('--filter-boundary-dets', action='store_true', default=True,
                        help='filter detections near image boundaries')
    parser.add_argument('--count-aware-filtering', action='store_true', default=True,
                        help='limit detections based on expected count')

    # ====== DETECTION HEAD LEARNING RATE ======
    parser.add_argument('--head-lr', type=float, default=0.0002,
                        help='learning rate for detection head (creates param group if set)')

    # ====== EVALUATION & EARLY STOPPING ======
    parser.add_argument('--ap-dist-thresh', type=float, default=8.0,
                        help='distance threshold in pixels for AP matching')
    parser.add_argument('--eval-nms', type=str, default='radius',
                        help='NMS type in evaluation: radius|soft')
    parser.add_argument('--eval-nms-radius', type=float, default=3.0,
                        help='radius (pixels) for radius NMS')
    parser.add_argument('--eval-soft-nms-sigma', type=float, default=0.5,
                        help='sigma for Soft-NMS')
    parser.add_argument('--det-patience', type=int, default=10,
                        help='patience for detection AP early stopping')

    # ====== DATA AUGMENTATION (DETECTION) ======
    parser.add_argument('--aug-scale-min', type=float, default=1.0,
                        help='minimum scale factor for random resize')
    parser.add_argument('--aug-scale-max', type=float, default=1.0,
                        help='maximum scale factor for random resize')
    parser.add_argument('--aug-flip', action='store_true',
                        help='enable random horizontal flip augmentation')
    parser.add_argument('--aug-crop-size', type=int, default=0,
                        help='random crop size (0 to disable)')

    # ====== THERMAL PREPROCESSING (PHASE B) ======
    parser.add_argument('--thermal-clahe', action='store_true', default=True,
                        help='enable CLAHE contrast enhancement on thermal images')
    parser.add_argument('--thermal-clahe-clip', type=float, default=2.0,
                        help='CLAHE clip limit for contrast enhancement')

    # ====== COUNTING (DM-COUNT) ======
    parser.add_argument('--wot', type=float, default=0.1,
                        help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01,
                        help='weight on TV loss')
    parser.add_argument('--wrd', type=float, default=0.1,
                        help='weight of regional density loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0,
                        help='whether to normalize coordinates in OT')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # if output_stride is provided, use it as downsample ratio
    if getattr(args, 'output_stride', None) is not None:
        args.downsample_ratio = args.output_stride
    
    # Set CuDNN for reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    # when using torchrun/torch.distributed, LOCAL_RANK is set; otherwise fall back to args.device
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
