from utils.dm_regression_trainer import RegTrainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default=r'',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='directory to save models.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='default 224')
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
    parser.add_argument('--det-weight', type=float, default=1.0,
                        help='weight for detection loss when multi-task')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training (set by torchrun)')
    parser.add_argument('--det-patience', type=int, default=10,
                        help='patience (in validation epochs) for detection AP early stopping')

    # default
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=500,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')
    parser.add_argument('--save-all-best', type=bool, default=False,
                        help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')
    parser.add_argument('--output-stride', type=int, default=None,
                        help='(alias) output stride / downsample ratio; overrides --downsample-ratio when provided')
    parser.add_argument('--det-pos-weight', type=float, default=1.0,
                        help='positive pixel weight for detection heatmap BCE (>=1.0 increases weight on positives)')
    parser.add_argument('--head-lr', type=float, default=None,
                        help='learning rate for detection head/adaptor (if set, creates optimizer param group)')
    parser.add_argument('--ap-dist-thresh', type=float, default=8.0,
                        help='distance threshold in pixels for AP matching (compute_ap)')
    parser.add_argument('--use-det-adaptor', action='store_true',
                        help='create a small 1x1 conv adaptor (det_adaptor) before det_head')
    parser.add_argument('--use-bce-logits', action='store_true',
                        help='use BCEWithLogitsLoss and have head output raw logits (no sigmoid)')
    parser.add_argument('--det-use-gn', action='store_true',
                        help='use GroupNorm in detection head and adaptor instead of BatchNorm')
    # Detection tuning flags
    parser.add_argument('--det-sigma', type=float, default=None,
                        help='gaussian sigma used to generate heatmaps in DetectionDataset (if provided)')
    parser.add_argument('--use-focal-heatmap', action='store_true',
                        help='use focal loss on heatmap (requires logits); alpha/gamma configurable')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='focal loss alpha (class balance)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='focal loss gamma (focus on hard examples)')
    parser.add_argument('--det-neg-topk-ratio', type=float, default=None,
                        help='optional ratio of hardest negative pixels to include in heatmap loss (0-1); if None, include all negatives')
    parser.add_argument('--use-iou-size', action='store_true',
                        help='add IoU-based loss for size regression at positive locations')
    parser.add_argument('--iou-weight', type=float, default=0.5,
                        help='weight for IoU size loss relative to L1 size loss')
    # Eval-time NMS options
    parser.add_argument('--eval-nms', type=str, default=None,
                        help='optional NMS in evaluation/visualization: radius|soft')
    parser.add_argument('--eval-nms-radius', type=float, default=4.0,
                        help='radius (pixels) for radius NMS when eval-nms=radius')
    parser.add_argument('--eval-soft-nms-sigma', type=float, default=0.5,
                        help='sigma for Soft-NMS when eval-nms=soft')
    
    # CenterNet-style detection head options (Option B)
    parser.add_argument('--head-conv', type=int, default=256,
                        help='detection head conv channels (CenterNet default: 256)')
    parser.add_argument('--use-deconv', action='store_true',
                        help='use ConvTranspose2d for upsampling (CenterNet style); else use bilinear+conv')
    parser.add_argument('--nms-kernel', type=int, default=3,
                        help='NMS kernel size for heatmap decode (default 3, CenterNet uses 3)')
                        
    # For DM-Count
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')
    
    # For RD Loss
    parser.add_argument('--wrd', type=float, default=0.1, help='weight of regional density loss')   

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # if output_stride is provided, use it as downsample ratio
    if getattr(args, 'output_stride', None) is not None:
        args.downsample_ratio = args.output_stride
    torch.backends.cudnn.benchmark = True
    # when using torchrun/torch.distributed, LOCAL_RANK is set; otherwise fall back to args.device
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # keep existing behavior for single-node

    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
