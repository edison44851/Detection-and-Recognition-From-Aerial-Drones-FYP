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
                        help='task to run: counting | detection | multi')
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
    parser.add_argument('--save-by', type=str, default='count', choices=['count','det','multi','combined'],
                        help='Which metric to use for saving best model')
    parser.add_argument('--combined-alpha', type=float, default=1.0,
                        help='alpha weight for AP in combined score (AP normalized)')
    parser.add_argument('--combined-beta', type=float, default=1.0,
                        help='beta weight for GAME0 in combined score (GAME0 normalized)')

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
    torch.backends.cudnn.benchmark = True
    # when using torchrun/torch.distributed, LOCAL_RANK is set; otherwise fall back to args.device
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # keep existing behavior for single-node

    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
