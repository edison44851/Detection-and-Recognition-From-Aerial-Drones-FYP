import os
import sys
sys.path.insert(0, os.path.abspath('Fine-tune'))
from utils.dm_regression_trainer import RegTrainer
from types import SimpleNamespace
from typing import Any, cast
import torch
from torch.utils.data import Subset, DataLoader, Dataset

# minimal args similar to train.py
args = SimpleNamespace(
    data_dir='./.data/DroneRGBT_counting',
    save_dir='./tmp_check',
    lr=1e-5,
    resume='.weights/drone_rgbt_best_494_781.pth',
    device='0',
    crop_size=224,
    task='detection',
    freeze_backbone=True,
    freeze_counter=True,
    unfreeze_epoch=-1,
    det_weight=1.0,
    local_rank=0,
    weight_decay=1e-4,
    max_model_num=1,
    max_epoch=1,
    val_epoch=1,
    val_start=0,
    save_all_best=False,
    batch_size=1,
    num_workers=2,
    downsample_ratio=8,
    wot=0.1,
    wtv=0.01,
    reg=10.0,
    num_of_iter_in_ot=100,
    norm_cood=0,
    wrd=0.1
)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

trainer = RegTrainer(args)
trainer.setup()

# replace dataloaders with small subsets (first 2 samples) to make a quick dry-run
try:
    if isinstance(trainer.dataloader, DataLoader):
        train_ds = cast(Dataset[Any], trainer.dataloader.dataset)
        n_train = len(cast(Any, train_ds))
        trainer.dataloader = DataLoader(
            Subset(train_ds, list(range(min(2, n_train)))),
            batch_size=1, shuffle=False, num_workers=0
        )
    if isinstance(trainer.val_dataloader, DataLoader):
        val_ds = cast(Dataset[Any], trainer.val_dataloader.dataset)
        n_val = len(cast(Any, val_ds))
        trainer.val_dataloader = DataLoader(
            Subset(val_ds, list(range(min(2, n_val)))),
            batch_size=1, shuffle=False, num_workers=0
        )
    if isinstance(trainer.test_dataloader, DataLoader):
        test_ds = cast(Dataset[Any], trainer.test_dataloader.dataset)
        n_test = len(cast(Any, test_ds))
        trainer.test_dataloader = DataLoader(
            Subset(test_ds, list(range(min(2, n_test)))),
            batch_size=1, shuffle=False, num_workers=0
        )
except Exception as e:
    print('Could not create subset dataloaders:', e)

# run one training epoch (single quick loop)
trainer.epoch = 0
trainer.train_epoch()
# run validation/test to print new diagnostics
trainer.val_epoch()
trainer.test_epoch()

print('Quick dry-run completed')
