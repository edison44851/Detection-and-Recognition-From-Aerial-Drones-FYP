import os
import sys
import numpy as np
from PIL import Image
import torch
# make Fine-tune package importable during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.detection.center_head import CenterHead
from datasets.dm_detection import DetectionDataset


def make_dummy_dataset(root):
    # create structure train/val/test with one sample
    for split in ['train', 'val', 'test']:
        d = os.path.join(root, split)
        os.makedirs(d, exist_ok=True)
        # create sample files: id_RGB.jpg, id_T.jpg, id_GT.npy
        id0 = '0001'
        rgb = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype('uint8'))
        t = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype('uint8'))
        rgb.save(os.path.join(d, id0 + '_RGB.jpg'))
        t.save(os.path.join(d, id0 + '_T.jpg'))
        # GT: one point near center
        pts = np.array([[64.0, 64.0]], dtype=np.float32)
        np.save(os.path.join(d, id0 + '_GT.npy'), pts)


def test_center_head_and_dataset_shapes(tmp_path):
    data_root = tmp_path / '.data' / 'DroneRGBT_counting'
    data_root = str(data_root)
    make_dummy_dataset(data_root)

    ds = DetectionDataset(data_root, split='train', output_stride=4)
    sample = ds[0]
    # check dataset shapes
    assert 'rgb' in sample and 'heatmap' in sample and 'size' in sample and 'offset' in sample
    rgb = sample['rgb']
    heatmap = sample['heatmap']
    size = sample['size']
    offset = sample['offset']
    assert rgb.shape[0] == 3
    # output dims should match heatmap spatial dims
    _, H_out, W_out = heatmap.shape
    assert size.shape[1] == H_out and size.shape[2] == W_out
    assert offset.shape[1] == H_out and offset.shape[2] == W_out

    # test CenterHead forward compatibility
    head = CenterHead(in_channels=768)
    feats = torch.randn(1, 768, H_out, W_out)
    heat, sz, off = head(feats)
    assert heat.shape == (1, 1, H_out, W_out)
    assert sz.shape == (1, 2, H_out, W_out)
    assert off.shape == (1, 2, H_out, W_out)

    # minimal integration: ensure heatmap channel matches head output
    # (we won't attach head to backbone here)
    assert heat.max() <= 1.0 and heat.min() >= 0.0
