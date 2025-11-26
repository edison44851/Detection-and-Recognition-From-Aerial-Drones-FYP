import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def gaussian2d(shape, sigma=1):
    """Return a 2D gaussian kernel with given shape (h,w)"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class DetectionDataset(Dataset):
    """Detection dataset that reads `*_RGB.jpg`, `*_T.jpg`, and `*_GT.npy` files.

    Generates heatmap, size and offset targets on the fly. By default the head
    output stride is assumed to be `output_stride` (so heatmap size = image size // output_stride).
    """

    def __init__(self, root, split='train', transform=None, output_stride=4, sigma=2):
        self.root = root
        self.split = split
        self.output_stride = output_stride
        self.sigma = sigma
        self.transform = transform or T.Compose([T.ToTensor()])

        self.dir = os.path.join(root, split)
        files = [f for f in os.listdir(self.dir) if f.endswith('_RGB.jpg')]
        files.sort()
        self.ids = [f[:-8] for f in files]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id0 = self.ids[idx]
        rgb_p = os.path.join(self.dir, id0 + '_RGB.jpg')
        t_p = os.path.join(self.dir, id0 + '_T.jpg')
        gt_p = os.path.join(self.dir, id0 + '_GT.npy')

        rgb = Image.open(rgb_p).convert('RGB')
        t = Image.open(t_p).convert('RGB')

        img = self.transform(rgb)  # tensor [3,H,W]
        timg = self.transform(t)

        H, W = img.shape[1], img.shape[2]
        H_out = H // self.output_stride
        W_out = W // self.output_stride

        heatmap = np.zeros((H_out, W_out), dtype=np.float32)
        size_map = np.zeros((2, H_out, W_out), dtype=np.float32)
        offset_map = np.zeros((2, H_out, W_out), dtype=np.float32)

        points = np.load(gt_p)  # expected shape (N,2) as (x,y) or (col,row)
        if points is None:
            points = np.zeros((0, 2))

        # Precompute coordinate grids in output space for efficient gaussian placement
        xs = np.arange(W_out, dtype=np.float32)
        ys = np.arange(H_out, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        for p in points:
            # assume (x, y) in pixel coordinates where x is column, y is row
            x, y = float(p[0]), float(p[1])
            fx = x / self.output_stride
            fy = y / self.output_stride
            # fractional offset
            offx = fx - np.floor(fx)
            offy = fy - np.floor(fy)
            # compute gaussian in output-space centered at (fx, fy)
            dist2 = (X - fx) ** 2 + (Y - fy) ** 2
            g = np.exp(-dist2 / (2 * (self.sigma ** 2)))
            # accumulate and clip later
            heatmap += g.astype(np.float32)
            ix = int(np.floor(fx))
            iy = int(np.floor(fy))
            if 0 <= ix < W_out and 0 <= iy < H_out:
                size_map[0, iy, ix] = max(size_map[0, iy, ix], 16.0 / self.output_stride)
                size_map[1, iy, ix] = max(size_map[1, iy, ix], 16.0 / self.output_stride)
                offset_map[0, iy, ix] = offx
                offset_map[1, iy, ix] = offy

        # Ensure heatmap values are in [0,1]
        heatmap = np.clip(heatmap, 0.0, 1.0)

        sample = {
            'rgb': img,
            't': timg,
            'heatmap': torch.from_numpy(heatmap).unsqueeze(0),
            'size': torch.from_numpy(size_map),
            'offset': torch.from_numpy(offset_map),
            'id': id0,
            'points': torch.from_numpy(points).float()
        }
        return sample


if __name__ == '__main__':
    # simple smoke test
    ds = DetectionDataset(root='.data/DroneRGBT_counting', split='train')
    s = ds[0]
    print('rgb', s['rgb'].shape, 'heatmap', s['heatmap'].shape, 'size', s['size'].shape)