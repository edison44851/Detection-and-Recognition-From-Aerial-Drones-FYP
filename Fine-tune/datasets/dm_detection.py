import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


def gaussian2d(shape, sigma=1):
    """Return a 2D gaussian kernel with given shape (h,w)"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def apply_random_resize(rgb, t, points, scale_range=(0.5, 2.0)):
    """Randomly resize image and adjust point coordinates.
    
    Args:
        rgb: PIL Image (RGB)
        t: PIL Image (thermal)
        points: numpy array [N, 2] of (x, y) point coordinates
        scale_range: tuple (min_scale, max_scale)
    
    Returns:
        rgb_resized, t_resized, points_resized, scale_factor
    """
    if random.random() > 0.5:
        # Skip augmentation with 50% probability
        return rgb, t, points, 1.0
    
    scale = random.uniform(scale_range[0], scale_range[1])
    w, h = rgb.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    rgb_resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    t_resized = t.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Adjust point coordinates
    if len(points) > 0:
        points_resized = points * scale
    else:
        points_resized = points
    
    return rgb_resized, t_resized, points_resized, scale


def apply_random_flip(rgb, t, points):
    """Randomly flip image horizontally and adjust point coordinates.
    
    Args:
        rgb: PIL Image (RGB)
        t: PIL Image (thermal)
        points: numpy array [N, 2] of (x, y) point coordinates
    
    Returns:
        rgb_flipped, t_flipped, points_flipped
    """
    if random.random() > 0.5:
        # Skip augmentation with 50% probability
        return rgb, t, points
    
    w = rgb.size[0]
    rgb_flipped = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    t_flipped = t.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    # Mirror x coordinates
    if len(points) > 0:
        points_flipped = points.copy()
        points_flipped[:, 0] = w - points_flipped[:, 0]
    else:
        points_flipped = points
    
    return rgb_flipped, t_flipped, points_flipped


def apply_random_crop(rgb, t, points, crop_size=224):
    """Randomly crop image and adjust point coordinates.
    
    Args:
        rgb: PIL Image (RGB)
        t: PIL Image (thermal)
        points: numpy array [N, 2] of (x, y) point coordinates
        crop_size: size of square crop (if 0, skip)
    
    Returns:
        rgb_cropped, t_cropped, points_cropped
    """
    if crop_size <= 0:
        return rgb, t, points
    
    w, h = rgb.size
    if w < crop_size or h < crop_size:
        # Image too small to crop, skip
        return rgb, t, points
    
    # Random crop position
    x_min = random.randint(0, w - crop_size)
    y_min = random.randint(0, h - crop_size)
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    
    rgb_cropped = rgb.crop((x_min, y_min, x_max, y_max))
    t_cropped = t.crop((x_min, y_min, x_max, y_max))
    
    # Adjust points: translate and filter out-of-bounds
    if len(points) > 0:
        points_cropped = points.copy()
        points_cropped[:, 0] -= x_min
        points_cropped[:, 1] -= y_min
        
        # Keep only points within crop
        mask = (points_cropped[:, 0] >= 0) & (points_cropped[:, 0] < crop_size) & \
               (points_cropped[:, 1] >= 0) & (points_cropped[:, 1] < crop_size)
        points_cropped = points_cropped[mask]
    else:
        points_cropped = points
    
    return rgb_cropped, t_cropped, points_cropped


def apply_clahe_thermal(thermal_pil, clip_limit=2.0, tile_size=8):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to thermal image.
    
    CLAHE improves contrast in thermal images with low dynamic range by locally
    equalizing histograms. Standard preprocessing for thermal imagery.
    
    Args:
        thermal_pil: PIL Image (thermal, already converted to RGB 3-channel)
        clip_limit: Contrast amplification limit (higher = more contrast, 2.0 is standard)
        tile_size: Size of tile grid for local processing (8 is standard)
    
    Returns:
        PIL Image with CLAHE applied, ready for normalization transform
    """
    if not HAS_CV2:
        # cv2 not available, return original
        return thermal_pil
    assert cv2 is not None
    
    # Convert PIL to numpy array
    thermal_np = np.array(thermal_pil, dtype=np.uint8)  # Ensure uint8 for cv2
    
    if thermal_np.ndim == 3:
        # 3-channel thermal (RGB-formatted)
        # Apply CLAHE to L channel of LAB for better results
        thermal_lab = cv2.cvtColor(thermal_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        thermal_lab[:, :, 0] = clahe.apply(thermal_lab[:, :, 0])
        thermal_np = cv2.cvtColor(thermal_lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale thermal
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        thermal_np = clahe.apply(thermal_np)
    
    # Convert back to PIL
    return Image.fromarray(thermal_np)


class DetectionDataset(Dataset):
    """Detection dataset that reads `*_RGB.jpg`, `*_T.jpg`, and `*_GT.npy` files.

    Generates heatmap, size and offset targets on the fly. By default the head
    output stride is assumed to be `output_stride` (so heatmap size = image size // output_stride).
    
    Supports data augmentation for training:
    - Random resize (scale_range): scales object sizes to handle scale variation
    - Random flip: horizontal flipping for robustness
    - Random crop: crops random regions to simulate localized detection
    """

    def __init__(self, root, split='train', transform=None, output_stride=4, sigma=0.8,
                 rgb_transform=None, t_transform=None, aug_scale=None, aug_flip=False, 
                 aug_crop_size=0, thermal_clahe=True, thermal_clahe_clip=2.0):
        self.root = root
        self.split = split
        self.output_stride = output_stride
        self.sigma = sigma
        self.thermal_clahe = thermal_clahe and HAS_CV2  # Only use if cv2 available
        self.thermal_clahe_clip = thermal_clahe_clip
        self.sigma = sigma
        
        # Augmentation parameters
        self.aug_scale = aug_scale if aug_scale is not None else (1.0, 1.0)  # (min, max)
        self.aug_flip = aug_flip and (split == 'train')  # Only augment training split
        self.aug_crop_size = aug_crop_size if split == 'train' else 0  # Only augment training split
        
        # default to same normalization used by dm_crowd (keep ToTensor if custom provided)
        if rgb_transform is None:
            rgb_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.407, 0.389, 0.396], std=[0.241, 0.246, 0.242])
            ])
        if t_transform is None:
            # Use RGBT-CC computed stats: mean=[0.499, 0.168, 0.431], std=[0.308, 0.168, 0.181]
            # These are based on actual distribution of RGBT-CC thermal images
            # Previously used hardcoded stats: mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191]
            t_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.499, 0.168, 0.431], std=[0.308, 0.168, 0.181])
            ])
        # backward-compatible single transform argument
        if transform is not None:
            self.rgb_transform = transform
            self.t_transform = transform
        else:
            self.rgb_transform = rgb_transform
            self.t_transform = t_transform

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
        
        # Ensure RGB and thermal have the same spatial dimensions
        if rgb.size != t.size:
            # Resize thermal to match RGB dimensions
            t = t.resize(rgb.size, Image.Resampling.BILINEAR)
        
        # Apply CLAHE to thermal image for contrast enhancement (Phase B)
        if self.thermal_clahe:
            t = apply_clahe_thermal(t, clip_limit=self.thermal_clahe_clip)

        points = np.load(gt_p)  # expected shape (N,2) as (x,y) or (col,row)
        if points is None:
            points = np.zeros((0, 2))
        
        # Apply augmentations (only on training split)
        if self.split == 'train':
            # 1. Random resize to handle scale variation
            if self.aug_scale != (1.0, 1.0):
                rgb, t, points, _ = apply_random_resize(rgb, t, points, self.aug_scale)
            
            # 2. Random horizontal flip
            if self.aug_flip:
                rgb, t, points = apply_random_flip(rgb, t, points)
            
            # 3. Random crop
            if self.aug_crop_size > 0:
                rgb, t, points = apply_random_crop(rgb, t, points, self.aug_crop_size)

        # Apply normalization transforms
        img = self.rgb_transform(rgb)
        timg = self.t_transform(t)
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        if not isinstance(timg, torch.Tensor):
            timg = T.ToTensor()(timg)

        H, W = img.shape[1], img.shape[2]
        H_out = H // self.output_stride
        W_out = W // self.output_stride

        heatmap = np.zeros((H_out, W_out), dtype=np.float32)
        size_map = np.zeros((2, H_out, W_out), dtype=np.float32)
        offset_map = np.zeros((2, H_out, W_out), dtype=np.float32)

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
            # assign size/offset to the nearest output grid cell (sub-pixel aware)
            ix = int(np.round(fx))
            iy = int(np.round(fy))
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