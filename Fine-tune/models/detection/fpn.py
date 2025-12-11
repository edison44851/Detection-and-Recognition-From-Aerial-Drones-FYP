import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPNNeck(nn.Module):
    """Lightweight FPN-style neck built from a single high-res feature map.

    We synthesize a pyramid (P4/P8/P16) by pooling the input feature and use
    lateral 1x1 projections to a common channel size, then merge everything
    back to stride-4 for the detection head. This keeps compute light while
    providing multi-scale context.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        self.lateral_p4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.lateral_p8 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.lateral_p16 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is stride-4 fused feature (C=in_channels)
        p4 = self.lateral_p4(x)
        p8 = self.lateral_p8(F.avg_pool2d(x, kernel_size=2, stride=2))
        p16 = self.lateral_p16(F.avg_pool2d(x, kernel_size=4, stride=4))

        # Upsample to stride-4 and fuse
        p8_up = F.interpolate(p8, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p16_up = F.interpolate(p16, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        fused = p4 + p8_up + p16_up
        fused = self.smooth(fused)
        return fused
