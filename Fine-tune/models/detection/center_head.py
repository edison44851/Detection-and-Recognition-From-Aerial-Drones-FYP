import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterHead(nn.Module):
    """Simple center-based detection head.

    Expects backbone features of shape [B, C, H, W] (C=768 by default).
    Outputs:
      - heatmap: [B,1,H,W] (sigmoid)
      - size: [B,2,H,W] (w,h)
      - offset: [B,2,H,W]
    """

    def __init__(self, in_channels=768, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, kernel_size=1)
        )

        # size predicts width and height
        self.size_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 2, kernel_size=1)
        )

        # offset predicts fractional center offsets
        self.offset_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 2, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        x = self.conv1(feats)
        x = self.bn1(x)
        x = self.relu(x)

        heat = self.heatmap_head(x)
        size = F.relu(self.size_head(x))
        offset = self.offset_head(x)

        heat = torch.sigmoid(heat)
        return heat, size, offset


if __name__ == '__main__':
    # smoke test
    net = CenterHead()
    feats = torch.randn(2, 768, 32, 32)
    h, s, o = net(feats)
    print('heat', h.shape, 'size', s.shape, 'offset', o.shape)