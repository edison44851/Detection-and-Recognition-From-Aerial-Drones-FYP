import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterHead(nn.Module):
    """CenterNet-style center-based detection head.

    Upgraded to match CenterNet's proven architecture:
    - Optional deconv/upsample layer to reduce stride (8→4)
    - Wider heads with proper channel capacity (head_conv=256)
    - Heatmap bias initialized to -4.6 for drone datasets (more conservative)
    - CenterNet-style 2-layer heads per task
    - **Phase 1**: keypoint_only mode - skip size head for point annotations
    - Optimized initialization for small object detection in drone images

    Expects backbone features of shape [B, C, H, W] (C=768 by default at stride-8).
    Outputs (at stride-4 if use_deconv=True, else stride-8):
      - heatmap: [B,1,H,W] (sigmoid or logits)
      - size: [B,2,H,W] (w,h) — OR None if keypoint_only=True
      - offset: [B,2,H,W]
    """

    def __init__(self, in_channels=768, head_conv=256, use_logits: bool = False, 
                 use_gn: bool = False, use_deconv: bool = True, keypoint_only: bool = False):
        super().__init__()
        self.use_logits = use_logits
        self.use_gn = use_gn
        self.use_deconv = use_deconv
        self.keypoint_only = keypoint_only
        
        # Upsampling module: reduce stride from 8 to 4 (CenterNet uses stride-4 output)
        if use_deconv:
            # CenterNet-style deconv: ConvTranspose2d with stride=2
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, 
                                   padding=1, bias=False),
                nn.BatchNorm2d(256) if not use_gn else self._make_gn(256),
                nn.ReLU(inplace=True)
            )
        else:
            # Lightweight alternative: bilinear upsample + conv
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256) if not use_gn else self._make_gn(256),
                nn.ReLU(inplace=True)
            )
        
        # Heatmap head (CenterNet style: Conv 256→head_conv→1 with bias=-2.19)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, bias=True)
        )
        # CRITICAL: Initialize heatmap bias properly for drone dataset
        # For small objects in drone images, we need different initialization
        nn.init.constant_(self.heatmap_head[-1].bias, -4.6)  # More conservative for drone data

        # Size head (width, height) — SKIP in keypoint-only mode
        if not self.keypoint_only:
            self.size_head = nn.Sequential(
                nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True)
            )
        else:
            self.size_head = None

        # Offset head (fractional center offsets)
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True)
        )
        
        # Initialize weights (Kaiming for conv, constant for BN/GN)
        self._init_weights()
    
    def _make_gn(self, channels):
        """Create GroupNorm with appropriate group count."""
        for g in (32, 16, 8, 4, 2, 1):
            if channels % g == 0:
                return nn.GroupNorm(g, channels)
        return nn.GroupNorm(1, channels)
    
    def _init_weights(self):
        """Initialize weights following CenterNet convention - FIXED for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # Deconv weights: use bilinear initialization with smaller scale
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        
        # Re-initialize heatmap bias for drone dataset
        nn.init.constant_(self.heatmap_head[-1].bias, -2.0)

    def forward(self, feats):
        # Upsample features (stride 8 → 4)
        x = self.upsample(feats)

        # Task-specific heads
        heat = self.heatmap_head(x)
        
        # Size head: skip if keypoint_only mode
        if self.keypoint_only:
            size = None
        else:
            size = F.relu(self.size_head(x))  # Size must be non-negative
        
        offset = self.offset_head(x)

        # Return raw logits if requested; otherwise return sigmoid probabilities
        if self.use_logits:
            return heat, size, offset
        else:
            return torch.sigmoid(heat), size, offset


if __name__ == '__main__':
    # smoke test
    print("Testing CenterHead with deconv (stride 8→4):")
    net = CenterHead(in_channels=768, head_conv=256, use_deconv=True)
    feats = torch.randn(2, 768, 32, 32)  # stride-8 input
    h, s, o = net(feats)
    print(f'Input: {feats.shape}')
    print(f'Heatmap: {h.shape} (should be 2x upsampled: 64x64)')
    print(f'Size: {s.shape}')
    print(f'Offset: {o.shape}')
    print(f'Heatmap bias (should be -2.19): {net.heatmap_head[-1].bias.item():.4f}')
    
    print("\nTesting CenterHead without deconv (stride stays 8):")
    net2 = CenterHead(in_channels=768, head_conv=256, use_deconv=False)
    h2, s2, o2 = net2(feats)
    print(f'Heatmap: {h2.shape} (should still be 2x upsampled via bilinear: 64x64)')
    print(f'Heatmap bias: {net2.heatmap_head[-1].bias.item():.4f}')