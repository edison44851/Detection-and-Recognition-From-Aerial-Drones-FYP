import torch
import torch.nn as nn
from models.detection.center_head import CenterHead
from models.detection.fpn import SimpleFPNNeck


class DetectionHeadWrapper(nn.Module):
    """Lightweight wrapper exposing a detection head that accepts UNet/backbone features.

    This file no longer instantiates the backbone. The intended usage is to attach the
    head to an existing `Swin_BM_RGBT` instance (which now exposes a `det_adaptor`
    and can accept an attached `det_head`).
    """

    def __init__(self, in_channels: int = 768, hidden: int = 256, 
                 head_conv: int = 256, use_deconv: bool = True, keypoint_only: bool = False,
                 use_fpn: bool = False):
        """
        Args:
            in_channels: input feature channels (from backbone/adaptor)
            hidden: deprecated, kept for backward compatibility (not used)
            head_conv: channels for detection head conv layers (CenterNet-style)
            use_deconv: whether to use ConvTranspose2d for upsampling
            keypoint_only: if True, skip size_head (for point-only annotations)
        """
        super().__init__()
        # Note: 'hidden' parameter is ignored; upsample module always outputs 256 channels
        self.use_fpn = use_fpn
        # If FPN is enabled, we first project to head_conv channels, then feed the head
        head_in_channels = head_conv if use_fpn else in_channels
        self.fpn = SimpleFPNNeck(in_channels=in_channels, out_channels=head_conv) if use_fpn else None
        self.head = CenterHead(in_channels=head_in_channels, head_conv=head_conv, 
                    use_deconv=use_deconv, keypoint_only=keypoint_only)

    def forward(self, feats: torch.Tensor):
        """Forward the head given fused features from the model's UNet/backbone.

        Args:
            feats: tensor [B, C, H, W]
        Returns:
            heat, size, offset
        """
        if self.use_fpn:
            feats = self.fpn(feats)
        return self.head(feats)


if __name__ == '__main__':
    # smoke test for the head alone
    net = DetectionHeadWrapper()
    feats = torch.randn(2, 768, 32, 32)
    h, s, o = net(feats)
    print('heat', h.shape, 'size', s.shape, 'offset', o.shape)
