import torch
import torch.nn as nn
from models.detection.center_head import CenterHead


class DetectionHeadWrapper(nn.Module):
    """Lightweight wrapper exposing a detection head that accepts UNet/backbone features.

    This file no longer instantiates the backbone. The intended usage is to attach the
    head to an existing `Swin_BM_RGBT` instance (which now exposes a `det_adaptor`
    and can accept an attached `det_head`).
    """

    def __init__(self, in_channels: int = 768, hidden: int = 256, 
                 head_conv: int = 256, use_deconv: bool = True):
        """
        Args:
            in_channels: input feature channels (from backbone/adaptor)
            hidden: deprecated, kept for backward compatibility (not used)
            head_conv: channels for detection head conv layers (CenterNet-style)
            use_deconv: whether to use ConvTranspose2d for upsampling
        """
        super().__init__()
        # Note: 'hidden' parameter is ignored; upsample module always outputs 256 channels
        self.head = CenterHead(in_channels=in_channels, head_conv=head_conv, 
                                use_deconv=use_deconv)

    def forward(self, feats: torch.Tensor):
        """Forward the head given fused features from the model's UNet/backbone.

        Args:
            feats: tensor [B, C, H, W]
        Returns:
            heat, size, offset
        """
        return self.head(feats)


if __name__ == '__main__':
    # smoke test for the head alone
    net = DetectionHeadWrapper()
    feats = torch.randn(2, 768, 32, 32)
    h, s, o = net(feats)
    print('heat', h.shape, 'size', s.shape, 'offset', o.shape)
