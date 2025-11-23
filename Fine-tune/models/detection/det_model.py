import torch
import torch.nn as nn
from models.counting.swin_unet import Swin_BM_RGBT
from models.detection.center_head import CenterHead


class DetectionModel(nn.Module):
    """Wrapper that composes the existing Swin_BM_RGBT backbone with CenterHead."""

    def __init__(self, backbone_pretrained=True, head_in_channels=768):
        super().__init__()
        self.backbone = Swin_BM_RGBT(pre_train=backbone_pretrained)
        self.head = CenterHead(in_channels=head_in_channels)

    def forward(self, rgb, t, return_feats=False):
        # backbone can return density or (density, feats) when return_feats True
        # we prefer to call get_backbone_features to get features from a single modality (rgb)
        feats = self.backbone.get_backbone_features(rgb)
        heat, size, offset = self.head(feats)
        # also compute density via backbone forward (preserve original behavior)
        density = self.backbone(rgb, t)
        if return_feats:
            return density, (heat, size, offset), feats
        return density, (heat, size, offset)


if __name__ == '__main__':
    import torch
    m = DetectionModel(backbone_pretrained=False)
    rgb = torch.randn(1, 3, 64, 64)
    t = torch.randn(1, 3, 64, 64)
    d, (h, s, o) = m(rgb, t)
    print(d.shape, h.shape, s.shape, o.shape)
