import torch
import torch.nn as nn


def Label2R(labels):
    # Label: [region_number, 1]
    # Matrix R_ij: [region_number, region_number]
    return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)


def Feature2F(features):
    # Feature: [region_number, feat_dim]
    # Matrix F_ij: [region_number, region_number]
    return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)


class RDLoss(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, feature, label):
        # Feature: [region_number, feat_dim]
        # Label: [region_number, 1]

        R = Label2R(label)
        F = Feature2F(feature) * self.sigma
        F_max, _ = torch.max(F, dim=1, keepdim=True)
        F -= F_max.detach()
        scaled_F = nn.functional.softmax(F, dim=1)

        n = F.shape[0]  # N = region_number

        # Delete diagonal elements
        F = F.masked_select((1 - torch.eye(n).to(F.device)).bool()).view(n, n - 1)
        scaled_F = scaled_F.masked_select((1 - torch.eye(n).to(F.device)).bool()).view(n, n - 1)
        R = R.masked_select((1 - torch.eye(n).to(F.device)).bool()).view(n, n - 1)

        # Determination for negatives in one step
        pos_F = F.transpose(0, 1)  # [N-1, N]
        pos_R = R.transpose(0, 1)  # [N-1, N] -> [N-1, N, 1]
        # Broadcast to create a mask: [N-1, N, N-1]
        neg_mask = (R.unsqueeze(0).repeat(n - 1, 1, 1) >= pos_R.view(n - 1, n, 1)).float()

        # Calculate positive log probabilities
        # scaled_F: [N, N-1] -> [N-1, N, 1]
        pos_log_probs = pos_F - torch.log(
            (neg_mask * scaled_F.unsqueeze(0).repeat(n - 1, 1, 1)).sum(dim=-1) + 1e-7)

        loss = - (pos_log_probs / (n * (n - 1))).sum()
        return loss


class CL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rd = RDLoss()

    def forward(self, feature, points, image_size=None):
        # feature: [B, C, H, W]
        assert feature.size(0) == 1
        all_points = torch.cat(points, dim=0)
        # Determine subregion grid based on pooled feature spatial size
        feat_pooled = self.mp(feature)
        _, _, Hf, Wf = feat_pooled.shape
        # If image_size provided, map points to pooled grid using image dimensions
        if image_size is not None:
            img_h, img_w = image_size
            label = self.subregion_dynamic(all_points, Hf, Wf, img_h, img_w)
        else:
            # fallback to legacy behavior: assume crop_size=224 and ds_ratio=16
            label = self.subregion(all_points)
        # feature: [B, C, H, W] -> [B, N, C]
        # label: [Hf, Wf] -> [N, 1]
        feature = feat_pooled.flatten(2).transpose(1, 2)
        label = label.flatten().unsqueeze(1)
        loss = self.rd(feature[0, :], label)
        return loss

    def subregion(self, all_points, crop_size=224, ds_ratio=16):
        H = crop_size // ds_ratio
        # Mapping coordinates to subregions
        subregion_indices = (all_points // ds_ratio).long()
        # Count the number of points in each subregion
        counts = torch.zeros((H, H), dtype=torch.int32, device=all_points.device)
        for i in range(all_points.size(0)):
            x, y = subregion_indices[i]
            x, y = min(x, H - 1), min(y, H - 1)
            counts[y, x] += 1
        return counts

    def subregion_dynamic(self, all_points, Hf, Wf, img_h, img_w):
        """
        Map point coordinates (in pixel space) to a pooled feature grid of size Hf x Wf.
        all_points: tensor [K, 2] with coordinates [x, y]
        img_h, img_w: image height and width in pixels
        """
        counts = torch.zeros((Hf, Wf), dtype=torch.int32, device=all_points.device)
        if all_points.numel() == 0:
            return counts
        # normalize and map
        xs = all_points[:, 0]
        ys = all_points[:, 1]
        # scale to [0, Wf), [0, Hf)
        x_idx = torch.clamp((xs / float(img_w) * float(Wf)).long(), 0, Wf - 1)
        y_idx = torch.clamp((ys / float(img_h) * float(Hf)).long(), 0, Hf - 1)
        for i in range(all_points.size(0)):
            counts[y_idx[i], x_idx[i]] += 1
        return counts


if __name__ == "__main__":
    points = [torch.abs(torch.randn(10, 2)).to("cuda")]
    feature = torch.randn(1, 768, 28, 28).to("cuda")
    cl = CL1().train()
    loss = cl(feature, points)
    print(loss)
