import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn
import logging

class OT_Loss(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        # allow rectangular crop sizes: c_size can be int (square) or (c_h, c_w)
        if isinstance(c_size, (list, tuple)):
            c_h, c_w = c_size
        else:
            c_h = c_w = c_size
        assert c_h % stride == 0 and c_w % stride == 0

        self.c_h = int(c_h)
        self.c_w = int(c_w)
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # coordinate is same to image space, set to constant since crop size is same
        self.cood_x = torch.arange(0, self.c_w, step=stride, dtype=torch.float32, device=device) + stride / 2
        self.cood_y = torch.arange(0, self.c_h, step=stride, dtype=torch.float32, device=device) + stride / 2
        self.cood_x = self.cood_x.unsqueeze(0)  # [1, W]
        self.cood_y = self.cood_y.unsqueeze(0)  # [1, H]
        if self.norm_cood:
            self.cood_x = self.cood_x / float(self.c_w) * 2 - 1
            self.cood_y = self.cood_y / float(self.c_h) * 2 - 1
        self.output_h = self.cood_y.size(1)
        self.output_w = self.cood_x.size(1)
        # keep a canonical c_size representation and output_size for compatibility
        # when downstream code expects `self.c_size` and `self.output_size` (used in set_grid)
        if self.c_h == self.c_w:
            self.c_size = float(self.c_w)
        else:
            # store as tensor [W, H] to allow broadcasting with point coords [N,2]
            self.c_size = torch.tensor([float(self.c_w), float(self.c_h)], device=self.device)
        # output_size kept for legacy code paths that expect a single integer (square case)
        self.output_size = self.output_w
        self.density_size = self.output_w


    def forward(self, normed_density, unnormed_density, points, image_size=None):
        """
        normed_density: (B,1,Hout,Wout)
        unnormed_density: same shape
        points: list of length B with tensors of shape [K,2] in image pixel coordinates (x,y)
        image_size: optional (H_img, W_img) to scale points into OT grid coordinate frame
        """
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        # Allow dynamic output sizes: caller should call `set_grid(...)` once to configure grid.
        h = normed_density.size(2)
        w = normed_density.size(3)
        if (self.output_h != h) or (self.output_w != w):
            raise AssertionError(f'OT_Loss output_size {(self.output_h, self.output_w)} does not match density size {(h, w)}')
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0 # wasserstain distance
        for idx, im_points in enumerate(points):
            if im_points is None:
                continue
            if len(im_points) > 0:
                # im_points are provided in image pixel coords (x,y). If image_size is provided and
                # differs from self.c_size, scale points accordingly to the OT grid coordinate frame.
                if image_size is not None:
                    img_h, img_w = image_size
                    # compute scale factors from image pixels to OT grid pixels (width and height separately)
                    scale_x = float(self.c_w) / float(img_w)
                    scale_y = float(self.c_h) / float(img_h)
                    # im_points may be torch tensor on cpu; convert and scale on the same device
                    im_points = im_points.to(self.cood_x.device).clone()
                    im_points[:, 0] = im_points[:, 0] * scale_x
                    im_points[:, 1] = im_points[:, 1] * scale_y
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1 # map to [-1, 1]
                x = im_points[:, 0].unsqueeze(1)  # [#gt, 1]
                y = im_points[:, 1].unsqueeze(1)
                # compute distances separately for x and y using rectangular grid
                x_dis = -2 * torch.matmul(x, self.cood_x) + x * x + self.cood_x * self.cood_x  # [#gt, W]
                y_dis = -2 * torch.matmul(y, self.cood_y) + y * y + self.cood_y * self.cood_y  # [#gt, H]
                # reshape to combine H and W -> broadcast add
                y_dis = y_dis.unsqueeze(2)  # [#gt, H, 1]
                x_dis = x_dis.unsqueeze(1)  # [#gt, 1, W]
                dis = y_dis + x_dis  # [#gt, H, W]
                dis = dis.view((dis.size(0), -1))  # size of [#gt, H * W]

                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
                # use sinkhorn to solve OT, compute optimal beta.
                try:
                    P, log = sinkhorn(target_prob, source_prob, dis, self.reg, maxIter=self.num_of_iter_in_ot, log=True)
                except AssertionError as e:
                    logging.warning('sinkhorn assertion: num_points=%s, source_prob_shape=%s, dis_shape=%s', len(im_points), tuple(source_prob.shape), tuple(dis.shape))
                    raise
                beta = log['beta'] # size is the same as source_prob: [#cood * #cood]
                # beta corresponds to flattened [H * W] grid; reshape using (output_h, output_w)
                ot_obj_values = ot_obj_values + torch.sum(normed_density[idx] * beta.view([1, self.output_h, self.output_w]))
                # compute the gradient of OT loss to predicted density (unnormed_density).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta # size of [#cood * #cood]
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8) # size of 1

                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, self.output_h, self.output_w])
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
                loss = loss + torch.sum(unnormed_density[idx] * im_grad)
                wd = wd + torch.sum(dis * P).item()

        return loss, wd, ot_obj_values

    def set_grid(self, c_size, stride):
        """
        Reinitialize internal coordinate grid for a different crop size / stride.
        c_size: image length in pixels (assumes square), stride: sampling stride
        """
        assert c_size % stride == 0
        self.c_size = c_size
        self.cood = torch.arange(0, c_size, step=stride, dtype=torch.float32, device=self.device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood = self.cood.unsqueeze(0)
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1
        self.output_size = self.cood.size(1)