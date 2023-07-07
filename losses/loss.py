# coding: utf8
'''
Author: Pengbo
Date: 2022-02-23 16:17:31
LastEditTime: 2023-03-01 21:21:33
Description: loss function

'''
import torch
from torch import nn
from torch.nn import functional as F
import kornia

import pdb
from utils.utils import low_pass, gradient


__all__ = ['L1Loss',
           'L2Loss',
           'MS_SSIMLoss',
           'GradientL1Loss',
           'GradientL2Loss',
           'AdaptiveGradientL2Loss',
           'BCELoss',
           'BCEFocalLoss',
           'FocalLoss',
           'DeCompositionLoss',
           'FusionLoss',
        ]


class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)


class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.pos_weight = 1
        self.reduction = 'mean'

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        loss = - self.pos_weight * target * torch.log(logits) - \
            (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=0.2, num_classes=2):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = 2
        self.reduction = 'mean'

    def forward(self, logits, target):
        alpha = self.alpha
        gamma = self.gamma
        loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
            (1 - alpha) * logits ** gamma * \
            (1 - target) * torch.log(1 - logits)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss, -α(1-yi)**γ *ce_loss(xi,yi)
        :param alpha: loss weight for each class. 
        :param gamma: hyper-parameter to adjust hard sample
        :param num_classes:
        :param size_average:
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            # set alpha to [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        :param preds:   prediction. the size: [B,N,C] or [B,C]
        :param labels:  ground-truth. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax),
                          self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class MS_SSIMLoss(nn.Module):
    def __init__(self, window_size=11, max_val=1.0):
        super(MS_SSIMLoss, self).__init__()
        self.ssim = kornia.losses.SSIMLoss(
            window_size=window_size, max_val=max_val, reduction='mean')
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        return self.ssim(img1, img2) + self.mse(img1, img2)


class GradientL1Loss(nn.Module):
    def __init__(self):
        super(GradientL1Loss, self).__init__()
        self.gradient_fun = kornia.filters.SpatialGradient()
        self.l1loss = L1Loss()

    def forward(self, img1, img2):
        return self.l1loss(self.gradient_fun(img1), self.gradient_fun(img2))


class GradientL2Loss(nn.Module):
    def __init__(self):
        super(GradientL2Loss, self).__init__()
        self.gradient_fun = kornia.filters.SpatialGradient()
        self.l2loss = L2Loss()

    def forward(self, img1, img2):
        return self.l2loss(self.gradient_fun(img1), self.gradient_fun(img2))


class AdaptiveGradientL2Loss(nn.Module):
    def __init__(self):
        super(AdaptiveGradientL2Loss, self).__init__()
        self.l2loss = L2Loss(reduction='')

    def forward(self, fused_img, vis, ir):
        # vis        = torch.mean(vis, dim=1, keepdim=True)
        # ir         = torch.mean(ir, dim=1, keepdim=True)
        # fused_img  = torch.mean(fused_img, dim=1, keepdim=True)

        vis_grad_lowpass = torch.abs(gradient(low_pass(vis)))
        ir_grad_lowpass = torch.abs(gradient(low_pass(ir)))
        vis_score = torch.sign(
            vis_grad_lowpass - torch.minimum(vis_grad_lowpass, ir_grad_lowpass))
        ir_score = 1 - vis_score

        fused_img_grad = gradient(fused_img)
        vis_grad = gradient(vis)
        ir_grad = gradient(ir)

        loss_gradient = torch.mean(torch.mul(vis_score.detach(), torch.square(fused_img_grad - vis_grad))) \
            + torch.mean(torch.mul(ir_score.detach(),
                         torch.square(fused_img_grad - ir_grad)))
        return loss_gradient


class DeCompositionLoss(nn.Module):
    def __init__(self):
        super(DeCompositionLoss, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def cc(self, img1, img2):
        """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
        N, C, _, _ = img1.shape
        img1 = img1.reshape(N, C, -1)
        img2 = img2.reshape(N, C, -1)
        img1 = img1 - img1.mean(dim=-1, keepdim=True)
        img2 = img2 - img2.mean(dim=-1, keepdim=True)
        _div = torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * \
            torch.sqrt(torch.sum(img2**2, dim=-1))
        cc = torch.sum(img1 * img2, dim=-1) / (self.eps + _div)
        cc = torch.clamp(cc, -1., 1.)
        return cc.mean()

    def forward(self, vis_base_feat, vis_base_detail, ir_base_feat, ir_detail_feat):
        cc_loss_B = self.cc(vis_base_feat, ir_base_feat)
        cc_loss_D = self.cc(vis_base_detail, ir_detail_feat)
        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
        return loss_decomp


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in+10*loss_grad
        return loss_total, loss_in, loss_grad
