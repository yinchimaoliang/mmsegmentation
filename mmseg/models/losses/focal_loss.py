import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import get_one_hot
from ..builder import LOSSES
from .utils import gkern

@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, class_weight=None, one_hot=True, ignore=None, loss_weight=1, reduction='mean',
                 gauss_scale=None,
                 gauss_kernel=None,
                 gauss_sigma=None,
                 **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.class_weight = class_weight
        self.one_hot = one_hot
        self.ignore = ignore
        if gauss_scale is not None:
            assert gauss_kernel is not None
            assert gauss_sigma is not None
        self.gauss_scale = gauss_scale
        self.gauss_kernel = gauss_kernel
        self.gauss_sigma = gauss_sigma
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, img, input, targets, weight, ignore_index, mul_label_weight=None):
        '''
        only support ignore at 0
        '''
        B, C, H, W = input.size()

        losses = []
        if targets.ndim == 3:
            targets = targets.unsqueeze(dim=3)
        if weight is None:
            weight = torch.ones_like(input)
        if self.class_weight is not None:
            weight *= torch.tensor(self.class_weight).reshape(1, C, 1, 1).expand_as(weight).to(weight)

        if self.gauss_scale is not None:
            kernel = gkern(self.gauss_kernel, self.gauss_sigma)
            kernel = torch.from_numpy(kernel).to(img).expand(1, 3, self.gauss_kernel, self.gauss_kernel)
            img_blurred = F.conv2d(img, nn.Parameter(kernel), padding=(self.gauss_kernel - 1) // 2)
            img_weight = 1 + self.gauss_scale * torch.abs(img_blurred - torch.mean(img, dim=1, keepdim=True))
            img_weight = img_weight.squeeze(1)

        for i in range(targets.shape[3]):
            target = targets[..., i]
            if ignore_index is not None:
                mask = (target != ignore_index)
                mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2]).expand_as(weight)
                single_weight = mask * weight

            if self.one_hot: target = get_one_hot(target, C)

            # input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
            probs = F.softmax(input, dim=1)
            probs = (probs * target)
            probs = probs.clamp(self.eps, 1. - self.eps)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)
            batch_loss *= single_weight
            if self.reduction == 'mean':
                loss = batch_loss.mean(dim=1)
            if self.reduction == 'sum':
                loss = batch_loss.sum(dim=1)
            loss *= img_weight
            losses.append(loss)

        losses = torch.stack(losses).permute(1, 0, 2, 3)

        if mul_label_weight is not None:
            losses *= mul_label_weight

        if self.reduction == 'mean':
            losses = losses.mean()
        if self.reduction == 'sum':
            losses = losses.sum()
        return losses * self.loss_weight