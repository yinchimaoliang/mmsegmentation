import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import get_one_hot
from ..builder import LOSSES

@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, one_hot=True, ignore=None, loss_weight=1, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.one_hot = one_hot
        self.ignore = ignore
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, img, input, target, weight, ignore_index):
        '''
        only support ignore at 0
        '''
        B, C, H, W = input.size()
        if weight is None:
            weight = torch.ones_like(input)
        if ignore_index is not None:
            mask = (target != ignore_index)
            mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2]).expand_as(weight)
            weight *= mask

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

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        if self.reduction == 'sum':
            loss = batch_loss.sum()
        return loss * self.loss_weight