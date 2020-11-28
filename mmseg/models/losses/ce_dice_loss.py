from torch import nn as nn

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .dice_loss import DiceLoss


@LOSSES.register_module()
class CrossEntropyDiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 beta=1,
                 eps=1e-7,
                 threshold=None,
                 activation='sigmoid'):
        self.ce_loss = CrossEntropyLoss(use_sigmoid, use_mask, reduction,
                                        class_weight, loss_weight)

        self.dice_loss = DiceLoss(beta, eps, threshold, activation)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        return self.dice_loss(cls_score, label, weight, avg_factor,
                              reduction_override, **kwargs) + self.ce_loss(
                                  cls_score, label, weight, avg_factor,
                                  reduction_override, **kwargs)
