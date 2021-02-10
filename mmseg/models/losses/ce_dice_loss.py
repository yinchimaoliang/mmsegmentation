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
                 ce_loss_weight=1.0,
                 dice_loss_weight=0.4,
                 beta=1,
                 eps=1e-7,
                 threshold=None,
                 activation='sigmoid',
                 **kwargs):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce_loss = CrossEntropyLoss(use_sigmoid, use_mask, reduction,
                                        class_weight, ce_loss_weight)

        self.dice_loss = DiceLoss(
            dice_loss_weight=dice_loss_weight,
            beta=beta,
            eps=eps,
            threshold=threshold,
            activation=activation)

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
