from torch import nn as nn

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .dice_loss import DiceLoss


@LOSSES.register_module()
class CrossEntropyDiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 loss_weight=1.0,
                 reduction='mean',
                 beta=1,
                 eps=1e-7,
                 threshold=None,
                 activation='sigmoid',
                 gauss_scale=None,
                 gauss_kernel=None,
                 gauss_sigma=None,
                 **kwargs):
        super(CrossEntropyDiceLoss, self).__init__()
        self.ce_loss = CrossEntropyLoss(use_sigmoid,
                 use_mask,
                 reduction,
                 class_weight,
                 loss_weight,
                 gauss_scale,
                 gauss_kernel,
                 gauss_sigma)

        self.dice_loss = DiceLoss(class_weight,
                 loss_weight,
                 reduction,
                 beta,
                 eps,
                 threshold,
                 activation,
                 gauss_scale,
                 gauss_kernel,
                 gauss_sigma)

    def forward(self,
                img,
                logits,
                labels,
                weight=None,
                mul_label_weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        return self.dice_loss(img,
                logits,
                labels,
                weight,
                mul_label_weight,
                avg_factor,
                reduction_override,
                ignore_index, **kwargs) + self.ce_loss(
            img,
            logits,
            labels,
            weight,
            avg_factor,
            reduction_override,
            mul_label_weight,
            ignore_index=ignore_index,
            **kwargs)
