from torch import nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_func = nn.L1Loss()

    def forward(self, feature1, feature2, **kwargs):
        loss = self.loss_func(feature1, feature2)
        return self.loss_weight * loss
