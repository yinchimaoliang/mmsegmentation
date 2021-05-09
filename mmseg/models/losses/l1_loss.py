import copy

import torch
import torch.nn.functional as F
from torch import nn as nn

from mmseg.models.utils import get_one_hot
from ..builder import LOSSES
from .utils import gkern


@LOSSES.register_module()
class L1Loss(nn.Module):

    def __init__(self,
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
        super().__init__()
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.activation = activation
        if gauss_scale is not None:
            assert gauss_kernel is not None
            assert gauss_sigma is not None
        self.gauss_scale = gauss_scale
        self.gauss_kernel = gauss_kernel
        self.gauss_sigma = gauss_sigma
        self.loss_func = nn.L1Loss()

    def forward(self, feature1, feature2, **kwargs):
        loss = self.loss_func(feature1, feature2)
        return self.loss_weight * loss
