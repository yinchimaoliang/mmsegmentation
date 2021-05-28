import torch.nn as nn
from mmcv.ops import softmax_focal_loss

from ..builder import LOSSES


@LOSSES.register_module()
class SoftmaxFocalLoss(nn.Module):

    def __init__(self,
                 gamma,
                 alpha,
                 weight=None,
                 reduction='mean',
                 *args,
                 **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self,
                input,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        target[target == 255] = 0
        input = input.permute(0, 2, 3, 1).reshape(-1, input.shape[1])
        return softmax_focal_loss(input, target.flatten(), self.gamma,
                                  self.alpha, self.weight, self.reduction)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s
