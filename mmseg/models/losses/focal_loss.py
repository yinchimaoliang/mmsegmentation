import torch.nn as nn
from mmcv.ops import softmax_focal_loss as softmax_focal_loss_func

from mmseg.models.losses.utils import weight_reduce_loss
from ..builder import LOSSES


def softmax_focal_loss(pred,
                       target,
                       gamma,
                       alpha,
                       weight=None,
                       class_weight=None,
                       reduction='mean',
                       avg_factor=None,
                       ignore_index=-100):
    pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target = target.flatten().contiguous()
    valid = (target != ignore_index)
    loss = softmax_focal_loss_func(pred[valid], target[valid], gamma, alpha,
                                   weight, 'none')
    reduce_loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)
    return reduce_loss


def sigmoid_focal_loss(pred,
                       label,
                       weight=None,
                       class_weight=None,
                       reduction='mean',
                       avg_factor=None,
                       ignore_index=-100):
    pass


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 gamma,
                 alpha,
                 use_sigmoid=False,
                 class_weight=None,
                 reduction='mean',
                 loss_weight=1.0,
                 *args,
                 **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if use_sigmoid:
            self.cls_criterion = sigmoid_focal_loss
        else:
            self.cls_criterion = softmax_focal_loss
        self.class_weight = class_weight
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.gamma,
            self.alpha,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(gamma={self.gamma}, '
        s += f'alpha={self.alpha}, '
        s += f'reduction={self.reduction})'
        return s
