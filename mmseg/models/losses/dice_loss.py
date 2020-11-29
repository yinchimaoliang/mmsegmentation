import copy
import torch
from torch import nn as nn

from ..builder import LOSSES


def _get_one_hot(label, N):
    new_label = copy.deepcopy(label)
    # Remove label == 255
    new_label[new_label == 255] = 0
    size = list(new_label.size())
    new_label = new_label.view(-1)  # reshape 为向量
    ones = torch.eye(N).type_as(new_label)
    ones = ones.index_select(0, new_label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size).permute([0, 3, 1, 2])


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 class_weight=None,
                 loss_weight=1.0,
                 reduction='mean',
                 beta=1,
                 eps=1e-7,
                 threshold=None,
                 activation='sigmoid',
                 **kwargs):
        super().__init__()
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.activation = activation

    @staticmethod
    def f_score(pr,
                gt,
                beta=1,
                eps=1e-7,
                threshold=None,
                activation='sigmoid'):
        """
        Args:
            pr (torch.Tensor): A list of predicted elements
            gt (torch.Tensor):  A list of elements that are to be predicted
            beta (float): positive constant
            eps (float): epsilon to avoid zero division
            threshold: threshold for outputs binarization
        Returns:
            float: F score
        """

        if activation is None or activation == 'none':
            activation_fn = lambda x: x  # noqa: E731
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid()
        elif activation == 'softmax2d':
            activation_fn = torch.nn.Softmax2d()
        else:
            raise NotImplementedError(
                'Activation implemented for sigmoid and softmax2d')

        pr = activation_fn(pr)

        if threshold is not None:
            pr = (pr > threshold).float()

        tp = torch.sum(gt * pr, dim=[0, 2, 3])
        fp = torch.sum(pr, dim=[0, 2, 3]) - tp
        fn = torch.sum(gt, dim=[0, 2, 3]) - tp

        score = ((1 + beta**2) * tp + eps) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + eps)

        return score

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        cls_num = cls_score.shape[1]
        label_onehot = _get_one_hot(label, cls_num)
        dice_coef = self.f_score(cls_score, label_onehot, self.beta, self.eps,
                                 self.threshold, self.activation)
        if self.class_weight is not None:
            loss = (torch.ones_like(dice_coef) - dice_coef) * self.class_weight
        else:
            loss = (torch.ones_like(dice_coef) - dice_coef)
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        return self.loss_weight * loss
