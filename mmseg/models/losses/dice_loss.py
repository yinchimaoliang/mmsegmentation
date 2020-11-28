import torch
from torch import nn as nn
from torch.nn import functional as F

from ..builder import LOSSES


def _get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)  # reshape 为向量
    ones = torch.eye(N).type_as(label)
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size).permute([0, 3, 1, 2])


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
        super().__init__()
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

        pr = F.interpolate(pr, size=[gt.shape[2], gt.shape[3]])
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp

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
        loss = 0
        label[label == 255] = 0
        cls_num = cls_score.shape[1]
        label = _get_one_hot(label, cls_num)
        if type(cls_score).__name__ == 'list':
            for output in cls_score:
                loss += 1 - self.f_score(output, label, self.beta, self.eps,
                                         self.threshold, self.activation)
        else:
            loss += 1 - self.f_score(cls_score, label, self.beta, self.eps,
                                     self.threshold, self.activation)

        return loss
