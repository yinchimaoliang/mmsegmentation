import copy
import torch
from torch import nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import gkern
from mmseg.models.utils import get_one_hot


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

    @staticmethod
    def f_score(pr,
                gt,
                weight,
                beta=1,
                eps=1e-7,
                threshold=None,
                activation='sigmoid',
                ignore_index=255,
                gauss_scale=None,
                gauss_kernel=None,
                gauss_sigma=None
                ):
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

        # gt[gt==255] = 0
        tp = torch.sum(gt * pr * weight, dim=[2, 3])
        fp = torch.sum(pr*(1-gt) * weight, dim=[2, 3])
        fn = torch.sum(gt*(1-pr) * weight, dim=[2, 3])

        score = ((1 + beta**2) * tp + eps) / (
            (1 + beta**2) * tp + beta**2 * fn + fp + eps)

        return score

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
        cls_num = logits.shape[1]
        if labels.ndim == 3:
            labels = labels.unsqueeze(dim=3)
        losses = []
        for i in range(labels.shape[3]):
            label_onehot = get_one_hot(labels[..., i], cls_num)
            if weight is None:
                weight = torch.ones_like(logits)
            if self.gauss_scale is not None:
                kernel = gkern(self.gauss_kernel, self.gauss_sigma)
                kernel = torch.from_numpy(kernel).to(img).expand(1,3,self.gauss_kernel,self.gauss_kernel)
                img_blurred = F.conv2d(img,nn.Parameter(kernel), padding=(self.gauss_kernel-1)//2)
                weight = 1 + self.gauss_scale * torch.abs(img_blurred - torch.mean(img, dim=1, keepdim=True))
                weight = weight.repeat(1, cls_num, 1, 1)
            if self.class_weight is not None:
                weight = weight * torch.tensor(self.class_weight).reshape(1, cls_num, 1, 1).expand_as(weight).to(weight)
            dice_coef = self.f_score(logits, label_onehot, weight, self.beta, self.eps,
                                     self.threshold, self.activation)
            loss = torch.ones_like(dice_coef) - dice_coef
            if self.reduction == 'sum':
                loss = torch.sum(loss, dim=1)
            elif self.reduction == 'mean':
                loss = torch.mean(loss, dim=1)
            losses.append(loss)

        losses = torch.stack(losses).T
        if mul_label_weight is not None:
            losses = losses * mul_label_weight

        if self.reduction == 'mean':
            losses = losses.mean()
        if self.reduction == 'sum':
            losses = losses.sum()
        return self.loss_weight * losses
