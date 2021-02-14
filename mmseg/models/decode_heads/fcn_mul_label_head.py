import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class FCNMulLabelHead(FCNHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 label_ind=None,
                 **kwargs):
        super(FCNMulLabelHead, self).__init__(**kwargs)
        self.label_ind = label_ind

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

    def forward_train(self, img, inputs, img_metas, gt_semantic_seg, train_cfg, weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        loss_segs = []
        acc_segs = []
        if self.label_ind is not None:
            for i in self.label_ind:
                losses = self.losses(img, seg_logits, gt_semantic_seg[..., i].squeeze(-1))
                loss_segs.append(losses['loss_seg'])
                acc_segs.append(losses['acc_seg'])
            loss_segs = torch.stack(loss_segs)
            acc_segs = torch.stack(acc_segs).squeeze(1)
            if weight is None:
                weight = torch.ones_like(loss_segs)
            loss_segs *= weight
            acc_segs *= weight
            losses = dict(loss_seg=loss_segs.mean(), acc_seg=[acc_segs.mean()])
        else:
            losses = self.losses(img, seg_logits, gt_semantic_seg)
        return losses