import torch
import torch.nn as nn

from ..builder import HEADS, build_backbone, build_neck
from .fcn_head import FCNHead

from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses import accuracy


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
                 wei_net_backbone,
                 wei_net_neck,
                 wei_net_in_channels,
                 wei_net_out_channels,
                 mul_label_ind=None,
                 final_label_ind=None,
                 **kwargs):
        super(FCNMulLabelHead, self).__init__(**kwargs)
        self.label_ind = mul_label_ind
        self.final_label_ind = final_label_ind
        self.wei_net_backbone = build_backbone(wei_net_backbone)
        self.wei_net_neck = build_neck(wei_net_neck)
        self.wei_net_fc = nn.Linear(wei_net_in_channels, wei_net_out_channels)
        self.wei_net_softmax = nn.Softmax(dim=1)

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
        weight = self.wei_net_softmax(self.wei_net_fc(self.wei_net_neck(self.wei_net_backbone(img)[-1])))
        losses = self.losses(img, seg_logits, gt_semantic_seg[..., self.label_ind], weight)
        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, img, seg_logit, seg_label, mul_label_weight):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:4],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            img,
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index,
            mul_label_weight=mul_label_weight)
        loss['acc_seg'] = accuracy(seg_logit, seg_label[..., self.final_label_ind])
        return loss