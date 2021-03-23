import torch
import torch.nn as nn
from mmcv import cnn
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmseg.ops import resize
from ..builder import HEADS, build_backbone, build_loss
from ..losses import accuracy
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
                 wei_net_backbone,
                 wei_net_conv,
                 mul_label_ind=None,
                 final_label_ind=None,
                 pretrained=None,
                 loss_single=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 sigma=5,
                 loss_step=100,
                 num_experts=3,
                 fc_in_channels=147456,
                 **kwargs):
        super(FCNMulLabelHead, self).__init__(**kwargs)
        self.label_ind = mul_label_ind
        self.final_label_ind = final_label_ind
        self.wei_net_backbone = build_backbone(wei_net_backbone)
        self.wei_expert_fc = nn.Linear(fc_in_channels, num_experts)
        self.wei_expert_pool = nn.AvgPool2d(kernel_size=256)
        self.wei_expert_softmax = nn.Softmax(dim=1)
        self.wei_net_conv = cnn.build_conv_layer(wei_net_conv)
        self.wei_net_softmax = nn.Softmax(dim=1)
        self.wei_net_backbone.init_weights(pretrained)
        self.loss_single = build_loss(loss_single)
        self.sigma = sigma
        self.iter_num = 0
        self.loss_step = loss_step

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output

    def forward_train(self,
                      img,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      weight_map=None):
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
        feature_map = self.wei_net_backbone(img)[-1]
        bs = feature_map.shape[0]
        weight_map = self.wei_net_softmax(self.wei_net_conv(feature_map))
        weight_expert = self.wei_expert_softmax(
            self.wei_expert_fc(self.wei_expert_pool(feature_map).view(bs, -1)))
        weight_expert = weight_expert.reshape(
            [weight_expert.shape[0], weight_expert.shape[1], 1,
             1]).expand_as(weight_map)
        weight_map = weight_expert * weight_map
        weight_map = F.interpolate(weight_map, gt_semantic_seg.shape[2:4])
        losses = self.losses(img, seg_logits, gt_semantic_seg, weight_map)
        return losses

    @force_fp32(apply_to=('seg_logit', ))
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
        loss_single_label = self.loss_single(
            img,
            seg_logit,
            seg_label[..., self.final_label_ind],
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss_mul_label = self.loss_decode(
            img,
            seg_logit,
            seg_label[..., self.label_ind],
            weight=seg_weight,
            ignore_index=self.ignore_index,
            mul_label_weight=mul_label_weight)

        iter_num_sig = self.sigma * (torch.sigmoid(
            torch.tensor(self.iter_num // self.loss_step).float()) - 1 / 2) * 2
        iter_num_sig = iter_num_sig.type_as(seg_logit)
        loss['loss_seg'] = (1 / (1 + iter_num_sig)) * loss_single_label + (
            iter_num_sig / (1 + iter_num_sig)) * loss_mul_label

        self.iter_num += 1

        loss['acc_seg'] = accuracy(seg_logit, seg_label[...,
                                                        self.final_label_ind])
        return loss
