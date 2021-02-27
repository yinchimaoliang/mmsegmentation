import torch
from torch.nn import functional as F
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv import cnn

from mmseg.ops import resize
from mmcv.runner import force_fp32
from ..builder import HEADS, build_backbone, build_loss
from .aspp_head import ASPPHead, ASPPModule
from ..losses import accuracy


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)


@HEADS.register_module()
class DepthwiseSeparableASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super(DepthwiseSeparableASPPHead, self).__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class DepthwiseSeparableASPPMulLabelHead(DepthwiseSeparableASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, wei_net_backbone,
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
                 **kwargs):
        super(DepthwiseSeparableASPPMulLabelHead, self).__init__(**kwargs)
        self.label_ind = mul_label_ind
        self.final_label_ind = final_label_ind
        self.wei_net_backbone = build_backbone(wei_net_backbone)
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
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output

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
        loss_single_label = self.loss_single(img, seg_logit, seg_label[..., self.final_label_ind], weight=seg_weight
                                             , ignore_index=self.ignore_index)
        loss_mul_label = self.loss_decode(
            img,
            seg_logit,
            seg_label[..., self.label_ind],
            weight=seg_weight,
            ignore_index=self.ignore_index,
            mul_label_weight=mul_label_weight)

        iter_num_sig = self.sigma * (torch.sigmoid(torch.tensor(self.iter_num // self.loss_step).float()) - 1 / 2) * 2
        iter_num_sig = iter_num_sig.type_as(seg_logit)
        loss['loss_seg'] = (1 / (1 + iter_num_sig)) * loss_single_label + (
                    iter_num_sig / (1 + iter_num_sig)) * loss_mul_label

        self.iter_num += 1

        loss['acc_seg'] = accuracy(seg_logit, seg_label[..., self.final_label_ind])
        return loss

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
        weight = self.wei_net_softmax(self.wei_net_conv(self.wei_net_backbone(img)[-1]))
        weight = F.interpolate(weight, gt_semantic_seg.shape[2:4])
        losses = self.losses(img, seg_logits, gt_semantic_seg, weight)
        return losses
