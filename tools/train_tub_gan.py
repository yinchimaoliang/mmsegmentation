import torch
import torch.nn as nn

from mmseg.models.builder import build_backbone, build_head

norm_cfg = dict(type='BN', requires_grad=True)

BACKBONE_CFG = dict(
    type='UNet',
    in_channels=1,
    base_channels=64,
    num_stages=5,
    strides=(1, 1, 1, 1, 1),
    enc_num_convs=(2, 2, 2, 2, 2),
    dec_num_convs=(2, 2, 2, 2),
    downsamples=(True, True, True, True),
    enc_dilations=(1, 1, 1, 1, 1),
    dec_dilations=(1, 1, 1, 1),
    with_cp=False,
    conv_cfg=None,
    norm_cfg=norm_cfg,
    act_cfg=dict(type='ReLU'),
    upsample_cfg=dict(type='InterpConv'),
    norm_eval=False)

DECODE_HEAD_CFG = dict(
    type='FCNHead',
    in_channels=64,
    in_index=4,
    channels=64,
    num_convs=1,
    concat_input=False,
    dropout_ratio=0.1,
    num_classes=3,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))


class Generator(nn.Module):

    def __init__(self,
                 backbone_cfg=BACKBONE_CFG,
                 decode_head_cfg=DECODE_HEAD_CFG):
        super(Generator, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.decode_head = build_head(decode_head_cfg)

    def forward(self, x):
        features = self.backbone(x)
        output = self.decode_head(features)
        return output


if __name__ == '__main__':
    generator = Generator()
    x = torch.rand(4, 1, 512, 512)
    syn = generator(x)
    print(syn)
