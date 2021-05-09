import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

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


class Discriminator(nn.Module):

    def __init__(self,
                 img_size=(512, 512),
                 in_channels=4 + 4,
                 act_cfg=dict(type='LeakyReLU')):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=4, stride=4),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=4, stride=4),
            ConvModule(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=4, stride=4))
        self.fc = nn.Linear(
            int(img_size[0] * img_size[1] / (64 * 64) * 128), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feature = self.net(input)
        score = self.sigmoid(self.fc(feature.view(feature.shape[0], -1)))
        return score


if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()
    img = torch.rand(4, 3, 512, 512)
    x = torch.rand(4, 1, 512, 512)
    syn = generator(x)
    gt = torch.randint(0, 4, (4, 1, 512, 512)).float()
    gt = gt / 4
    syn_input = torch.cat([syn, gt], dim=1)
    real_input = torch.cat([img, gt], dim=1)
    input = torch.cat([syn_input, real_input], dim=1)
    score = discriminator(input)
