import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.cnn import kaiming_init  # noqa F401
from mmcv.cnn import ConvModule, constant_init
from torch import nn, optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, Dataset

from mmseg.models import build_backbone, build_head, build_loss  # noqa F401

norm_cfg = dict(type='BN', requires_grad=True)
U_LAYER_CFG = dict(type='deconv', scale_factor=2)

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

FEATURE_NET_CFG = dict(
    type='ResNetV1c',
    depth=18,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    dilations=(1, 1, 2, 4),
    strides=(1, 2, 1, 1),
    norm_cfg=norm_cfg,
    norm_eval=False,
    style='pytorch',
    contract_dilation=True)

STYLE_LAYERS = [0, 1, 3]
CONTENT_LAYERS = [2]

LOSS_CE_CFG = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
LOSS_L1_CFG = dict(type='L1Loss', loss_weight=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train tub-gan')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Batch size (default=128)')
    parser.add_argument(
        '--g-lr',
        type=float,
        default=0.002,
        help='Learning rate (default=0.01)')
    parser.add_argument(
        '--d-lr',
        type=float,
        default=0.0001,
        help='Learning rate (default=0.01)')
    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument(
        '--train-data-root',
        default='./data/DRIVE/train',
        help='Dir for style image path.')
    parser.add_argument(
        '--style-img-root',
        default='data/DRIVE/test',
        help='Root dir for training data.')
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./work_dirs/tub_gan',
        help='Dir to save results.')

    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    return args


class DriveDataset(Dataset):

    def __init__(self, data_root):
        self.data_root = data_root
        self.names = os.listdir(os.path.join(data_root, 'annotations'))

    def __getitem__(self, idx):
        name = self.names[idx]
        img = mmcv.imread(
            os.path.join(self.data_root, 'images',
                         name.replace('_manual1', '')))
        img = mmcv.imresize(img, (512, 512))
        img = img / 255
        ann = mmcv.imread(os.path.join(self.data_root, 'annotations', name), 0)
        ann = mmcv.imresize(ann, (512, 512))
        img = torch.from_numpy(img)
        ann = torch.from_numpy(ann)
        ann = ann.unsqueeze(2)
        return img.permute(2, 0, 1).float(), ann.permute(2, 0, 1).float()

    def __len__(self):
        return len(self.names)


class SynGenerator(nn.Module):

    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 act_cfg=dict(type='LeakyReLU'),
                 out_channels=3):
        super(SynGenerator, self).__init__()
        self.d_layer1 = ConvModule(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=4,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.d_layer2 = ConvModule(
            in_channels=base_channels,
            out_channels=2 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.d_layer3 = ConvModule(
            in_channels=2 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.d_layer4 = ConvModule(
            in_channels=4 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.d_layer5 = ConvModule(
            in_channels=8 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.d_layer6 = ConvModule(
            in_channels=8 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.u_layer1 = ConvModule(
            in_channels=8 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))
        self.u_layer2 = ConvModule(
            in_channels=12 * base_channels,
            out_channels=8 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))
        self.u_layer3 = ConvModule(
            in_channels=12 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))
        self.u_layer4 = ConvModule(
            in_channels=16 * base_channels,
            out_channels=4 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))
        self.u_layer5 = ConvModule(
            in_channels=8 * base_channels,
            out_channels=2 * base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))
        self.u_layer6 = ConvModule(
            in_channels=4 * base_channels,
            out_channels=base_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=None,
            act_cfg=act_cfg,
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))

        self.final_layer = ConvModule(
            in_channels=2 * base_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            norm_cfg=None,
            act_cfg=dict(type='Tanh'),
            conv_cfg=dict(type='ConvTranspose2d', output_padding=1))
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, x, z=None):
        features1 = self.d_layer1(x)
        features2 = self.d_layer2(features1)
        features3 = self.d_layer3(features2)
        features4 = self.d_layer4(features3)
        features5 = self.d_layer5(features4)
        features6 = self.d_layer6(features5)
        features5 = torch.cat([features5, self.u_layer1(features6)], dim=1)
        features4 = torch.cat([features4, self.u_layer2(features5)], dim=1)
        features3 = torch.cat([features3, self.u_layer4(features4)], dim=1)
        features2 = torch.cat([features2, self.u_layer5(features3)], dim=1)
        features1 = torch.cat([features1, self.u_layer6(features2)], dim=1)
        syn = self.relu(self.final_layer(features1))
        return syn

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


class Discriminator(nn.Module):

    def __init__(self,
                 img_size=(512, 512),
                 in_channels=4,
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
            int(img_size[0] * img_size[1] / (64 * 64) * 128), 2)
        self.init_weights()

    def forward(self, input):
        feature = self.net(input)
        feature = feature.reshape(feature.shape[0], -1)
        score = self.fc(feature)
        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


class FeatureGenerator(nn.Module):

    def __init__(self, in_channels=3, act_cfg=dict(type='LeakyReLU')):
        super(FeatureGenerator, self).__init__()
        self.net = build_backbone(FEATURE_NET_CFG)
        self.init_weights()

    def forward_content(self, x):
        features = self.net.forward(x)
        content_features = []
        for i, feature in enumerate(features):
            if i in CONTENT_LAYERS:
                content_features.append(feature)
        return content_features

    def forward_style(self, x):
        features = self.net.forward(x)
        style_features = []
        for i, feature in enumerate(features):
            if i in STYLE_LAYERS:
                feature = feature.reshape(feature.shape[0], feature.shape[1],
                                          -1)
                feature_T = feature.transpose(1, 2)
                style_features.append(
                    torch.bmm(feature, feature_T) /
                    (feature.shape[0] * feature.shape[2] * len(STYLE_LAYERS)))

        return style_features

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


class Train():

    def __init__(self):
        self.syn_generator = SynGenerator()
        self.discriminator = Discriminator()
        self.feature_generator = FeatureGenerator()
        self.args = parse_args()
        args = parse_args()
        self.bs = args.batch_size
        d_lr = args.d_lr
        g_lr = args.g_lr
        self.work_dir = args.work_dir
        train_data_root = args.train_data_root
        mmcv.mkdir_or_exist(self.work_dir)
        mmcv.mkdir_or_exist(osp.join(self.work_dir, 'syns'))
        self.epochs = args.epochs
        self.cuda = args.cuda
        self.optim_d = optim.SGD(self.discriminator.parameters(), lr=d_lr)
        self.optim_g = optim.SGD(self.syn_generator.parameters(), lr=g_lr)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_l1 = build_loss(LOSS_L1_CFG)
        train_dataset = DriveDataset(data_root=train_data_root)
        self.style_dataset = DriveDataset(data_root=args.style_img_root)
        self.train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.bs)

    def get_g_loss(self, train_x, train_y, style_x):
        syn = self.syn_generator(train_y)
        style_real = self.feature_generator.forward_style(style_x)
        style_syn = self.feature_generator.forward_style(syn)
        content_real = self.feature_generator.forward_content(style_x)
        content_syn = self.feature_generator.forward_content(syn)
        syn_input = torch.cat([syn, train_y], dim=1)
        syn_score = self.discriminator(syn_input)
        dev_loss = self.loss_l1(syn, train_x)
        g_adversarial_loss = self.loss_ce(
            syn_score,
            syn_score.new_ones(syn_score.shape[:-1]).long())
        g_content_loss = 0
        for i in range(len(content_real)):
            g_content_loss = g_content_loss + self.loss_l1(
                content_real[i], content_syn[i])
        # g_content_loss.backward(retain_graph=True)
        g_style_loss = 0
        for i in range(len(style_real)):
            g_style_loss = g_style_loss + self.loss_l1(style_real[i],
                                                       style_syn[i])
        # g_style_loss.backward(retain_graph=True)
        g_loss = g_adversarial_loss + g_content_loss + 10 * g_style_loss
        +100 * dev_loss
        # g_loss = g_adversarial_loss + dev_loss
        return g_loss

    def get_d_loss(self, train_x, train_y):
        syn = self.syn_generator(train_y)
        syn_input = torch.cat([syn, train_y], dim=1)
        real_input = torch.cat([train_x, train_y], dim=1)
        syn_score = self.discriminator(syn_input)
        real_score = self.discriminator(real_input)
        d_real_loss = self.loss_ce(
            real_score,
            real_score.new_ones(real_score.shape[:-1]).long())
        d_fake_loss = self.loss_ce(
            syn_score,
            syn_score.new_zeros(syn_score.shape[:-1]).long())
        d_loss = d_real_loss + d_fake_loss
        return d_loss

    def train_step(self, train_x, train_y, style_x):
        self.optim_g.zero_grad()
        g_loss = self.get_g_loss(train_x, train_y, style_x)
        g_loss.backward(retain_graph=True)
        self.optim_g.step()
        self.optim_g.zero_grad()
        g_loss = self.get_g_loss(train_x, train_y, style_x)
        g_loss.backward(retain_graph=True)
        self.optim_g.step()
        # g_loss = 0
        self.optim_d.zero_grad()
        d_loss = self.get_d_loss(train_x, train_y)
        d_loss.backward(retain_graph=True)
        self.optim_d.step()
        return d_loss, g_loss

    def show_result(self, syn, idx):
        bs = syn.shape[0]
        for i in range(bs):
            img = syn[i].detach().permute(1, 2, 0).cpu().numpy()
            img = img * 255
            mmcv.imwrite(
                img.astype(np.uint8),
                os.path.join(self.work_dir, 'syns', f'{idx}_{i}.png'))

    def train(self):
        mmcv.mkdir_or_exist(osp.join(self.work_dir, 'syns'))
        for _ in range(self.epochs):
            self.syn_generator.train()
            self.discriminator.train()
            self.feature_generator.train()
            if self.cuda:
                self.syn_generator.cuda()
                self.discriminator.cuda()
                self.feature_generator.cuda()
            for batch_idx, (train_x, train_y) in enumerate(self.train_loader):
                style_x = torch.stack([self.style_dataset[0][0]
                                       ]).repeat(train_x.shape[0], 1, 1, 1)
                if self.cuda:
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                    style_x = style_x.cuda()
                d_loss, g_loss = self.train_step(train_x, train_y, style_x)
                syn = self.syn_generator(train_y)
                self.show_result(syn, batch_idx)
                print(f'd_loss: {d_loss} | g_loss:{g_loss}')


if __name__ == '__main__':
    train = Train()
    train.train()
