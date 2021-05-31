import argparse
import os
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.cnn import ConvModule
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from mmseg.models import build_backbone, build_head, build_loss

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

LOSS_CE_CFG = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
LOSS_L1_CFG = dict(type='L1Loss', loss_weight=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train tub-gan')
    parser.add_argument(
        '--batch_size', type=int, default=2, help='Batch size (default=128)')
    parser.add_argument(
        '--lr', type=float, default=0.01, help='Learning rate (default=0.01)')
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./work_dirs/tub_gan',
        help='Dir to save results.')

    parser.add_argument('--cuda', action='store_true', help='Enable cuda')
    args = parser.parse_args()

    return args


class GleasonDataset(Dataset):

    def __init__(self, data_root):
        self.data_root = data_root
        self.names = os.listdir(os.path.join(data_root, 'annotations'))

    def __getitem__(self, idx):
        name = self.names[idx]
        img = mmcv.imread(os.path.join(self.data_root, 'images', name))
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
                 backbone_cfg=BACKBONE_CFG,
                 decode_head_cfg=DECODE_HEAD_CFG):
        super(SynGenerator, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.decode_head = build_head(decode_head_cfg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        output = self.sigmoid(self.decode_head(features))
        return output


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
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        feature = self.net(input)
        score = self.sigmoid(self.fc(feature.view(feature.shape[0], -1)))
        return score


class StyleGenerator(nn.Module):

    def __init__(self, in_channels=3, act_cfg=dict(type='LeakyReLU')):
        super(StyleGenerator, self).__init__()
        self.layer1 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid())
        self.layer2 = nn.Sequential(
            ConvModule(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid())
        self.layer3 = nn.Sequential(
            ConvModule(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid())
        self.layer4 = nn.Sequential(
            ConvModule(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid())
        self.layer5 = nn.Sequential(
            ConvModule(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid())

    def forward(self, x):
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        feature5 = self.layer5(feature4)
        feature1 = feature1.reshape(
            (-1, feature1.shape[1], feature1.shape[2] * feature1.shape[3]))
        feature2 = feature2.reshape(
            (-1, feature2.shape[1], feature2.shape[2] * feature2.shape[3]))
        feature3 = feature3.reshape(
            (-1, feature3.shape[1], feature3.shape[2] * feature3.shape[3]))
        feature4 = feature4.reshape(
            (-1, feature4.shape[1], feature4.shape[2] * feature4.shape[3]))
        feature5 = feature5.reshape(
            (-1, feature5.shape[1], feature5.shape[2] * feature5.shape[3]))
        feature1_T = feature1.transpose(2, 1)
        feature2_T = feature2.transpose(2, 1)
        feature3_T = feature3.transpose(2, 1)
        feature4_T = feature4.transpose(2, 1)
        feature5_T = feature5.transpose(2, 1)
        gram1 = torch.matmul(feature1, feature1_T) / (
            feature1.shape[1] * feature1.shape[2])
        gram2 = torch.matmul(feature2, feature2_T) / (
            feature2.shape[1] * feature2.shape[2])
        gram3 = torch.matmul(feature3, feature3_T) / (
            feature3.shape[1] * feature3.shape[2])
        gram4 = torch.matmul(feature4, feature4_T) / (
            feature4.shape[1] * feature4.shape[2])
        gram5 = torch.matmul(feature5, feature5_T) / (
            feature5.shape[1] * feature5.shape[2])

        return [gram1, gram2, gram3, gram4, gram5]


class Train():

    def __init__(self):
        self.syn_generator = SynGenerator()
        self.discriminator = Discriminator()
        self.style_generator = StyleGenerator()
        self.args = parse_args()
        args = parse_args()
        bs = args.batch_size
        lr = args.lr
        self.work_dir = args.work_dir
        mmcv.mkdir_or_exist(self.work_dir)
        mmcv.mkdir_or_exist(osp.join(self.work_dir, 'syns'))
        self.epochs = args.epochs
        self.cuda = args.cuda
        self.optim_d = optim.SGD(self.discriminator.parameters(), lr=lr)
        self.optim_g = optim.SGD(self.syn_generator.parameters(), lr=lr)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_l1 = build_loss(LOSS_L1_CFG)
        train_dataset = GleasonDataset(
            data_root='./data/gleason_2019/train/train')
        self.train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=bs)

    def get_g_loss(self, syn_score, syn, real, style_real, style_syn):
        g_adversarial_loss = self.loss_ce(
            syn_score,
            syn_score.new_ones(syn_score.shape[:-1]).long())
        g_content_loss = self.loss_l1(syn, real)
        g_style_loss = 0
        for i in range(len(style_real)):
            g_style_loss = g_style_loss + self.loss_l1(style_real[i],
                                                       style_syn[i])
        # x = self.loss_l1(syn[..., 1:, :], syn[..., -1:, :])
        # y = self.loss_l1(syn[..., :, 1:], syn[..., :, -1:])
        # g_tv_loss = x + y
        g_loss = g_adversarial_loss + g_content_loss + 10 * g_style_loss
        return g_loss

    def get_d_loss(self, real_score, syn_score):
        d_real_loss = self.loss_ce(
            real_score,
            real_score.new_ones(real_score.shape[:-1]).long())
        d_fake_loss = self.loss_ce(
            syn_score,
            syn_score.new_zeros(syn_score.shape[:-1]).long())
        d_loss = d_real_loss + d_fake_loss
        return d_loss

    def train_step(self, train_x, train_y):
        syn = self.syn_generator(train_y)
        style_real = self.style_generator(train_x)
        style_syn = self.style_generator(syn)
        syn_input = torch.cat([syn, train_y], dim=1)
        real_input = torch.cat([train_x, train_y], dim=1)
        syn_score = self.discriminator(syn_input)
        real_score = self.discriminator(real_input)
        self.optim_d.zero_grad()
        d_loss = self.get_d_loss(real_score, syn_score)
        d_loss.backward(retain_graph=True)
        self.optim_d.step()
        self.optim_g.zero_grad()
        g_loss = self.get_g_loss(syn_score, syn, train_x, style_real,
                                 style_syn)
        g_loss.backward(retain_graph=True)
        self.optim_g.step()
        return d_loss, g_loss, syn

    def show_result(self, syn):
        bs = syn.shape[0]
        for i in range(bs):
            img = syn[i].detach().permute(1, 2, 0).cpu().numpy()
            img = img * 255
            mmcv.imwrite(
                img.astype(np.uint8),
                os.path.join(self.work_dir, 'syns', f'{i}.png'))

    def train(self):
        for epoch_idx in range(self.epochs):
            self.syn_generator.train()
            self.discriminator.train()
            self.style_generator.train()
            if self.cuda:
                self.syn_generator.cuda()
                self.discriminator.cuda()
                self.style_generator.cuda()
            for batch_idx, (train_x, train_y) in enumerate(self.train_loader):
                if self.cuda:
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                d_loss, g_loss, syn = self.train_step(train_x, train_y)
                if batch_idx % 10 == 0:
                    self.show_result(syn)
                    print(f'd_loss: {d_loss} | g_loss:{g_loss}')


if __name__ == '__main__':
    generator = SynGenerator()
    discriminator = Discriminator()
    img = torch.rand(4, 3, 512, 512)
    x = torch.rand(4, 1, 512, 512)
    syn = generator(x)
    gt = torch.randint(0, 4, (4, 1, 512, 512)).float()
    gt = gt / 4
    syn_input = torch.cat([syn, gt], dim=1)
    real_input = torch.cat([img, gt], dim=1)
    syn_score = discriminator(syn_input)
    real_score = discriminator(real_input)
    train = Train()
    train.train()
