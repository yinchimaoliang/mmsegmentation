import argparse

import mmcv
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
        '--batch_size', type=int, default=128, help='Batch size (default=128)')
    parser.add_argument(
        '--lr', type=float, default=0.01, help='Learning rate (default=0.01)')
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./work_dirs/tub_gan',
        help='Dir to save results.')
    args = parser.parse_args()

    return args


class GleasonDataset(Dataset):

    def __init__(self, data_root):
        self.data_root = data_root
        self.names = os.listdir(os.path.join(data_root, 'annotations'))

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv.imread(os.path.join(self.data_root, 'images', name))
        img = cv.resize(img, (512, 512))
        img = img / 255
        ann = cv.imread(os.path.join(self.data_root, 'annotations', name), 0)
        ann = cv.resize(ann, (512, 512))
        img = torch.from_numpy(img)
        ann = torch.from_numpy(ann)
        return img.permute(2, 0, 1), ann

    def __len__(self):
        return len(self.names)


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


class Train():

    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.args = parse_args()
        args = parse_args()
        bs = args.batch_size
        lr = args.lr
        epochs = args.epochs
        work_dir = args.work_dir
        mmcv.mkdir_or_exist(work_dir)
        self.optim_d = optim.SGD(self.discriminator.parameters(), lr=args.lr)
        self.optim_g = optim.SGD(self.generator.parameters(), lr=args.lr)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_l1 = build_loss(LOSS_L1_CFG)

    def get_loss(self, syn, real, syn_score, real_score):
        d_loss_real = self.loss_ce(real_score,
                                   torch.ones(real_score.shape[:-1]).long())
        d_loss_fake = self.loss_ce(syn_score,
                                   torch.zeros(syn_score.shape[:-1]).long())
        d_loss = d_loss_real + d_loss_fake
        g_loss_adversarial = self.loss_ce(
            syn_score,
            torch.ones(syn_score.shape[:-1]).long())
        g_content_loss = self.loss_l1(syn, real)
        g_loss = g_loss_adversarial + g_content_loss
        return d_loss, g_loss

    def train(self):
        pass


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
    syn_score = discriminator(syn_input)
    real_score = discriminator(real_input)
    train = Train()
    d_loss, g_loss = train.get_loss(syn, img, syn_score, real_score)
    print(d_loss, g_loss)
