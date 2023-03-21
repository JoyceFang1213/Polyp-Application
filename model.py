from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import pvt_v2
from timm.models.vision_transformer import _cfg


# python train.py --dataset=Kvasir --train-dataset-root=/home/u8961288/polyp/TrainDataset/ --test-dataset-root=/home/u8961288/polyp/TestDataset/Kvasir/  --weight 1450_100_4
class sa_layer(nn.Module):
    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

    
class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(),
            sa_layer(out_channels, 8)
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(),
            sa_layer(out_channels, 8)
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)
    
    
class FB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            sa_layer(out_channels, 8)
        )

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h 


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    def __init__(self):

        super().__init__()

        embed_dims = [64, 128, 320, 512]
        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=embed_dims,
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        # checkpoint = torch.load("pvt_v2_b3.pth")
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    FB(embed_dims[i], 64), FB(64, 64), nn.Upsample(scale_factor=2, mode="bilinear")
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64), nn.Upsample(scale_factor=2, mode="bilinear")))
            
    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(
                self.LE[i](
                    F.interpolate(pyramid[i], scale_factor=2, mode="bilinear")
                    if i == len(pyramid) - 1 else pyramid[i]
                )
            )

        l_list = []
        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            if self.training:
                l_list.append(l)
            l_i = l
        
        return l, l_list


class FCBFormer(nn.Module):
    def __init__(self, size=352):
        super().__init__()
        self.TB = TB()
        self.PH = nn.Sequential(
            RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x1, l_list = self.TB(x)
        out = self.PH(x1)

        if self.training:
            return out, l_list
        else:
            return out
