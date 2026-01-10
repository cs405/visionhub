"""DPN (Dual Path Networks) Implementation

Paper: Dual Path Networks
https://arxiv.org/abs/1707.01629
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DualPathBlock(nn.Module):
    """DPN Block"""
    def __init__(self, in_channels, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal'):
        super().__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc

        if block_type == 'proj':
            key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            key_stride = 2
            self.has_proj = True
        else:  # normal
            key_stride = 1
            self.has_proj = False

        if self.has_proj:
            self.c1x1_w = ConvBN(in_channels, num_1x1_c + 2 * inc, 1, key_stride)

        self.c1x1_a = ConvBN(in_channels, num_1x1_a, 1)
        self.c3x3_b = ConvBN(num_1x1_a, num_3x3_b, 3, key_stride, 1, groups=groups)
        self.c1x1_c = ConvBN(num_3x3_b, num_1x1_c + inc, 1)

    def forward(self, x):
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x

        if self.has_proj:
            x_s = self.c1x1_w(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            if isinstance(x, tuple):
                x_s1 = x[0]
                x_s2 = x[1]
            else:
                x_s1 = x
                x_s2 = 0

        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)

        out1 = x_in[:, :self.num_1x1_c, :, :]
        out2 = x_in[:, self.num_1x1_c:, :, :]

        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)

        return resid, dense


class DPN(nn.Module):
    """DPN Network"""
    def __init__(self, num_init_features=64, k_r=96, groups=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), num_classes=1000):
        super().__init__()

        blocks = {
            'c1': num_init_features,
            'c2': k_r,
            'c3': k_r,
            'c4': k_r,
            'c5': k_r
        }

        # Stem
        self.conv1 = ConvBN(3, num_init_features, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Build blocks
        self.features = nn.Sequential()

        # Block 1
        bw = 64
        inc = inc_sec[0]
        r = (k_r * bw) // 256
        self.features.add_module('conv2_1', DualPathBlock(blocks['c1'], r, r, bw, inc, groups, 'proj'))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            self.features.add_module(f'conv2_{i}', DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal'))
            in_chs += inc

        # Block 2
        bw = 128
        inc = inc_sec[1]
        r = (k_r * bw) // 256
        self.features.add_module('conv3_1', DualPathBlock(in_chs, r, r, bw, inc, groups, 'down'))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            self.features.add_module(f'conv3_{i}', DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal'))
            in_chs += inc

        # Block 3
        bw = 256
        inc = inc_sec[2]
        r = (k_r * bw) // 256
        self.features.add_module('conv4_1', DualPathBlock(in_chs, r, r, bw, inc, groups, 'down'))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            self.features.add_module(f'conv4_{i}', DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal'))
            in_chs += inc

        # Block 4
        bw = 512
        inc = inc_sec[3]
        r = (k_r * bw) // 256
        self.features.add_module('conv5_1', DualPathBlock(in_chs, r, r, bw, inc, groups, 'down'))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            self.features.add_module(f'conv5_{i}', DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal'))
            in_chs += inc

        self.final_channels = in_chs

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_chs, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        for name, module in self.features.named_children():
            x = module(x)

        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@register_backbone('dpn68')
class DPN68(DPN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_init_features=10, k_r=128, groups=32,
                        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
                        num_classes=num_classes)


@register_backbone('dpn92')
class DPN92(DPN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_init_features=64, k_r=96, groups=32,
                        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                        num_classes=num_classes)


@register_backbone('dpn98')
class DPN98(DPN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_init_features=96, k_r=160, groups=40,
                        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
                        num_classes=num_classes)


@register_backbone('dpn107')
class DPN107(DPN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_init_features=128, k_r=200, groups=50,
                        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
                        num_classes=num_classes)


@register_backbone('dpn131')
class DPN131(DPN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_init_features=128, k_r=160, groups=40,
                        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
                        num_classes=num_classes)

