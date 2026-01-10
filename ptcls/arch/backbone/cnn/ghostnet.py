"""Small Niche Backbone Networks

小众骨干网络集合
"""

import torch
import torch.nn as nn


class GhostModule(nn.Module):
    """Ghost模块 - 生成更多特征图"""
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_c = out_c
        init_channels = out_c // ratio
        new_channels = out_c - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_c, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_c, :, :]


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck块"""
    def __init__(self, in_c, mid_c, out_c, dw_kernel_size=3, stride=1, use_se=False):
        super().__init__()
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_c, mid_c, relu=True)

        # Depth-wise conv
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_c, mid_c, dw_kernel_size, stride=stride,
                                    padding=(dw_kernel_size-1)//2, groups=mid_c, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_c)

        # SE
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_c, mid_c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c // 4, mid_c, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_c, out_c, relu=False)

        # Shortcut
        if in_c == out_c and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, in_c, dw_kernel_size, stride=stride,
                         padding=(dw_kernel_size-1)//2, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x = x * self.se(x)

        x = self.ghost2(x)

        return x + self.shortcut(residual)


class GhostNet(nn.Module):
    """GhostNet - 轻量级网络"""

    def __init__(self, num_classes=1000, width=1.0):
        super().__init__()

        # Stem
        output_channel = int(16 * width)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # Building blocks
        stages = []
        block = GhostBottleneck

        # Stage configurations
        cfgs = [
            # k, t, c, SE, s
            [[3,  16,  16, 0, 1]],
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            [[5,  72,  40, 1, 2]],
            [[5, 120,  40, 1, 1]],
            [[3, 240,  80, 0, 2]],
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 1, 1],
             [3, 672, 112, 1, 1]],
            [[5, 672, 160, 1, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 1, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 1, 1]]
        ]

        for cfg in cfgs:
            layers = []
            for k, exp_size, c, use_se, s in cfg:
                output_channel = int(c * width)
                hidden_channel = int(exp_size * width)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = int(960 * width)
        stages.append(nn.Sequential(GhostModule(input_channel, output_channel, relu=True)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(1280, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def ghostnet_x1_0(num_classes=1000):
    """GhostNet 1.0x"""
    return GhostNet(num_classes=num_classes, width=1.0)


__all__ = ['GhostNet', 'ghostnet_x1_0', 'GhostModule', 'GhostBottleneck']

