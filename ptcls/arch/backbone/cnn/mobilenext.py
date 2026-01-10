"""MobileNeXt: Rethinking Bottleneck Structure for Efficient Mobile Network Design

完整迁移自PaddleClas MobileNeXt
Paper: Rethinking Bottleneck Structure for Efficient Mobile Network Design
"""

import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class SandglassBlock(nn.Module):
    """沙漏块 - MobileNeXt的核心模块"""

    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw-linear
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.identity_tensor_channels == x.size(1):
                identity = x
            else:
                identity = x[:, :self.identity_tensor_channels, :, :]

            return identity + out[:, :self.identity_tensor_channels, :, :]
        else:
            return out


class MobileNeXt(nn.Module):
    """MobileNeXt模型

    Args:
        num_classes: 分类数
        width_mult: 宽度乘数
        identity_tensor_multiplier: identity tensor乘数
        sand_glass_setting: 沙漏块配置
    """

    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        identity_tensor_multiplier=1.0,
        sand_glass_setting=None
    ):
        super().__init__()

        block = SandglassBlock
        input_channel = 32
        last_channel = 1280

        if sand_glass_setting is None:
            sand_glass_setting = [
                # t, c,  b, s
                [2, 96,  1, 2],
                [6, 144, 1, 1],
                [6, 192, 3, 2],
                [6, 288, 3, 2],
                [6, 384, 4, 1],
                [6, 576, 4, 2],
                [6, 960, 3, 1],
                [6, 1280, 1, 1],
            ]

        # 第一层
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        # 构建沙漏块
        for t, c, b, s in sand_glass_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, t, identity_tensor_multiplier)
                )
                input_channel = output_channel

        self.features = nn.Sequential(*features)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def mobilenext(num_classes=1000, width_mult=1.0):
    """MobileNeXt"""
    return MobileNeXt(num_classes=num_classes, width_mult=width_mult)


__all__ = ['MobileNeXt', 'mobilenext']

