"""EfficientNetV2: Improved EfficientNet

Implementation of EfficientNetV2 architecture
Paper: EfficientNetV2: Smaller Models and Faster Training
"""

import torch.nn as nn
import math


class ConvBNAct(nn.Module):
    """Conv + BN + Activation"""
    def __init__(self, in_c, out_c, k, s, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_c, se_c):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, se_c, 1),
            nn.SiLU(),
            nn.Conv2d(se_c, in_c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    """MBConv block with expansion"""
    def __init__(self, in_c, out_c, k, s, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (s == 1 and in_c == out_c)
        hidden_c = int(in_c * expand_ratio)

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_c, hidden_c, 1, 1, 0))

        # Depthwise
        layers.append(ConvBNAct(hidden_c, hidden_c, k, s, groups=hidden_c))

        # SE
        if se_ratio > 0:
            se_c = max(1, int(in_c * se_ratio))
            layers.append(SqueezeExcitation(hidden_c, se_c))

        # Projection
        layers.append(ConvBNAct(hidden_c, out_c, 1, 1, 0, act=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class FusedMBConv(nn.Module):
    """Fused MBConv block"""
    def __init__(self, in_c, out_c, k, s, expand_ratio):
        super().__init__()
        self.use_residual = (s == 1 and in_c == out_c)
        hidden_c = int(in_c * expand_ratio)

        layers = []
        # Fused expansion + depthwise
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_c, hidden_c, k, s))
        else:
            layers.append(ConvBNAct(in_c, hidden_c, k, s))

        # Projection
        if expand_ratio != 1:
            layers.append(ConvBNAct(hidden_c, out_c, 1, 1, 0, act=False))

        self.block = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class EfficientNetV2(nn.Module):
    """EfficientNetV2模型

    Args:
        width_mult: 宽度乘数
        depth_mult: 深度乘数
        num_classes: 分类数
        dropout: dropout率
    """

    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000, dropout=0.2):
        super().__init__()

        def round_channels(c):
            return int(math.ceil(c * width_mult / 8) * 8)

        def round_repeats(r):
            return int(math.ceil(r * depth_mult))

        # Stem
        self.stem = ConvBNAct(3, round_channels(24), 3, 2)

        # Stages configuration: [block_type, repeat, in_c, out_c, kernel, stride, expand]
        stages_config = [
            ['fused', 2, 24, 24, 3, 1, 1],
            ['fused', 4, 24, 48, 3, 2, 4],
            ['fused', 4, 48, 64, 3, 2, 4],
            ['mbconv', 6, 64, 128, 3, 2, 4],
            ['mbconv', 9, 128, 160, 3, 1, 6],
            ['mbconv', 15, 160, 256, 3, 2, 6],
        ]

        # Build stages
        stages = []
        for block_type, repeat, in_c, out_c, k, s, expand in stages_config:
            in_c = round_channels(in_c)
            out_c = round_channels(out_c)
            repeat = round_repeats(repeat)

            for i in range(repeat):
                stride = s if i == 0 else 1
                if block_type == 'fused':
                    stages.append(FusedMBConv(in_c, out_c, k, stride, expand))
                else:
                    stages.append(MBConv(in_c, out_c, k, stride, expand))
                in_c = out_c

        self.stages = nn.Sequential(*stages)

        # Head
        last_c = round_channels(1280)
        self.head = nn.Sequential(
            ConvBNAct(round_channels(256), last_c, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(last_c, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
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
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head[0](x)  # Conv
        x = self.head[1](x)  # AdaptiveAvgPool2d
        x = self.head[2](x)  # Flatten
        return x


def efficientnetv2_s(num_classes=1000):
    """EfficientNetV2-S"""
    return EfficientNetV2(width_mult=1.0, depth_mult=1.0, num_classes=num_classes)


def efficientnetv2_m(num_classes=1000):
    """EfficientNetV2-M"""
    return EfficientNetV2(width_mult=1.0, depth_mult=1.3, num_classes=num_classes)


def efficientnetv2_l(num_classes=1000):
    """EfficientNetV2-L"""
    return EfficientNetV2(width_mult=1.0, depth_mult=1.8, num_classes=num_classes)


__all__ = [
    'EfficientNetV2',
    'efficientnetv2_s',
    'efficientnetv2_m',
    'efficientnetv2_l',
]

