"""RegNet: Designing Network Design Spaces

Implementation of RegNet architecture
Paper: Designing Network Design Spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SqueezeExcitation(nn.Module):
    """SE模块"""
    def __init__(self, in_channels, se_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        return x * scale


class BottleneckTransform(nn.Module):
    """RegNet Bottleneck"""
    def __init__(
        self,
        width_in,
        width_out,
        stride,
        group_width,
        bottleneck_multiplier,
        se_ratio=None
    ):
        super().__init__()

        # 计算中间宽度
        width_intermediate = int(width_out * bottleneck_multiplier)
        groups = width_intermediate // group_width

        # 1x1
        self.conv1 = nn.Conv2d(width_in, width_intermediate, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width_intermediate)

        # 3x3 grouped
        self.conv2 = nn.Conv2d(
            width_intermediate, width_intermediate, 3,
            stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width_intermediate)

        # SE
        if se_ratio is not None:
            se_channels = int(width_in * se_ratio)
            self.se = SqueezeExcitation(width_intermediate, se_channels)
        else:
            self.se = None

        # 1x1
        self.conv3 = nn.Conv2d(width_intermediate, width_out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(width_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        if self.se is not None:
            x = self.se(x)

        x = self.conv3(x)
        x = self.bn3(x)
        return x


class ResBottleneckBlock(nn.Module):
    """RegNet残差瓶颈块"""
    def __init__(
        self,
        width_in,
        width_out,
        stride,
        group_width,
        bottleneck_multiplier,
        se_ratio=None
    ):
        super().__init__()

        # 主路径
        self.transform = BottleneckTransform(
            width_in, width_out, stride,
            group_width, bottleneck_multiplier, se_ratio
        )

        # 快捷连接
        if (width_in != width_out) or (stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(width_in, width_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(width_out)
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.transform(x)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        x = x + identity
        x = self.relu(x)
        return x


class AnyStage(nn.Module):
    """RegNet的一个stage"""
    def __init__(
        self,
        width_in,
        width_out,
        stride,
        depth,
        group_width,
        bottleneck_multiplier,
        se_ratio=None
    ):
        super().__init__()

        blocks = []
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_width_in = width_in if i == 0 else width_out

            blocks.append(
                ResBottleneckBlock(
                    block_width_in, width_out, block_stride,
                    group_width, bottleneck_multiplier, se_ratio
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class RegNet(nn.Module):
    """RegNet模型

    Args:
        depths: 每个stage的深度
        widths: 每个stage的宽度
        group_width: 分组卷积的组宽度
        bottleneck_multiplier: 瓶颈乘数
        se_ratio: SE比率
        num_classes: 分类数
    """

    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_width: int = 8,
        bottleneck_multiplier: float = 1.0,
        se_ratio: float = None,
        num_classes: int = 1000
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Stages
        stages = []
        width_in = 32

        for i, (depth, width) in enumerate(zip(depths, widths)):
            stride = 2 if i > 0 else 1
            stages.append(
                AnyStage(
                    width_in, width, stride, depth,
                    group_width, bottleneck_multiplier, se_ratio
                )
            )
            width_in = width

        self.stages = nn.Sequential(*stages)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width_in, num_classes)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def regnet_y_400mf(num_classes=1000):
    """RegNetY-400MF"""
    return RegNet(
        depths=[2, 5, 11, 1],
        widths=[48, 104, 208, 440],
        group_width=8,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
        num_classes=num_classes
    )


def regnet_y_800mf(num_classes=1000):
    """RegNetY-800MF"""
    return RegNet(
        depths=[2, 6, 14, 2],
        widths=[64, 128, 256, 512],
        group_width=16,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
        num_classes=num_classes
    )


def regnet_y_1_6gf(num_classes=1000):
    """RegNetY-1.6GF"""
    return RegNet(
        depths=[2, 6, 17, 2],
        widths=[48, 120, 336, 888],
        group_width=24,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
        num_classes=num_classes
    )


def regnet_y_3_2gf(num_classes=1000):
    """RegNetY-3.2GF"""
    return RegNet(
        depths=[2, 8, 20, 1],
        widths=[72, 216, 576, 1512],
        group_width=24,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
        num_classes=num_classes
    )


__all__ = [
    'RegNet',
    'regnet_y_400mf',
    'regnet_y_800mf',
    'regnet_y_1_6gf',
    'regnet_y_3_2gf',
]

