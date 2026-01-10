"""PeleeNet: An efficient DenseNet architecture for mobile devices

Paper: Pelee: A Real-Time Object Detection System on Mobile Devices
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k, s=1, p=0, groups=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


class StemBlock(nn.Module):
    """PeleeNet的Stem块"""
    def __init__(self, in_c=3, out_c=32):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c, 3, 2, 1)

        self.branch1a = ConvBNReLU(out_c, out_c // 2, 1, 1, 0)
        self.branch1b = ConvBNReLU(out_c // 2, out_c, 3, 2, 1)

        self.branch2 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBNReLU(out_c * 2, out_c, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.branch1a(x)
        x1 = self.branch1b(x1)

        x2 = self.branch2(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x


class DenseBlock(nn.Module):
    """密集块"""
    def __init__(self, in_c, growth_rate, num_layers, bottleneck_width=4):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(
                self._make_layer(in_c + i * growth_rate, growth_rate, bottleneck_width)
            )

    def _make_layer(self, in_c, growth_rate, bottleneck_width):
        inter_c = growth_rate * bottleneck_width
        return nn.Sequential(
            ConvBNReLU(in_c, inter_c, 1, 1, 0),
            ConvBNReLU(inter_c, growth_rate, 3, 1, 1)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class TransitionBlock(nn.Module):
    """过渡块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBNReLU(in_c, out_c, 1, 1, 0)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PeleeNet(nn.Module):
    """PeleeNet模型

    专为移动设备设计的高效DenseNet架构

    Args:
        num_classes: 分类数
        growth_rate: 增长率
        block_config: 每个dense block的层数
        num_init_features: 初始特征数
    """

    def __init__(
        self,
        num_classes=1000,
        growth_rate=32,
        block_config=None,
        num_init_features=32
    ):
        super().__init__()

        if block_config is None:
            block_config = [3, 4, 8, 6]

        # Stem
        self.stem = StemBlock(3, num_init_features)

        # Dense blocks
        num_features = num_init_features
        self.features = nn.Sequential()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_features,
                growth_rate,
                num_layers,
                bottleneck_width=4
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionBlock(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # Final bn
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def peleenet(num_classes=1000):
    """PeleeNet"""
    return PeleeNet(
        num_classes=num_classes,
        growth_rate=32,
        block_config=[3, 4, 8, 6]
    )


__all__ = ['PeleeNet', 'peleenet']

