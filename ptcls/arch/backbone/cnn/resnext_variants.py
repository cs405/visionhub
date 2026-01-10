"""ResNeXt Extended Variants

Additional ResNeXt models including 152-layer variants
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


class ResNeXtBottleneck(nn.Module):
    """ResNeXt Bottleneck Block with grouped convolutions"""
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, downsample=None):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * cardinality

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    """ResNeXt backbone"""
    def __init__(self, layers, cardinality=32, base_width=4, num_classes=1000):
        super().__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * ResNeXtBottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * ResNeXtBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * ResNeXtBottleneck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * ResNeXtBottleneck.expansion)
            )

        layers = []
        layers.append(ResNeXtBottleneck(self.in_channels, out_channels, stride,
                                        self.cardinality, self.base_width, downsample))
        self.in_channels = out_channels * ResNeXtBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(ResNeXtBottleneck(self.in_channels, out_channels,
                                           cardinality=self.cardinality, base_width=self.base_width))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@register_backbone('resnext152_32x4d')
class ResNeXt152_32x4d(ResNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], cardinality=32, base_width=4, num_classes=num_classes)


@register_backbone('resnext152_64x4d')
class ResNeXt152_64x4d(ResNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], cardinality=64, base_width=4, num_classes=num_classes)

