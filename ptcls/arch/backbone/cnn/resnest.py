"""ResNeSt: Split-Attention Networks

完整迁移自PaddleClas ResNeSt
Paper: ResNeSt: Split-Attention Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class rSoftmax(nn.Module):
    """Radix Softmax"""
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttention(nn.Module):
    """Split Attention模块"""
    def __init__(self, channels, radix=2, cardinality=1, reduction=4):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels
        inter_channels = max(channels * radix // reduction, 32)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=cardinality)
        self.rsoftmax = rSoftmax(radix, cardinality)

    def forward(self, x):
        batch, rchannel = x.size(0), x.size(1)

        splited = None
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x

        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1 and splited is not None:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for att, split in zip(attens, splited)])
        else:
            out = atten * x

        return out


class SplitConv(nn.Module):
    """Split Convolution"""
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, radix=2, reduction_factor=4):
        super().__init__()
        self.radix = radix

        self.conv = nn.Conv2d(
            in_channels, channels * radix, kernel_size,
            stride, padding, dilation, groups=groups * radix, bias=False
        )
        self.bn = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU(inplace=True)

        self.split_attn = SplitAttention(
            channels, radix=radix, cardinality=groups, reduction=reduction_factor
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.split_attn(x)
        return x


class ResNeStBottleneck(nn.Module):
    """ResNeSt Bottleneck"""
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 radix=2, cardinality=1, base_width=64, avd=False,
                 avd_first=False, is_first=False):
        super().__init__()

        group_width = int(channels * (base_width / 64.)) * cardinality

        self.conv1 = nn.Conv2d(in_channels, group_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.conv2 = SplitConv(
            group_width, group_width, 3, stride=stride, padding=1,
            groups=cardinality, radix=radix, reduction_factor=4
        )

        self.conv3 = nn.Conv2d(group_width, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNeSt(nn.Module):
    """ResNeSt模型

    Args:
        block: Bottleneck类
        layers: 每个stage的block数量
        radix: split attention的radix
        cardinality: 分组数
        base_width: 基础宽度
        num_classes: 分类数
    """

    def __init__(self, block, layers, radix=2, cardinality=1, base_width=64,
                 deep_stem=True, avg_down=True, avd=True, avd_first=False,
                 num_classes=1000):
        super().__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        self.inplanes = 64

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample,
                  radix=self.radix, cardinality=self.cardinality,
                  base_width=self.base_width, avd=self.avd,
                  avd_first=self.avd_first, is_first=is_first)
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,
                      radix=self.radix, cardinality=self.cardinality,
                      base_width=self.base_width, avd=self.avd,
                      avd_first=self.avd_first)
            )

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

    def forward_features(self, x):
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
        return x


def resnest50(num_classes=1000):
    """ResNeSt-50"""
    return ResNeSt(ResNeStBottleneck, [3, 4, 6, 3],
                   radix=2, cardinality=1, base_width=64,
                   num_classes=num_classes)


def resnest101(num_classes=1000):
    """ResNeSt-101"""
    return ResNeSt(ResNeStBottleneck, [3, 4, 23, 3],
                   radix=2, cardinality=1, base_width=64,
                   num_classes=num_classes)


def resnest200(num_classes=1000):
    """ResNeSt-200"""
    return ResNeSt(ResNeStBottleneck, [3, 24, 36, 3],
                   radix=2, cardinality=1, base_width=64,
                   num_classes=num_classes)


__all__ = [
    'ResNeSt',
    'resnest50',
    'resnest101',
    'resnest200',
]

