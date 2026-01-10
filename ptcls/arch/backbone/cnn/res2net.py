"""Res2Net: A New Multi-scale Backbone Architecture

Paper: Res2Net: A New Multi-scale Backbone Architecture
"""

import torch
import torch.nn as nn


class Bottle2neck(nn.Module):
    """Res2Net Bottleneck

    多尺度特征提取的瓶颈块
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4):
        super().__init__()

        width = int(planes * (baseWidth / 64.))
        self.conv1 = nn.Conv2d(inplanes, width * scale, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        # scale个并行的3x3卷积
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, 3, stride, 1, bias=False))
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 分割成scale份
        spx = torch.split(out, self.width, 1)
        sp = spx[0]

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    """Res2Net模型

    Args:
        block: Bottle2neck类
        layers: 每个stage的block数量
        baseWidth: 基础宽度
        scale: 尺度数
        num_classes: 分类数
    """

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample,
                  baseWidth=self.baseWidth, scale=self.scale)
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,
                      baseWidth=self.baseWidth, scale=self.scale)
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


def res2net50(num_classes=1000, **kwargs):
    """Res2Net-50"""
    return Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4,
                   num_classes=num_classes)


def res2net101(num_classes=1000, **kwargs):
    """Res2Net-101"""
    return Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4,
                   num_classes=num_classes)


def res2net50_26w_4s(num_classes=1000):
    """Res2Net-50 26w-4s"""
    return Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4,
                   num_classes=num_classes)


def res2net50_26w_8s(num_classes=1000):
    """Res2Net-50 26w-8s"""
    return Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8,
                   num_classes=num_classes)


__all__ = [
    'Res2Net',
    'res2net50',
    'res2net101',
    'res2net50_26w_4s',
    'res2net50_26w_8s',
]

