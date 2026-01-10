"""ResNet_VC: ResNet with Variant C improvements

ResNet Variant C with improved downsampling strategy
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """ResNet Bottleneck with Variant C improvements"""
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None, groups=1, base_width=64):
        super().__init__()
        width = int(out_c * (base_width / 64.)) * groups

        # 1x1
        self.conv1 = nn.Conv2d(in_c, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1
        self.conv3 = nn.Conv2d(width, out_c * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_VC(nn.Module):
    """ResNet with Variant C improvements

    改进的下采样策略：
    - Variant C: 使用3个3x3卷积替代7x7卷积
    - 更好的跨度策略

    Args:
        block: Bottleneck类
        layers: 每个stage的层数
        num_classes: 分类数
        groups: 分组数
        width_per_group: 每组宽度
    """

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64):
        super().__init__()

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        # Variant C: 3个3x3卷积替代7x7
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
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
            # Improved downsampling
            downsample = nn.Sequential(
                nn.AvgPool2d(2, 2) if stride > 1 else nn.Identity(),
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, 1, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width)
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width)
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


def resnet50_vc(num_classes=1000):
    """ResNet-50 with Variant C"""
    return ResNet_VC(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101_vc(num_classes=1000):
    """ResNet-101 with Variant C"""
    return ResNet_VC(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


__all__ = ['ResNet_VC', 'resnet50_vc', 'resnet101_vc']

