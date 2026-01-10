"""RedNet: Residual Encoder-Decoder Network

完整迁移自PaddleClas RedNet
Paper: Involution: Inverting the Inherence of Convolution for Visual Recognition
"""

import torch
import torch.nn as nn


class Involution(nn.Module):
    """Involution: 反卷积操作"""

    def __init__(self, channels, kernel_size=3, stride=1, group_channels=16, reduction_ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels

        self.conv1 = nn.Conv2d(
            channels,
            channels // reduction_ratio,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels // reduction_ratio,
            kernel_size * kernel_size * self.groups,
            kernel_size=1
        )

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.relu(self.bn(self.conv1(x))))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size * self.kernel_size, h, w).unsqueeze(2)

        out = self.unfold(x)
        out = out.view(b, self.groups, self.group_channels, self.kernel_size * self.kernel_size, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)

        return out


class BottleneckBlock(nn.Module):
    """RedNet瓶颈块"""
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, use_involution=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        if use_involution:
            self.conv2 = Involution(channels, kernel_size=3, stride=stride)
        else:
            self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class RedNet(nn.Module):
    """RedNet模型

    使用Involution替代部分卷积层

    Args:
        block: 块类型
        layers: 每个stage的层数
        num_classes: 分类数
        involution_layers: 使用involution的层
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        involution_layers=None
    ):
        super().__init__()

        if involution_layers is None:
            involution_layers = [0, 0, 0, 0]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], involution_layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], involution_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], involution_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], involution_layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, involution_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        use_involution = involution_blocks > 0
        layers.append(
            block(self.inplanes, planes, stride, downsample, use_involution)
        )
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            use_involution = i < involution_blocks
            layers.append(block(self.inplanes, planes, use_involution=use_involution))

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


def rednet26(num_classes=1000):
    """RedNet-26"""
    return RedNet(BottleneckBlock, [1, 2, 4, 1], num_classes=num_classes,
                  involution_layers=[0, 1, 2, 0])


def rednet38(num_classes=1000):
    """RedNet-38"""
    return RedNet(BottleneckBlock, [2, 3, 5, 2], num_classes=num_classes,
                  involution_layers=[0, 2, 3, 0])


def rednet50(num_classes=1000):
    """RedNet-50"""
    return RedNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes,
                  involution_layers=[0, 2, 4, 0])


def rednet101(num_classes=1000):
    """RedNet-101"""
    return RedNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes,
                  involution_layers=[0, 2, 10, 0])


__all__ = [
    'RedNet',
    'rednet26',
    'rednet38',
    'rednet50',
    'rednet101',
]

