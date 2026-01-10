"""ResNet_vd (ResNet variant with modifications for better performance)

The _vd suffix means:
- Replace 7x7 conv stride 2 with three 3x3 convs
- Add stride 2 to 3x3 conv in bottleneck instead of 1x1 conv
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleneckBlock_vd(nn.Module):
    """ResNet_vd bottleneck block with stride on 3x3 conv"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, 1, 1)
        self.conv2 = ConvBNLayer(out_channels, out_channels, 3, stride)
        self.conv3 = ConvBNLayer(out_channels, out_channels * self.expansion, 1, 1, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock_vd(nn.Module):
    """ResNet_vd basic block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBNLayer(out_channels, out_channels, 3, 1, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet_vd(nn.Module):
    def __init__(self, layers, block_type='bottleneck', num_classes=1000):
        super().__init__()
        self.in_channels = 64

        block = BottleneckBlock_vd if block_type == 'bottleneck' else BasicBlock_vd

        # Stem: three 3x3 convs instead of one 7x7
        self.conv1_1 = ConvBNLayer(3, 32, 3, 2)
        self.conv1_2 = ConvBNLayer(32, 32, 3, 1)
        self.conv1_3 = ConvBNLayer(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Use AvgPool2d + 1x1 conv for downsampling when stride=2
            if stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


@register_backbone('resnet18_vd')
class ResNet18_vd(ResNet_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([2, 2, 2, 2], 'basic', num_classes)


@register_backbone('resnet34_vd')
class ResNet34_vd(ResNet_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 4, 6, 3], 'basic', num_classes)


@register_backbone('resnet50_vd')
class ResNet50_vd(ResNet_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 4, 6, 3], 'bottleneck', num_classes)


@register_backbone('resnet101_vd')
class ResNet101_vd(ResNet_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 4, 23, 3], 'bottleneck', num_classes)


@register_backbone('resnet152_vd')
class ResNet152_vd(ResNet_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], 'bottleneck', num_classes)


@register_backbone('resnet200_vd')
class ResNet200_vd(ResNet_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 12, 48, 3], 'bottleneck', num_classes)

