"""Final Remaining Models - ResNeXt_vd and SE variants

ResNeXt152_vd, SE_ResNeXt, SE_HRNet
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


# ========== SEModule ==========

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ========== ResNeXt_vd Bottleneck ==========

class ResNeXtBottleneck_vd(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, downsample=None):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * cardinality

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # ResNet-D improvement: avgpool before conv with stride
        if stride == 2:
            self.avd_layer = nn.AvgPool2d(3, 2, padding=1)
            conv_stride = 1
        else:
            self.avd_layer = None
            conv_stride = stride

        self.conv2 = nn.Conv2d(width, width, 3, conv_stride, 1, groups=cardinality, bias=False)
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

        if self.avd_layer is not None:
            out = self.avd_layer(out)

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


class ResNeXt_vd(nn.Module):
    def __init__(self, layers, cardinality=32, base_width=4, num_classes=1000):
        super().__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.in_channels = 64

        # ResNet-D improvement: 3x3 convs in stem
        self.conv1_1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * ResNeXtBottleneck_vd.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * ResNeXtBottleneck_vd.expansion:
            # ResNet-D: use avgpool for downsampling
            downsample = nn.Sequential()
            if stride == 2:
                downsample.add_module('avd', nn.AvgPool2d(2, 2, ceil_mode=True))
            downsample.add_module('conv', nn.Conv2d(self.in_channels, out_channels * ResNeXtBottleneck_vd.expansion,
                                                   1, 1, bias=False))
            downsample.add_module('bn', nn.BatchNorm2d(out_channels * ResNeXtBottleneck_vd.expansion))

        layers = []
        layers.append(ResNeXtBottleneck_vd(self.in_channels, out_channels, stride,
                                           self.cardinality, self.base_width, downsample))
        self.in_channels = out_channels * ResNeXtBottleneck_vd.expansion
        for _ in range(1, blocks):
            layers.append(ResNeXtBottleneck_vd(self.in_channels, out_channels,
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
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
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


@register_backbone('resnext152_vd_32x4d')
class ResNeXt152_vd_32x4d(ResNeXt_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], cardinality=32, base_width=4, num_classes=num_classes)


@register_backbone('resnext152_vd_64x4d')
class ResNeXt152_vd_64x4d(ResNeXt_vd):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], cardinality=64, base_width=4, num_classes=num_classes)


# ========== SE_ResNeXt ==========

class SEResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4,
                 downsample=None, reduction=16):
        super().__init__()
        width = int(out_channels * (base_width / 64)) * cardinality

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(out_channels * self.expansion, reduction)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SE_ResNeXt(nn.Module):
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
        self.fc = nn.Linear(512 * SEResNeXtBottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * SEResNeXtBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * SEResNeXtBottleneck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * SEResNeXtBottleneck.expansion)
            )

        layers = []
        layers.append(SEResNeXtBottleneck(self.in_channels, out_channels, stride,
                                          self.cardinality, self.base_width, downsample))
        self.in_channels = out_channels * SEResNeXtBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(SEResNeXtBottleneck(self.in_channels, out_channels,
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


@register_backbone('se_resnext152_64x4d')
class SE_ResNeXt152_64x4d(SE_ResNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], cardinality=64, base_width=4, num_classes=num_classes)

