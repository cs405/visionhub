"""Face Recognition Backbones

人脸识别专用Backbone:
- AdaFace (Adaptive Face Recognition)
- MobileFaceNet (轻量级人脸识别)
- IR-Net (Improved ResNet for Face)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class IRBlock(nn.Module):
    """Improved ResNet Block for Face Recognition"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if use_se:
            self.se = SEModule(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class MobileFaceNet(nn.Module):
    """MobileFaceNet for efficient face recognition

    Paper: MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices
    """

    def __init__(self, num_classes=1000, embedding_size=512):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.conv2_dw = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.conv_23 = self._make_bottleneck(64, 64, 2, 5)
        self.conv_3 = self._make_bottleneck(64, 128, 2, 1)
        self.conv_34 = self._make_bottleneck(128, 128, 1, 6)
        self.conv_4 = self._make_bottleneck(128, 128, 2, 1)
        self.conv_45 = self._make_bottleneck(128, 128, 1, 2)

        self.conv_6_sep = nn.Sequential(
            nn.Conv2d(128, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )

        self.conv_6_dw = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512)
        )

        self.conv_6_flatten = Flatten()

        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

        self.fc = nn.Linear(embedding_size, num_classes)

        self._init_weights()

    def _make_bottleneck(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(self._bottleneck(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _bottleneck(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)

        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)

        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)

        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)

        # For classification
        out = self.fc(x)

        return out


class IRNet(nn.Module):
    """Improved ResNet for Face Recognition

    Variants: IR-50, IR-101, IR-152
    """

    def __init__(self, num_layers=50, num_classes=1000, use_se=True):
        super().__init__()

        if num_layers == 50:
            layers = [3, 4, 14, 3]
        elif num_layers == 100:
            layers = [3, 13, 30, 3]
        elif num_layers == 152:
            layers = [3, 8, 36, 3]
        else:
            raise ValueError(f"Unsupported num_layers: {num_layers}")

        self.use_se = use_se

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = Flatten()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(IRBlock(in_channels, out_channels, stride, self.use_se))
        for _ in range(1, num_blocks):
            layers.append(IRBlock(out_channels, out_channels, 1, self.use_se))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.bn3(x)

        # For classification
        out = self.classifier(x)

        return out


# Factory functions
def mobilefacenet(num_classes=1000, pretrained=False, **kwargs):
    """MobileFaceNet"""
    model = MobileFaceNet(num_classes=num_classes, **kwargs)
    return model


def ir_net_50(num_classes=1000, pretrained=False, **kwargs):
    """IR-Net-50"""
    model = IRNet(num_layers=50, num_classes=num_classes, **kwargs)
    return model


def ir_net_100(num_classes=1000, pretrained=False, **kwargs):
    """IR-Net-100"""
    model = IRNet(num_layers=100, num_classes=num_classes, **kwargs)
    return model


def ir_net_152(num_classes=1000, pretrained=False, **kwargs):
    """IR-Net-152"""
    model = IRNet(num_layers=152, num_classes=num_classes, **kwargs)
    return model

