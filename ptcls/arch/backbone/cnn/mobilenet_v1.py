"""MobileNetV1 Implementation

Paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                     groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DepthwiseSeparable(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = ConvBNReLU(in_channels, in_channels, 3, stride, groups=in_channels)
        self.pointwise = ConvBNReLU(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()

        # First conv layer
        input_channel = _make_divisible(32 * width_mult, 8)
        self.features = nn.Sequential(
            ConvBNReLU(3, input_channel, 3, 2),
        )

        # Depthwise separable conv layers
        # [out_channels, stride]
        settings = [
            [64, 1],
            [128, 2],
            [128, 1],
            [256, 2],
            [256, 1],
            [512, 2],
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [1024, 2],
            [1024, 1],
        ]

        for out_ch, stride in settings:
            output_channel = _make_divisible(out_ch * width_mult, 8)
            self.features.append(DepthwiseSeparable(input_channel, output_channel, stride))
            input_channel = output_channel

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


@register_backbone('mobilenet_v1_x0_25')
class MobileNetV1_x0_25(MobileNetV1):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=0.25)


@register_backbone('mobilenet_v1_x0_5')
class MobileNetV1_x0_5(MobileNetV1):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=0.5)


@register_backbone('mobilenet_v1_x0_75')
class MobileNetV1_x0_75(MobileNetV1):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=0.75)


@register_backbone('mobilenet_v1')
class MobileNetV1_x1_0(MobileNetV1):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=1.0)

