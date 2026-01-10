"""ReXNet and StarNet Implementations

ReXNet: Rethinking Channel Dimensions for Efficient Model Design
StarNet: Star Shaped Networks
"""

import torch
import torch.nn as nn
from ..registry import register_backbone
import math


# ============ ReXNet ============

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
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


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_ratio, use_se=True):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expansion_ratio))

        layers = []
        if expansion_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            SEModule(hidden_dim) if use_se else nn.Identity(),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ReXNet(nn.Module):
    """ReXNet - Rethinking Channel Dimensions"""
    def __init__(self, width_mult=1.0, num_classes=1000):
        super().__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        input_channel = 32
        final_channel = 180

        input_channel = _make_divisible(input_channel * width_mult)
        final_channel = _make_divisible(final_channel * width_mult)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        # Building inverted residual blocks
        for i in range(len(layers)):
            output_channel = _make_divisible(16 * (i + 2) * width_mult)
            for j in range(layers[i]):
                stride = strides[i] if j == 0 else 1
                features.append(LinearBottleneck(input_channel, output_channel, stride,
                                                expansion_ratio=6, use_se=use_ses[i]))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, final_channel, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_channel, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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


@register_backbone('rexnet_1_0')
class ReXNet_1_0(ReXNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.0, num_classes=num_classes)


@register_backbone('rexnet_1_3')
class ReXNet_1_3(ReXNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.3, num_classes=num_classes)


@register_backbone('rexnet_1_5')
class ReXNet_1_5(ReXNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.5, num_classes=num_classes)


@register_backbone('rexnet_2_0')
class ReXNet_2_0(ReXNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=2.0, num_classes=num_classes)


@register_backbone('rexnet_3_0')
class ReXNet_3_0(ReXNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=3.0, num_classes=num_classes)


# ============ StarNet ============

class StarBlock(nn.Module):
    """Star-shaped block"""
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = nn.Conv2d(mlp_ratio * dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.act = nn.ReLU6()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.drop(x)
        x = self.g(x)
        x = self.dwconv2(x)
        x = input + x
        return x


class StarNet(nn.Module):
    """Star-shaped Network"""
    def __init__(self, depths, dims, mlp_ratios, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 4, 4),
            nn.BatchNorm2d(dims[0])
        )

        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[StarBlock(dims[i], mlp_ratios[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

            if i < len(depths) - 1:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], 2, 2),
                    nn.BatchNorm2d(dims[i+1])
                ))

        self.norm = nn.BatchNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


@register_backbone('starnet_s1')
class StarNet_S1(StarNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[2, 2, 8, 3], dims=[24, 48, 96, 192],
                        mlp_ratios=[4, 4, 4, 4], num_classes=num_classes)


@register_backbone('starnet_s2')
class StarNet_S2(StarNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[2, 2, 8, 3], dims=[32, 64, 128, 256],
                        mlp_ratios=[4, 4, 4, 4], num_classes=num_classes)


@register_backbone('starnet_s3')
class StarNet_S3(StarNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[3, 3, 12, 5], dims=[32, 64, 128, 256],
                        mlp_ratios=[4, 4, 4, 4], num_classes=num_classes)


@register_backbone('starnet_s4')
class StarNet_S4(StarNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[3, 3, 12, 5], dims=[48, 96, 192, 384],
                        mlp_ratios=[3, 3, 3, 3], num_classes=num_classes)

