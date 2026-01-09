"""EfficientNet implementation in PyTorch

Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/abs/1905.11946
"""

import torch
import torch.nn as nn
import math
from ..registry import register_backbone


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            SEBlock(hidden_dim, se_ratio),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, width_mult, depth_mult, dropout_rate, num_classes=1000):
        super().__init__()

        # [expand_ratio, channels, layers, stride, kernel_size]
        settings = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3]
        ]

        out_channels = _make_divisible(32 * width_mult, 8)
        self.features = [
            nn.Sequential(
                nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                Swish()
            )
        ]

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _make_divisible(c * width_mult, 8)
            layers_num = int(math.ceil(n * depth_mult))
            for i in range(layers_num):
                stride = s if i == 0 else 1
                self.features.append(MBConv(in_channels, out_channels, t, k, stride))
                in_channels = out_channels

        last_channels = _make_divisible(1280 * width_mult, 8)
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_channels, last_channels, 1, bias=False),
                nn.BatchNorm2d(last_channels),
                Swish()
            )
        )

        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

        self._initialize_weights()

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


# EfficientNet-B0 to B7
@register_backbone('efficientnet_b0')
class EfficientNetB0(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.0, 1.0, 0.2, num_classes)
        if pretrained:
            try:
                from torchvision.models import efficientnet_b0
                state_dict = efficientnet_b0(pretrained=True).state_dict()
                self.load_state_dict({k: v for k, v in state_dict.items() if k in self.state_dict() and v.shape == self.state_dict()[k].shape}, strict=False)
            except: pass


@register_backbone('efficientnet_b1')
class EfficientNetB1(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.0, 1.1, 0.2, num_classes)


@register_backbone('efficientnet_b2')
class EfficientNetB2(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.1, 1.2, 0.3, num_classes)


@register_backbone('efficientnet_b3')
class EfficientNetB3(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.2, 1.4, 0.3, num_classes)


@register_backbone('efficientnet_b4')
class EfficientNetB4(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.4, 1.8, 0.4, num_classes)


@register_backbone('efficientnet_b5')
class EfficientNetB5(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.6, 2.2, 0.4, num_classes)


@register_backbone('efficientnet_b6')
class EfficientNetB6(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(1.8, 2.6, 0.5, num_classes)


@register_backbone('efficientnet_b7')
class EfficientNetB7(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(2.0, 3.1, 0.5, num_classes)

