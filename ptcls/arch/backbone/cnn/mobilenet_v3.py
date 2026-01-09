"""MobileNetV3 implementation in PyTorch

Paper: Searching for MobileNetV3
https://arxiv.org/abs/1905.02244
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register_backbone


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if hidden_dim != inp:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SELayer(hidden_dim) if use_se else nn.Identity(),
            nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@register_backbone('mobilenet_v3_small')
class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, pretrained=False):
        super(MobileNetV3Small, self).__init__()
        input_channel = 16
        last_channel = 1024

        # bneck config: [k, exp, c, se, nl, s]
        # k: kernel_size, exp: expansion, c: output_channel, se: use_se, nl: use_hardswish, s: stride
        inverted_residual_setting = [
            [3, 16, 16, True, False, 2],
            [3, 72, 24, False, False, 2],
            [3, 88, 24, False, False, 1],
            [5, 96, 40, True, True, 2],
            [5, 240, 40, True, True, 1],
            [5, 240, 40, True, True, 1],
            [5, 120, 48, True, True, 1],
            [5, 144, 48, True, True, 1],
            [5, 288, 96, True, True, 2],
            [5, 576, 96, True, True, 1],
            [5, 576, 96, True, True, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, 8)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.Hardswish(inplace=True)
        ]

        for k, exp, c, se, nl, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            features.append(InvertedResidual(input_channel, exp_channel, output_channel, k, s, se, nl))
            input_channel = output_channel

        features.append(nn.Conv2d(input_channel, exp_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(exp_channel))
        features.append(nn.Hardswish(inplace=True))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(exp_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._initialize_weights()
        if pretrained:
            self._load_pretrained()

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

    def _load_pretrained(self):
        try:
            from torchvision.models import mobilenet_v3_small
            pretrained_dict = mobilenet_v3_small(pretrained=True).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        except Exception as e:
            print(f"[WARNING] Failed to load pretrained weights: {e}")


@register_backbone('mobilenet_v3_large')
class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, pretrained=False):
        super(MobileNetV3Large, self).__init__()
        input_channel = 16
        last_channel = 1280

        inverted_residual_setting = [
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, 8)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.Hardswish(inplace=True)
        ]

        for k, exp, c, se, nl, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            features.append(InvertedResidual(input_channel, exp_channel, output_channel, k, s, se, nl))
            input_channel = output_channel

        features.append(nn.Conv2d(input_channel, exp_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(exp_channel))
        features.append(nn.Hardswish(inplace=True))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(exp_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        self._initialize_weights()
        if pretrained:
            self._load_pretrained()

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

    def _load_pretrained(self):
        try:
            from torchvision.models import mobilenet_v3_large
            pretrained_dict = mobilenet_v3_large(pretrained=True).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        except Exception as e:
            print(f"[WARNING] Failed to load pretrained weights: {e}")

