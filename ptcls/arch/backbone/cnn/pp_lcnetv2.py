"""PP-LCNetV2: Lightweight CPU Network V2

Second generation lightweight network optimized for CPU devices
"""

import torch
import torch.nn as nn


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp(x + 3, 0, 6) / 6


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3, 0, 6) / 6


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)
        self.hardsigmoid = HardSigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return identity * x


class ConvBNLayer(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=None, groups=1, act='relu'):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'hardswish':
            self.act = HardSwish()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepDepthwiseSeparable(nn.Module):
    """重参数化深度可分离卷积"""
    def __init__(self, in_c, out_c, stride, dw_size=3, use_se=False, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.use_se = use_se

        if deploy:
            # 部署模式
            self.dw_conv = nn.Conv2d(in_c, in_c, dw_size, stride, (dw_size-1)//2,
                                     groups=in_c, bias=True)
        else:
            # 训练模式
            self.dw_conv = ConvBNLayer(in_c, in_c, dw_size, stride, groups=in_c)
            if stride == 1:
                self.dw_conv_1x1 = ConvBNLayer(in_c, in_c, 1, 1, 0, groups=in_c)

        if use_se:
            self.se = SEModule(in_c)

        self.pw_conv = ConvBNLayer(in_c, out_c, 1, 1, 0, act='hardswish')

    def forward(self, x):
        if self.deploy:
            x = self.dw_conv(x)
        else:
            if hasattr(self, 'dw_conv_1x1'):
                x = self.dw_conv(x) + self.dw_conv_1x1(x)
            else:
                x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPLCNetV2(nn.Module):
    """PP-LCNetV2模型

    Args:
        scale: 网络宽度缩放
        num_classes: 分类数
        dropout_prob: dropout概率
    """

    def __init__(self, scale=1.0, num_classes=1000, dropout_prob=0.2):
        super().__init__()
        self.scale = scale

        # Stage配置
        NET_CONFIG = {
            'stage1': [[3, 16, 32, 1, False]],
            'stage2': [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
            'stage3': [[3, 64, 128, 2, False], [5, 128, 128, 1, False]],
            'stage4': [[5, 128, 256, 2, False], [5, 256, 256, 1, True],
                      [5, 256, 256, 1, True], [5, 256, 256, 1, True]],
            'stage5': [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
        }

        # Conv1
        self.conv1 = ConvBNLayer(3, make_divisible(32 * scale), 3, 2, act='hardswish')

        # Stages
        self.stage1 = self._make_stage(NET_CONFIG['stage1'], scale)
        self.stage2 = self._make_stage(NET_CONFIG['stage2'], scale)
        self.stage3 = self._make_stage(NET_CONFIG['stage3'], scale)
        self.stage4 = self._make_stage(NET_CONFIG['stage4'], scale)
        self.stage5 = self._make_stage(NET_CONFIG['stage5'], scale)

        # Head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        last_c = make_divisible(512 * scale)
        self.last_conv = nn.Conv2d(last_c, 1280, 1, bias=False)
        self.hardswish = HardSwish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(1280, num_classes)

        self._init_weights()

    def _make_stage(self, config, scale):
        layers = []
        for k, in_c, out_c, s, se in config:
            layers.append(
                RepDepthwiseSeparable(
                    make_divisible(in_c * scale),
                    make_divisible(out_c * scale),
                    s, k, se
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
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
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = torch.flatten(x, 1)
        return x


def pplcnetv2_x0_5(num_classes=1000):
    return PPLCNetV2(scale=0.5, num_classes=num_classes)


def pplcnetv2_x0_75(num_classes=1000):
    return PPLCNetV2(scale=0.75, num_classes=num_classes)


def pplcnetv2_x1_0(num_classes=1000):
    return PPLCNetV2(scale=1.0, num_classes=num_classes)


def pplcnetv2_x1_5(num_classes=1000):
    return PPLCNetV2(scale=1.5, num_classes=num_classes)


__all__ = [
    'PPLCNetV2',
    'pplcnetv2_x0_5',
    'pplcnetv2_x0_75',
    'pplcnetv2_x1_0',
    'pplcnetv2_x1_5',
]

