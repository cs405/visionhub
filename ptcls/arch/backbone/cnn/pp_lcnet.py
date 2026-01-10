"""PP-LCNet: A Lightweight CPU Convolutional Neural Network

完整迁移自PaddleClas PP-LCNet
Paper: PP-LCNet: A Lightweight CPU Convolutional Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v, divisor=8, min_value=None):
    """确保通道数可被divisor整除"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    """Hard Swish激活函数"""
    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    def forward(self, x):
        return F.relu6(x + 3) / 6


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            HardSigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ConvBNLayer(nn.Module):
    """Conv + BN + Activation"""
    def __init__(
        self,
        num_channels,
        filter_size,
        num_filters,
        stride,
        num_groups=1,
        act='hardswish'
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            filter_size,
            stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(num_filters)

        if act == 'hardswish':
            self.act = HardSwish()
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparable(nn.Module):
    """深度可分离卷积"""
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        dw_size=3,
        use_se=False,
        act='hardswish'
    ):
        super().__init__()
        self.use_se = use_se

        # Depthwise
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            act=act
        )

        # SE
        if use_se:
            self.se = SEModule(num_channels)

        # Pointwise
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            act=act
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


# Network configuration
# [k, in_c, out_c, s, use_se]
NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False]
    ],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


class PPLCNet(nn.Module):
    """PP-LCNet: A Lightweight CPU Convolutional Neural Network

    Args:
        scale: 网络宽度缩放系数
        num_classes: 分类数
        dropout_prob: dropout概率
        class_expand: 最后一层的通道扩展数
        stride_list: 各stage的stride列表
        use_last_conv: 是否使用最后的1x1卷积
        act: 激活函数类型
    """

    def __init__(
        self,
        scale=1.0,
        num_classes=1000,
        dropout_prob=0.2,
        class_expand=1280,
        stride_list=None,
        use_last_conv=True,
        act='hardswish'
    ):
        super().__init__()
        if stride_list is None:
            stride_list = [2, 2, 2, 2, 2]
        self.scale = scale
        self.class_expand = class_expand
        self.use_last_conv = use_last_conv

        # 修改stride
        net_config = {k: [[c[0], c[1], c[2], c[3], c[4]] for c in v]
                     for k, v in NET_CONFIG.items()}
        for i, stride in enumerate(stride_list[1:]):
            net_config[f"blocks{i+3}"][0][3] = stride

        # conv1
        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=stride_list[0],
            act=act
        )

        # blocks2-6
        self.blocks2 = self._make_layer(net_config["blocks2"], scale, act)
        self.blocks3 = self._make_layer(net_config["blocks3"], scale, act)
        self.blocks4 = self._make_layer(net_config["blocks4"], scale, act)
        self.blocks5 = self._make_layer(net_config["blocks5"], scale, act)
        self.blocks6 = self._make_layer(net_config["blocks6"], scale, act)

        # avg pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # last conv
        if use_last_conv:
            last_in = make_divisible(net_config["blocks6"][-1][2] * scale)
            self.last_conv = nn.Conv2d(last_in, class_expand, 1, bias=False)
            self.act = HardSwish() if act == 'hardswish' else nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=dropout_prob)
            fc_in = class_expand
        else:
            self.last_conv = None
            fc_in = make_divisible(net_config["blocks6"][-1][2] * scale)

        # fc
        self.fc = nn.Linear(fc_in, num_classes)

        self._init_weights()

    def _make_layer(self, config, scale, act):
        """构建一个stage"""
        layers = []
        for k, in_c, out_c, s, se in config:
            layers.append(DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                act=act
            ))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """初始化权重"""
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)

        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.act(x)
            x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        """提取特征"""
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        x = self.avg_pool(x)

        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.act(x)

        x = torch.flatten(x, 1)
        return x


def pplcnet_x0_25(num_classes=1000, **kwargs):
    """PP-LCNet x0.25"""
    return PPLCNet(scale=0.25, num_classes=num_classes, **kwargs)


def pplcnet_x0_35(num_classes=1000, **kwargs):
    """PP-LCNet x0.35"""
    return PPLCNet(scale=0.35, num_classes=num_classes, **kwargs)


def pplcnet_x0_5(num_classes=1000, **kwargs):
    """PP-LCNet x0.5"""
    return PPLCNet(scale=0.5, num_classes=num_classes, **kwargs)


def pplcnet_x0_75(num_classes=1000, **kwargs):
    """PP-LCNet x0.75"""
    return PPLCNet(scale=0.75, num_classes=num_classes, **kwargs)


def pplcnet_x1_0(num_classes=1000, **kwargs):
    """PP-LCNet x1.0"""
    return PPLCNet(scale=1.0, num_classes=num_classes, **kwargs)


def pplcnet_x1_5(num_classes=1000, **kwargs):
    """PP-LCNet x1.5"""
    return PPLCNet(scale=1.5, num_classes=num_classes, **kwargs)


def pplcnet_x2_0(num_classes=1000, **kwargs):
    """PP-LCNet x2.0"""
    return PPLCNet(scale=2.0, num_classes=num_classes, **kwargs)


def pplcnet_x2_5(num_classes=1000, **kwargs):
    """PP-LCNet x2.5"""
    return PPLCNet(scale=2.5, num_classes=num_classes, **kwargs)


__all__ = [
    'PPLCNet',
    'pplcnet_x0_25',
    'pplcnet_x0_35',
    'pplcnet_x0_5',
    'pplcnet_x0_75',
    'pplcnet_x1_0',
    'pplcnet_x1_5',
    'pplcnet_x2_0',
    'pplcnet_x2_5',
]

