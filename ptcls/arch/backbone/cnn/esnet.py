"""ESNet: Efficient Search Network

一个高效的神经网络搜索结果
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


def channel_shuffle(x, groups):
    """通道混洗"""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, num_channels, height, width)
    return x


class HardSwish(nn.Module):
    """Hard Swish激活函数"""
    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    def forward(self, x):
        return F.relu6(x + 3) / 6


class ConvBNLayer(nn.Module):
    """Conv + BN + Activation"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        if_act=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.if_act = if_act
        self.hardswish = HardSwish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.hardswish(x)
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
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


class ESBlock1(nn.Module):
    """ES Block Type 1 - 无下采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_1_1 = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1
        )
        self.dw_1 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            groups=out_channels // 2,
            if_act=False
        )
        self.se = SEModule(out_channels)
        self.pw_1_2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1
        )

    def forward(self, x):
        # Split
        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)

        x2 = self.pw_1_1(x2)
        x3 = self.dw_1(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self.se(x3)
        x3 = self.pw_1_2(x3)

        x = torch.cat([x1, x3], dim=1)
        return channel_shuffle(x, 2)


class ESBlock2(nn.Module):
    """ES Block Type 2 - 有下采样"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # branch1
        self.dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            if_act=False
        )
        self.pw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1
        )

        # branch2
        self.pw_2_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1
        )
        self.dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            groups=out_channels // 2,
            if_act=False
        )
        self.se = SEModule(out_channels // 2)
        self.pw_2_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1
        )

        # concat layers
        self.concat_dw = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=out_channels
        )
        self.concat_pw = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        # branch1
        x1 = self.dw_1(x)
        x1 = self.pw_1(x1)

        # branch2
        x2 = self.pw_2_1(x)
        x2 = self.dw_2(x2)
        x2 = self.se(x2)
        x2 = self.pw_2_2(x2)

        # concat
        x = torch.cat([x1, x2], dim=1)
        x = self.concat_dw(x)
        x = self.concat_pw(x)
        return x


class ESNet(nn.Module):
    """ESNet: Efficient Search Network

    Args:
        scale: 网络宽度缩放系数
        num_classes: 分类数
        dropout_prob: dropout概率
        class_expand: 最后一层的通道扩展数
    """

    def __init__(
        self,
        scale=1.0,
        num_classes=1000,
        dropout_prob=0.2,
        class_expand=1280
    ):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.class_expand = class_expand

        stage_repeats = [3, 7, 3]
        stage_out_channels = [
            -1, 24,
            make_divisible(116 * scale),
            make_divisible(232 * scale),
            make_divisible(464 * scale),
            1024
        ]

        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # blocks
        block_list = []
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = ESBlock2(
                        in_channels=stage_out_channels[stage_id + 1],
                        out_channels=stage_out_channels[stage_id + 2]
                    )
                else:
                    block = ESBlock1(
                        in_channels=stage_out_channels[stage_id + 2],
                        out_channels=stage_out_channels[stage_id + 2]
                    )
                block_list.append(block)
        self.blocks = nn.Sequential(*block_list)

        # conv2
        self.conv2 = ConvBNLayer(
            in_channels=stage_out_channels[-2],
            out_channels=stage_out_channels[-1],
            kernel_size=1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # last conv
        self.last_conv = nn.Conv2d(
            stage_out_channels[-1],
            class_expand,
            1,
            bias=False
        )
        self.hardswish = HardSwish()
        self.dropout = nn.Dropout(p=dropout_prob)

        # fc
        self.fc = nn.Linear(class_expand, num_classes)

        self._init_weights()

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
        x = self.max_pool(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        """提取特征"""
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = torch.flatten(x, 1)
        return x


def esnet_x0_25(num_classes=1000, **kwargs):
    """ESNet x0.25"""
    return ESNet(scale=0.25, num_classes=num_classes, **kwargs)


def esnet_x0_5(num_classes=1000, **kwargs):
    """ESNet x0.5"""
    return ESNet(scale=0.5, num_classes=num_classes, **kwargs)


def esnet_x0_75(num_classes=1000, **kwargs):
    """ESNet x0.75"""
    return ESNet(scale=0.75, num_classes=num_classes, **kwargs)


def esnet_x1_0(num_classes=1000, **kwargs):
    """ESNet x1.0"""
    return ESNet(scale=1.0, num_classes=num_classes, **kwargs)


__all__ = [
    'ESNet',
    'esnet_x0_25',
    'esnet_x0_5',
    'esnet_x0_75',
    'esnet_x1_0',
]

