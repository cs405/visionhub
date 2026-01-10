"""PP-HGNet: High Performance GPU Network

Optimized high-performance network for GPU devices
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


class ConvBNAct(nn.Module):
    """Conv + BN + Activation"""
    def __init__(self, in_c, out_c, k, s, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ESEModule(nn.Module):
    """Effective Squeeze-Excitation"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class HGBlock(nn.Module):
    """HG Block - 高性能GPU块"""
    def __init__(self, in_c, mid_c, out_c, layer_num, identity=False):
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        self.layers.append(ConvBNAct(in_c, mid_c, 3, 1))

        for _ in range(layer_num - 1):
            self.layers.append(ConvBNAct(mid_c, mid_c, 3, 1))

        # Feature aggregation
        total_c = in_c + layer_num * mid_c
        self.aggregation = ConvBNAct(total_c, out_c, 1, 1, 0)
        self.ese = ESEModule(out_c)

    def forward(self, x):
        identity = x
        outputs = [x]

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.aggregation(x)
        x = self.ese(x)

        if self.identity:
            x = x + identity

        return x


class HGStage(nn.Module):
    """HG Stage"""
    def __init__(self, in_c, mid_c, out_c, block_num, layer_num, downsample=True):
        super().__init__()

        self.downsample = None
        if downsample:
            self.downsample = ConvBNAct(in_c, in_c, 3, 2, groups=in_c, act=False)

        blocks = []
        blocks.append(HGBlock(in_c, mid_c, out_c, layer_num, identity=False))

        for _ in range(block_num - 1):
            blocks.append(HGBlock(out_c, mid_c, out_c, layer_num, identity=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNet(nn.Module):
    """PP-HGNet: High Performance GPU Network

    Args:
        stem_channels: Stem通道配置
        stage_config: Stage配置
        layer_num: HG Block的层数
        num_classes: 分类数
        dropout_prob: Dropout概率
        class_expand: 最后一层扩展通道数
    """

    def __init__(
        self,
        stem_channels=None,
        stage_config=None,
        layer_num=3,
        num_classes=1000,
        dropout_prob=0.2,
        class_expand=2048
    ):
        super().__init__()

        if stem_channels is None:
            stem_channels = [3, 32, 48]
        if stage_config is None:
            stage_config = {
                'stage1': [48, 48, 128, 1, 2],
                'stage2': [128, 96, 256, 1, 2],
                'stage3': [256, 192, 512, 2, 3],
                'stage4': [512, 384, 1024, 1, 3],
            }

        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(stem_channels[0], stem_channels[1], 3, 2),
            ConvBNAct(stem_channels[1], stem_channels[2], 3, 1)
        )

        # Stages
        self.stage1 = HGStage(*stage_config['stage1'], downsample=False)
        self.stage2 = HGStage(*stage_config['stage2'])
        self.stage3 = HGStage(*stage_config['stage3'])
        self.stage4 = HGStage(*stage_config['stage4'])

        # Head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_conv = nn.Conv2d(
            stage_config['stage4'][2], class_expand, 1, bias=False
        )
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(class_expand, num_classes)

        self._init_weights()

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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = self.last_conv(x)
        x = self.act(x)
        x = torch.flatten(x, 1)
        return x


def pphgnet_tiny(num_classes=1000):
    """PP-HGNet Tiny"""
    stage_config = {
        'stage1': [48, 48, 128, 1, 2],
        'stage2': [128, 96, 256, 1, 2],
        'stage3': [256, 192, 512, 2, 3],
        'stage4': [512, 384, 1024, 1, 3],
    }
    return PPHGNet(
        stem_channels=[3, 32, 48],
        stage_config=stage_config,
        num_classes=num_classes,
        class_expand=2048
    )


def pphgnet_small(num_classes=1000):
    """PP-HGNet Small"""
    stage_config = {
        'stage1': [64, 64, 256, 1, 2],
        'stage2': [256, 128, 512, 1, 3],
        'stage3': [512, 256, 1024, 2, 4],
        'stage4': [1024, 512, 2048, 1, 4],
    }
    return PPHGNet(
        stem_channels=[3, 48, 64],
        stage_config=stage_config,
        num_classes=num_classes,
        class_expand=2048
    )


__all__ = [
    'PPHGNet',
    'pphgnet_tiny',
    'pphgnet_small',
]

