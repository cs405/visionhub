"""PP-HGNetV2: High Performance GPU Network V2

完整迁移自PaddleClas PP-HGNetV2
PaddlePaddle第二代高性能GPU网络
"""

import torch
import torch.nn as nn


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


class LightConvBNAct(nn.Module):
    """轻量级卷积：DW + PW"""
    def __init__(self, in_c, out_c, k, s=1):
        super().__init__()
        self.dw_conv = ConvBNAct(in_c, in_c, k, s, groups=in_c)
        self.pw_conv = ConvBNAct(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class StemBlock(nn.Module):
    """PP-HGNetV2的Stem块"""
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.stem1 = ConvBNAct(in_c, mid_c, 3, 2)
        self.stem2a = ConvBNAct(mid_c, mid_c // 2, 2, 1, 0)
        self.stem2b = ConvBNAct(mid_c // 2, mid_c, 2, 1, 0)
        self.stem3 = ConvBNAct(mid_c * 2, mid_c, 3, 2)
        self.stem4 = ConvBNAct(mid_c, out_c, 1, 1, 0)

    def forward(self, x):
        x = self.stem1(x)
        x2a = self.stem2a(x)
        x2a = self.stem2b(x2a)
        x = torch.cat([x, x2a], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGV2Block(nn.Module):
    """PP-HGNetV2块"""
    def __init__(self, in_c, mid_c, out_c, kernel_size=3, layer_num=6, identity=False, light_block=True):
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()

        if light_block:
            block = LightConvBNAct
        else:
            block = lambda in_c, out_c, k, s: ConvBNAct(in_c, out_c, k, s)

        # 第一层
        self.layers.append(ConvBNAct(in_c, mid_c, 3, 1))

        # 中间层
        for i in range(layer_num - 1):
            self.layers.append(block(mid_c, mid_c, kernel_size, 1))

        # Aggregation
        total_c = in_c + mid_c * layer_num
        self.aggregation_conv = ConvBNAct(total_c, out_c, 1, 1, 0)

        # Attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(out_c, out_c // 4, 1, 1, 0),
            ConvBNAct(out_c // 4, out_c, 1, 1, 0, act=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        outputs = [x]

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.aggregation_conv(x)

        # Apply attention
        attn = self.attention(x)
        x = x * attn

        if self.identity:
            x = x + identity

        return x


class HGV2Stage(nn.Module):
    """PP-HGNetV2 Stage"""
    def __init__(self, in_c, mid_c, out_c, block_num, layer_num, downsample=True, light_block=True):
        super().__init__()

        self.downsample = None
        if downsample:
            self.downsample = ConvBNAct(in_c, in_c, 3, 2, groups=in_c, act=False)

        blocks = []
        blocks.append(HGV2Block(in_c, mid_c, out_c, 3, layer_num, False, light_block))

        for _ in range(block_num - 1):
            blocks.append(HGV2Block(out_c, mid_c, out_c, 3, layer_num, True, light_block))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNetV2(nn.Module):
    """PP-HGNetV2模型

    第二代高性能GPU网络，改进的架构和更好的性能

    Args:
        stem_channels: Stem通道配置
        stage_config: Stage配置
        num_classes: 分类数
        dropout_prob: Dropout概率
        use_light_block: 是否使用轻量级块
    """

    def __init__(
        self,
        stem_channels=None,
        stage_config=None,
        num_classes=1000,
        dropout_prob=0.2,
        use_light_block=True
    ):
        super().__init__()

        if stem_channels is None:
            stem_channels = [3, 32, 48, 96]

        if stage_config is None:
            # [in_c, mid_c, out_c, block_num, layer_num]
            stage_config = {
                'stage1': [96, 96, 224, 2, 6],
                'stage2': [224, 128, 448, 2, 6],
                'stage3': [448, 256, 512, 3, 6],
                'stage4': [512, 384, 768, 2, 6],
            }

        # Stem
        self.stem = StemBlock(stem_channels[0], stem_channels[1], stem_channels[3])

        # Stages
        self.stage1 = HGV2Stage(*stage_config['stage1'], downsample=False, light_block=use_light_block)
        self.stage2 = HGV2Stage(*stage_config['stage2'], downsample=True, light_block=use_light_block)
        self.stage3 = HGV2Stage(*stage_config['stage3'], downsample=True, light_block=use_light_block)
        self.stage4 = HGV2Stage(*stage_config['stage4'], downsample=True, light_block=use_light_block)

        # Head
        last_channels = stage_config['stage4'][2]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_conv = ConvBNAct(last_channels, 2048, 1, 1, 0)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(2048, num_classes)

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
        x = torch.flatten(x, 1)
        return x


def pphgnetv2_b0(num_classes=1000):
    """PP-HGNetV2-B0"""
    return PPHGNetV2(
        stem_channels=[3, 16, 16, 32],
        stage_config={
            'stage1': [32, 32, 64, 1, 4],
            'stage2': [64, 48, 128, 1, 4],
            'stage3': [128, 96, 256, 2, 6],
            'stage4': [256, 192, 512, 1, 6],
        },
        num_classes=num_classes
    )


def pphgnetv2_b1(num_classes=1000):
    """PP-HGNetV2-B1"""
    return PPHGNetV2(
        stem_channels=[3, 24, 32, 64],
        stage_config={
            'stage1': [64, 64, 128, 2, 5],
            'stage2': [128, 96, 256, 2, 5],
            'stage3': [256, 192, 512, 3, 6],
            'stage4': [512, 384, 1024, 2, 6],
        },
        num_classes=num_classes
    )


def pphgnetv2_b4(num_classes=1000):
    """PP-HGNetV2-B4"""
    return PPHGNetV2(num_classes=num_classes)


__all__ = [
    'PPHGNetV2',
    'pphgnetv2_b0',
    'pphgnetv2_b1',
    'pphgnetv2_b4',
]

