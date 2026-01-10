"""MobileViT: Light-weight, General-purpose Vision Transformer

轻量级通用视觉Transformer
Paper: MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
"""

import torch
import torch.nn as nn


class ConvBN(nn.Module):
    """Conv + BN + Activation"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV2风格的倒残差块"""
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        hidden_c = int(in_c * expand_ratio)
        self.use_residual = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBN(in_c, hidden_c, k=1, p=0))

        layers.extend([
            ConvBN(hidden_c, hidden_c, k=3, s=stride, groups=hidden_c),
            ConvBN(hidden_c, out_c, k=1, p=0, act=False)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """MobileViT块：结合卷积和Transformer"""
    def __init__(self, in_c, out_c, d_model, num_heads, num_layers):
        super().__init__()

        # Local representation
        self.local_rep = nn.Sequential(
            ConvBN(in_c, in_c, k=3),
            ConvBN(in_c, d_model, k=1, p=0)
        )

        # Global representation (Transformer)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        )

        # Fusion
        self.fusion = nn.Sequential(
            ConvBN(d_model, in_c, k=1, p=0),
            ConvBN(in_c, out_c, k=3)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Local features
        local_feat = self.local_rep(x)

        # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        transformer_input = local_feat.flatten(2).transpose(1, 2)

        # Global features
        global_feat = self.transformer(transformer_input)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        global_feat = global_feat.transpose(1, 2).reshape(B, -1, H, W)

        # Fusion
        out = self.fusion(global_feat)
        return out


class MobileViT(nn.Module):
    """MobileViT模型

    Args:
        width_multiplier: 宽度乘数
        num_classes: 分类数
    """

    def __init__(self, width_multiplier=1.0, num_classes=1000):
        super().__init__()

        # 根据width_multiplier调整通道数
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        channels = [16, 32, 64, 96, 128, 160, 640]
        channels = [_make_divisible(c * width_multiplier) for c in channels]

        # Stem
        self.stem = ConvBN(3, channels[0], k=3, s=2)

        # Stages
        self.stage1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], 1, 2),
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], 2, 2),
            InvertedResidual(channels[2], channels[2], 1, 2),
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], 2, 2),
            MobileViTBlock(channels[3], channels[3], d_model=144, num_heads=4, num_layers=2),
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], 2, 2),
            MobileViTBlock(channels[4], channels[4], d_model=192, num_heads=4, num_layers=4),
        )

        self.stage5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], 2, 2),
            MobileViTBlock(channels[5], channels[5], d_model=240, num_heads=4, num_layers=3),
        )

        # Head
        self.conv_last = ConvBN(channels[5], channels[6], k=1, p=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[6], num_classes)

        self._init_weights()

    def _init_weights(self):
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def mobilevit_xxs(num_classes=1000):
    """MobileViT-XXS"""
    return MobileViT(width_multiplier=0.5, num_classes=num_classes)


def mobilevit_xs(num_classes=1000):
    """MobileViT-XS"""
    return MobileViT(width_multiplier=0.75, num_classes=num_classes)


def mobilevit_s(num_classes=1000):
    """MobileViT-S"""
    return MobileViT(width_multiplier=1.0, num_classes=num_classes)


__all__ = [
    'MobileViT',
    'mobilevit_xxs',
    'mobilevit_xs',
    'mobilevit_s',
]

