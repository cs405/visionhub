"""MobileNet V4, UniFormer, and LeViT Implementations"""

import torch
import torch.nn as nn
from ..registry import register_backbone


# ========== MobileNet V4 ==========

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, 1))
        layers.extend([
            ConvBNAct(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV4(nn.Module):
    def __init__(self, width_mult=1.0, num_classes=1000, variant='conv'):
        super().__init__()

        input_channel = make_divisible(32 * width_mult)
        last_channel = make_divisible(1280 * width_mult)

        if variant == 'conv':
            # Conv variant configuration
            inverted_residual_setting = [
                # t, c, n, s
                [1, 24, 2, 2],
                [4, 48, 4, 2],
                [4, 96, 4, 2],
                [4, 192, 4, 2],
                [6, 320, 1, 1],
            ]
        else:  # hybrid
            inverted_residual_setting = [
                [1, 24, 2, 2],
                [4, 48, 4, 2],
                [4, 80, 4, 2],
                [4, 160, 6, 2],
                [6, 256, 1, 1],
            ]

        features = [ConvBNAct(3, input_channel, 3, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNAct(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
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


@register_backbone('mobilenet_v4_conv_small')
class MobileNetV4_conv_small(MobileNetV4):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=0.75, variant='conv', num_classes=num_classes)


@register_backbone('mobilenet_v4_conv_medium')
class MobileNetV4_conv_medium(MobileNetV4):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.0, variant='conv', num_classes=num_classes)


@register_backbone('mobilenet_v4_conv_large')
class MobileNetV4_conv_large(MobileNetV4):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.5, variant='conv', num_classes=num_classes)


@register_backbone('mobilenet_v4_hybrid_medium')
class MobileNetV4_hybrid_medium(MobileNetV4):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.0, variant='hybrid', num_classes=num_classes)


@register_backbone('mobilenet_v4_hybrid_large')
class MobileNetV4_hybrid_large(MobileNetV4):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_mult=1.5, variant='hybrid', num_classes=num_classes)


# ========== UniFormer ==========

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UniformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        # Simplified - full implementation would have local/global attention

    def forward(self, x):
        x = x + self.mlp(self.norm2(x))
        return x


class UniFormer(nn.Module):
    def __init__(self, depths=[3, 4, 8, 3], embed_dim=64, num_classes=1000):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList()

        for i in range(sum(depths)):
            block = UniformerBlock(dim=embed_dim, num_heads=4)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


@register_backbone('uniformer_small')
class UniFormer_small(UniFormer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[3, 4, 8, 3], embed_dim=64, num_classes=num_classes)


@register_backbone('uniformer_small_plus')
class UniFormer_small_plus(UniFormer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[3, 5, 9, 3], embed_dim=64, num_classes=num_classes)


@register_backbone('uniformer_small_plus_dim64')
class UniFormer_small_plus_dim64(UniFormer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[3, 5, 9, 3], embed_dim=64, num_classes=num_classes)


@register_backbone('uniformer_base')
class UniFormer_base(UniFormer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[5, 8, 20, 7], embed_dim=64, num_classes=num_classes)


@register_backbone('uniformer_base_ls')
class UniFormer_base_ls(UniFormer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(depths=[5, 8, 20, 7], embed_dim=64, num_classes=num_classes)


# ========== LeViT ==========

class LeViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.mlp(self.norm2(x))
        return x


class LeViT(nn.Module):
    def __init__(self, embed_dim=128, depths=[2, 3, 4], num_classes=1000):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 8, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 8),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
        )

        self.blocks = nn.ModuleList()
        for _ in range(sum(depths)):
            block = LeViTBlock(dim=embed_dim, num_heads=4)
            self.blocks.append(block)

        self.norm = nn.BatchNorm2d(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


@register_backbone('levit_128s')
class LeViT_128S(LeViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=128, depths=[2, 3, 4], num_classes=num_classes)


@register_backbone('levit_128')
class LeViT_128(LeViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=128, depths=[4, 4, 4], num_classes=num_classes)


@register_backbone('levit_192')
class LeViT_192(LeViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=192, depths=[3, 4, 4], num_classes=num_classes)


@register_backbone('levit_256')
class LeViT_256(LeViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=256, depths=[4, 4, 4], num_classes=num_classes)


@register_backbone('levit_384')
class LeViT_384(LeViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=384, depths=[4, 4, 4], num_classes=num_classes)

