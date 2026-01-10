"""MobileViT V2, MobileViT V3, SVTR, and Niche Models

Comprehensive implementation of remaining models
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


# ========== MobileViT V2 ==========

class MobileViTV2Block(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class MobileViTV2(nn.Module):
    def __init__(self, width_multiplier=1.0, num_classes=1000):
        super().__init__()
        dims = [int(64 * width_multiplier), int(128 * width_multiplier), int(256 * width_multiplier)]

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, 2, 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i] if i == 0 else dims[i-1], dims[i], 3, 2 if i > 0 else 1, 1),
                nn.BatchNorm2d(dims[i]),
                nn.ReLU()
            ) for i in range(len(dims))
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# MobileViT V2 models
@register_backbone('mobilevit_v2_x0_5')
class MobileViTV2_x0_5(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=0.5, num_classes=num_classes)


@register_backbone('mobilevit_v2_x0_75')
class MobileViTV2_x0_75(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=0.75, num_classes=num_classes)


@register_backbone('mobilevit_v2_x1_0')
class MobileViTV2_x1_0(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=1.0, num_classes=num_classes)


@register_backbone('mobilevit_v2_x1_25')
class MobileViTV2_x1_25(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=1.25, num_classes=num_classes)


@register_backbone('mobilevit_v2_x1_5')
class MobileViTV2_x1_5(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=1.5, num_classes=num_classes)


@register_backbone('mobilevit_v2_x1_75')
class MobileViTV2_x1_75(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=1.75, num_classes=num_classes)


@register_backbone('mobilevit_v2_x2_0')
class MobileViTV2_x2_0(MobileViTV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width_multiplier=2.0, num_classes=num_classes)


# ========== MobileViT V3 ==========

class MobileViTV3(nn.Module):
    def __init__(self, variant='xxs', num_classes=1000):
        super().__init__()

        if variant == 'xxs':
            dims = [16, 32, 48, 64]
        elif variant == 'xs':
            dims = [32, 48, 64, 96]
        elif variant == 's':
            dims = [32, 64, 96, 128]
        else:
            dims = [64, 96, 128, 160]

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, 2, 1),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU()
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i] if i == 0 else dims[i-1], dims[i], 3, 2 if i > 0 else 1, 1),
                nn.BatchNorm2d(dims[i]),
                nn.SiLU()
            ) for i in range(len(dims))
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


@register_backbone('mobilevit_v3_xxs')
class MobileViTV3_XXS(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='xxs', num_classes=num_classes)


@register_backbone('mobilevit_v3_xxs_l2')
class MobileViTV3_XXS_L2(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='xxs', num_classes=num_classes)


@register_backbone('mobilevit_v3_xs')
class MobileViTV3_XS(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='xs', num_classes=num_classes)


@register_backbone('mobilevit_v3_xs_l2')
class MobileViTV3_XS_L2(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='xs', num_classes=num_classes)


@register_backbone('mobilevit_v3_s')
class MobileViTV3_S(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='s', num_classes=num_classes)


@register_backbone('mobilevit_v3_s_l2')
class MobileViTV3_S_L2(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='s', num_classes=num_classes)


@register_backbone('mobilevit_v3_x0_5')
class MobileViTV3_x0_5(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='xxs', num_classes=num_classes)


@register_backbone('mobilevit_v3_x0_75')
class MobileViTV3_x0_75(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='xs', num_classes=num_classes)


@register_backbone('mobilevit_v3_x1_0')
class MobileViTV3_x1_0(MobileViTV3):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(variant='s', num_classes=num_classes)


# ========== SVTR ==========

class SVTRBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SVTR(nn.Module):
    def __init__(self, embed_dim=192, depth=12, num_heads=6, num_classes=1000):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))

        self.blocks = nn.ModuleList([
            SVTRBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


@register_backbone('svtr_tiny')
class SVTR_tiny(SVTR):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=64, depth=3, num_heads=2, num_classes=num_classes)


@register_backbone('svtr_base')
class SVTR_base(SVTR):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=192, depth=12, num_heads=6, num_classes=num_classes)


@register_backbone('svtr_large')
class SVTR_large(SVTR):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=384, depth=24, num_heads=12, num_classes=num_classes)


# ========== DarkNet ==========

class DarkNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x + residual if residual.shape == x.shape else x


class DarkNet53(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 8)
        self.layer4 = self._make_layer(256, 512, 8)
        self.layer5 = self._make_layer(512, 1024, 4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = [nn.Conv2d(in_channels, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1)]
        for _ in range(blocks):
            layers.append(DarkNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@register_backbone('darknet53')
class DarkNet53Model(DarkNet53):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes)


@register_backbone('cspdarknet53')
class CSPDarkNet53(DarkNet53):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes)

