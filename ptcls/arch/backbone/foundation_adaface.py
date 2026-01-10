"""Foundation Models and AdaFace

BEiT, CAE, MAE, EVA, MOCOV3, CLIP, AdaFace implementations
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


# ========== Foundation Models (BEiT/MAE/CAE) ==========

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
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


class FoundationViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


# BEiT V2
@register_backbone('beitv2_vit_base_patch16_224')
class BEiTv2_vit_base_patch16_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


@register_backbone('beitv2_vit_large_patch16_224')
class BEiTv2_vit_large_patch16_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


# CAE
@register_backbone('cae_vit_base_patch16_224')
class CAE_vit_base_patch16_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


@register_backbone('cae_base_patch16_224')
class cae_base_patch16_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


@register_backbone('cae_large_patch16_224')
class cae_large_patch16_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


# MAE
@register_backbone('mae_vit_base_patch16')
class MAE_vit_base_patch16(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


@register_backbone('mae_vit_large_patch16')
class MAE_vit_large_patch16(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


@register_backbone('mae_vit_huge_patch14')
class MAE_vit_huge_patch14(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(patch_size=14, embed_dim=1280, depth=32, num_heads=16, num_classes=num_classes)


# EVA
@register_backbone('eva_vit_giant_patch14')
class EVA_vit_giant_patch14(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(patch_size=14, embed_dim=1408, depth=40, num_heads=16, num_classes=num_classes)


# MOCOV3
@register_backbone('mocov3_vit_small')
class MOCOV3_vit_small(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=384, depth=12, num_heads=6, num_classes=num_classes)


@register_backbone('mocov3_vit_base')
class MOCOV3_vit_base(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


# CLIP
@register_backbone('clip_vit_base_patch32_224')
class CLIP_vit_base_patch32_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(patch_size=32, embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


@register_backbone('clip_vit_base_patch16_224')
class CLIP_vit_base_patch16_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)


@register_backbone('clip_vit_large_patch14_224')
class CLIP_vit_large_patch14_224(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(patch_size=14, embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


@register_backbone('clip_vit_large_patch14_336')
class CLIP_vit_large_patch14_336(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=336, patch_size=14, embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


@register_backbone('clip_large_patch14_224_aesthetic')
class CLIP_large_patch14_224_aesthetic(FoundationViT):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(patch_size=14, embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)


# ========== AdaFace ==========

class IResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class IResNet(nn.Module):
    def __init__(self, layers, num_classes=1000, use_se=False):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(64, 64, layers[0])
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * 7 * 7, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(IResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(IResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn3(x)

        return x


@register_backbone('adaface_ir_18')
class AdaFace_IR_18(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([2, 2, 2, 2], num_classes=num_classes)


@register_backbone('adaface_ir_34')
class AdaFace_IR_34(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 4, 6, 3], num_classes=num_classes)


@register_backbone('adaface_ir_50')
class AdaFace_IR_50(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 4, 14, 3], num_classes=num_classes)


@register_backbone('adaface_ir_101')
class AdaFace_IR_101(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 13, 30, 3], num_classes=num_classes)


@register_backbone('adaface_ir_152')
class AdaFace_IR_152(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], num_classes=num_classes)


@register_backbone('adaface_ir_se_50')
class AdaFace_IR_SE_50(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 4, 14, 3], num_classes=num_classes, use_se=True)


@register_backbone('adaface_ir_se_101')
class AdaFace_IR_SE_101(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 13, 30, 3], num_classes=num_classes, use_se=True)


@register_backbone('adaface_ir_se_152')
class AdaFace_IR_SE_152(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([3, 8, 36, 3], num_classes=num_classes, use_se=True)


@register_backbone('adaface_ir_se_200')
class AdaFace_IR_SE_200(IResNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([6, 12, 50, 6], num_classes=num_classes, use_se=True)

