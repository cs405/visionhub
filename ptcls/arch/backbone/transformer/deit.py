"""DeiT (Data-efficient Image Transformers) Implementation

Paper: Training data-efficient image transformers & distillation through attention
https://arxiv.org/abs/2012.12877
"""

import torch
import torch.nn as nn
from .vit_swin import VisionTransformer, PatchEmbed, Attention, Mlp
from ..registry import register_backbone


class DistilledVisionTransformer(VisionTransformer):
    """Vision Transformer with distillation token for DeiT models"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim,
                        depth, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)

        # Distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        # Distillation head
        self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2


# DeiT Tiny
@register_backbone('deit_tiny_patch16_224')
class DeiT_tiny_patch16_224(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=192, depth=12,
                        num_heads=3, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


@register_backbone('deit_tiny_distilled_patch16_224')
class DeiT_tiny_distilled_patch16_224(DistilledVisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=192, depth=12,
                        num_heads=3, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


# DeiT Small
@register_backbone('deit_small_patch16_224')
class DeiT_small_patch16_224(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=384, depth=12,
                        num_heads=6, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


@register_backbone('deit_small_distilled_patch16_224')
class DeiT_small_distilled_patch16_224(DistilledVisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=384, depth=12,
                        num_heads=6, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


# DeiT Base
@register_backbone('deit_base_patch16_224')
class DeiT_base_patch16_224(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


@register_backbone('deit_base_distilled_patch16_224')
class DeiT_base_distilled_patch16_224(DistilledVisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


# DeiT Base 384
@register_backbone('deit_base_patch16_384')
class DeiT_base_patch16_384(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=384, patch_size=16, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


@register_backbone('deit_base_distilled_patch16_384')
class DeiT_base_distilled_patch16_384(DistilledVisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=384, patch_size=16, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)

