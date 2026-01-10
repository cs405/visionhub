"""Additional ViT (Vision Transformer) models

Complete the ViT model family
"""

import torch
import torch.nn as nn
from .vit_swin import VisionTransformer
from ..registry import register_backbone


@register_backbone('vit_base_patch16_224')
class ViT_base_patch16_224(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


@register_backbone('vit_large_patch16_224')
class ViT_large_patch16_224(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=224, patch_size=16, embed_dim=1024, depth=24,
                        num_heads=16, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)


@register_backbone('vit_base_patch32_384')
class ViT_base_patch32_384(VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=384, patch_size=32, embed_dim=768, depth=12,
                        num_heads=12, mlp_ratio=4., qkv_bias=True, num_classes=num_classes)

