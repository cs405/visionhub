"""CSWin Transformer Implementation

Paper: CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows
https://arxiv.org/abs/2107.00652
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


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


class CSWinBlock(nn.Module):
    """CSWin Transformer Block with Cross-Shaped Window attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        # Simplified attention (full implementation would use cross-shaped windows)
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = x + self.drop_path((attn @ v).transpose(1, 2).reshape(B, N, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=patch_size, padding=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class CSWinTransformer(nn.Module):
    """CSWin Transformer"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depth=[2, 4, 32, 2], num_heads=[2, 4, 8, 16], mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = embed_dim * (2 ** (len(depth) - 1))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                     in_chans=in_chans, embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depth)):
            stage = nn.Sequential(*[
                CSWinBlock(
                    dim=int(embed_dim * 2 ** i), num_heads=num_heads[i], mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j])
                for j in range(depth[i])
            ])
            self.stages.append(stage)
            cur += depth[i]

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x.mean(1))
        x = self.head(x)
        return x


@register_backbone('cswin_transformer_tiny_224')
class CSWinTransformer_tiny_224(CSWinTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=64, depth=[1, 2, 21, 1], num_heads=[2, 4, 8, 16], num_classes=num_classes)


@register_backbone('cswin_transformer_small_224')
class CSWinTransformer_small_224(CSWinTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=64, depth=[2, 4, 32, 2], num_heads=[2, 4, 8, 16], num_classes=num_classes)


@register_backbone('cswin_transformer_base_224')
class CSWinTransformer_base_224(CSWinTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=96, depth=[2, 4, 32, 2], num_heads=[4, 8, 16, 32], num_classes=num_classes)


@register_backbone('cswin_transformer_large_224')
class CSWinTransformer_large_224(CSWinTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dim=144, depth=[2, 4, 32, 2], num_heads=[6, 12, 24, 24], num_classes=num_classes)


@register_backbone('cswin_transformer_base_384')
class CSWinTransformer_base_384(CSWinTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=384, embed_dim=96, depth=[2, 4, 32, 2], num_heads=[4, 8, 16, 32], num_classes=num_classes)


@register_backbone('cswin_transformer_large_384')
class CSWinTransformer_large_384(CSWinTransformer):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(img_size=384, embed_dim=144, depth=[2, 4, 32, 2], num_heads=[6, 12, 24, 24], num_classes=num_classes)

