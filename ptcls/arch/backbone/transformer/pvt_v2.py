"""PVT V2 (Pyramid Vision Transformer V2) Implementation

Paper: PVTv2: Improved Baselines with Pyramid Vision Transformer
https://arxiv.org/abs/2106.13797
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register_backbone


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_)
                x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_))
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                             proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, linear=linear)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    """PVT V2"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=False):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths

        # Patch embeddings
        self.patch_embeds = nn.ModuleList()
        self.patch_embeds.append(OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]))
        for i in range(1, len(depths)):
            self.patch_embeds.append(OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[i-1], embed_dim=embed_dims[i]))

        # Transformer encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            self.blocks.append(block)
            cur += depths[i]

        self.norm = nn.LayerNorm(embed_dims[3])

        self.head = nn.Linear(embed_dims[3], num_classes)

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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i != len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x


@register_backbone('pvt_v2_b0')
class PVT_V2_B0(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                        qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)


@register_backbone('pvt_v2_b1')
class PVT_V2_B1(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                        qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)


@register_backbone('pvt_v2_b2')
class PVT_V2_B2(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                        qkv_bias=True, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)


@register_backbone('pvt_v2_b2_linear')
class PVT_V2_B2_Linear(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                        qkv_bias=True, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, num_classes=num_classes)


@register_backbone('pvt_v2_b3')
class PVT_V2_B3(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                        qkv_bias=True, depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)


@register_backbone('pvt_v2_b4')
class PVT_V2_B4(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                        qkv_bias=True, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)


@register_backbone('pvt_v2_b5')
class PVT_V2_B5(PyramidVisionTransformerV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                        qkv_bias=True, depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)

