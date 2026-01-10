"""VAN (Visual Attention Network) Implementation

Paper: Visual Attention Network
https://arxiv.org/abs/2202.09741
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


class DWConv(nn.Module):
    """Depthwise Convolution"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    """Large Kernel Attention"""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
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
    """VAN Block"""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Layer scale
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                             stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class VAN(nn.Module):
    """Visual Attention Network"""
    def __init__(self, embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0., drop_path_rate=0., depths=[3, 3, 5, 2], num_classes=1000):
        super().__init__()

        self.depths = depths
        self.num_stages = len(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            # Patch embedding
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=3 if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])

            # Blocks
            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                drop_path=dpr[cur + j])
                for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            for blk in block:
                x = blk(x)

        x = x.mean(dim=[-2, -1])  # Global average pooling
        x = self.head(x)

        return x


@register_backbone('van_b0')
class VAN_B0(VAN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
                        drop_path_rate=0.0, depths=[3, 3, 5, 2], num_classes=num_classes)


@register_backbone('van_b1')
class VAN_B1(VAN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
                        drop_path_rate=0.1, depths=[2, 2, 4, 2], num_classes=num_classes)


@register_backbone('van_b2')
class VAN_B2(VAN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
                        drop_path_rate=0.2, depths=[3, 3, 12, 3], num_classes=num_classes)


@register_backbone('van_b3')
class VAN_B3(VAN):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
                        drop_path_rate=0.3, depths=[3, 5, 27, 3], num_classes=num_classes)

