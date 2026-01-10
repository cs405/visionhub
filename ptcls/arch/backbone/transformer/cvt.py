"""CVT: Convolutional vision Transformer

完整迁移自PaddleClas CVT
Paper: CvT: Introducing Convolutions to Vision Transformers
"""

import torch
import torch.nn as nn


class ConvEmbed(nn.Module):
    """卷积嵌入层"""
    def __init__(self, in_c, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class ConvAttention(nn.Module):
    """卷积注意力"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 深度卷积用于位置编码
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # 添加位置信息
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_2d = self.dwconv(x_2d)
        x = x + x_2d.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
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


class CVTBlock(nn.Module):
    """CVT Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class CVTStage(nn.Module):
    """CVT Stage"""
    def __init__(self, in_c, embed_dim, num_heads, depth, patch_size, stride, padding, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = ConvEmbed(in_c, embed_dim, patch_size, stride, padding)
        self.blocks = nn.ModuleList([
            CVTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).reshape(x.size(0), -1, H, W)
        return x


class CVT(nn.Module):
    """CVT模型

    Args:
        num_classes: 分类数
        embed_dims: 嵌入维度列表
        num_heads: 注意力头数列表
        depths: 每个stage的深度
    """

    def __init__(
        self,
        num_classes=1000,
        embed_dims=None,
        num_heads=None,
        depths=None,
        mlp_ratio=4.
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [64, 192, 384]
        if num_heads is None:
            num_heads = [1, 3, 6]
        if depths is None:
            depths = [1, 2, 10]

        self.num_stages = len(depths)

        # Stage配置
        stage_configs = [
            (3, embed_dims[0], num_heads[0], depths[0], 7, 4, 3),
            (embed_dims[0], embed_dims[1], num_heads[1], depths[1], 3, 2, 1),
            (embed_dims[1], embed_dims[2], num_heads[2], depths[2], 3, 2, 1),
        ]

        self.stages = nn.ModuleList()
        for in_c, dim, heads, depth, patch_size, stride, padding in stage_configs:
            self.stages.append(
                CVTStage(in_c, dim, heads, depth, patch_size, stride, padding, mlp_ratio)
            )

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward_features(self, x):
        for stage in self.stages:
            x = stage(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def cvt_13(num_classes=1000):
    """CVT-13"""
    return CVT(
        num_classes=num_classes,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )


def cvt_21(num_classes=1000):
    """CVT-21"""
    return CVT(
        num_classes=num_classes,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 4, 16]
    )


__all__ = ['CVT', 'cvt_13', 'cvt_21']

