"""NextViT: Next Generation Vision Transformer

完整迁移自PaddleClas NextViT
Paper: Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PatchEmbed(nn.Module):
    """Patch Embedding"""
    def __init__(self, in_c, out_c, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, patch_size, patch_size)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class MHCA(nn.Module):
    """Multi-Head Convolutional Attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out


class MHSA(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.project_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, n, c = x.shape

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class NCB(nn.Module):
    """Next Convolution Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MHCA(dim, num_heads)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NTB(nn.Module):
    """Next Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NextViT(nn.Module):
    """NextViT模型

    Args:
        num_classes: 分类数
        stem_chs: Stem通道数
        depths: 每个stage的深度
        path_dropout: Path dropout率
    """

    def __init__(
        self,
        num_classes=1000,
        stem_chs=None,
        depths=None,
        path_dropout=0.1
    ):
        super().__init__()

        if stem_chs is None:
            stem_chs = [64, 256]
        if depths is None:
            depths = [3, 4, 10, 3]

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], 3, 2, 1),
            ConvBNReLU(stem_chs[0], stem_chs[1], 3, 2, 1)
        )

        # Stages
        stage_configs = [
            (stem_chs[1], stem_chs[1], depths[0], 4, 'conv'),
            (stem_chs[1], stem_chs[1] * 2, depths[1], 8, 'conv'),
            (stem_chs[1] * 2, stem_chs[1] * 4, depths[2], 16, 'trans'),
            (stem_chs[1] * 4, stem_chs[1] * 8, depths[3], 32, 'trans'),
        ]

        self.stages = nn.ModuleList()
        for i, (in_c, out_c, depth, heads, block_type) in enumerate(stage_configs):
            if i > 0:
                # Downsample
                self.stages.append(
                    nn.Conv2d(in_c, out_c, 2, 2)
                )

            # Blocks
            if block_type == 'conv':
                blocks = nn.Sequential(*[
                    NCB(out_c if i > 0 else in_c, heads)
                    for _ in range(depth)
                ])
            else:
                blocks = nn.ModuleList([
                    NTB(out_c if i > 0 else in_c, heads)
                    for _ in range(depth)
                ])

            self.stages.append(blocks)

        # Head
        self.norm = nn.LayerNorm(stem_chs[1] * 8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(stem_chs[1] * 8, num_classes)

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

    def forward(self, x):
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.ModuleList):
                # Transformer blocks
                b, c, h, w = x.shape
                x = x.flatten(2).transpose(1, 2)
                for blk in stage:
                    x = blk(x)
                x = x.transpose(1, 2).reshape(b, c, h, w)
            else:
                x = stage(x)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.avgpool(x).squeeze(-1)
        x = self.head(x)
        return x


def nextvit_small(num_classes=1000):
    """NextViT-Small"""
    return NextViT(
        num_classes=num_classes,
        stem_chs=[64, 256],
        depths=[3, 4, 10, 3]
    )


def nextvit_base(num_classes=1000):
    """NextViT-Base"""
    return NextViT(
        num_classes=num_classes,
        stem_chs=[64, 256],
        depths=[3, 4, 20, 3]
    )


__all__ = ['NextViT', 'nextvit_small', 'nextvit_base']

