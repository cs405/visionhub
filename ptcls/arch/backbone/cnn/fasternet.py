"""FasterNet: Fast and Accurate Neural Network

Implementation of FasterNet architecture
Paper: Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
"""

import torch
import torch.nn as nn


class PartialConv(nn.Module):
    """Partial Convolution"""
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x_conv, x_untouched = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x_conv = self.conv(x_conv)
        x = torch.cat([x_conv, x_untouched], dim=1)
        return x


class MLPBlock(nn.Module):
    """MLP Block with Partial Conv"""
    def __init__(self, dim, mlp_ratio=2., n_div=4, act_layer=nn.GELU):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.pconv = PartialConv(dim, n_div)

    def forward(self, x):
        x = self.pconv(x) + x
        x = self.mlp(x) + x
        return x


class FasterNetBlock(nn.Module):
    """FasterNet Basic Block"""
    def __init__(self, dim, mlp_ratio=2., n_div=4, drop_path=0.):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mixer = PartialConv(dim, n_div)

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )

        self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding"""
    def __init__(self, in_c=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, patch_size, patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""
    def __init__(self, dim, out_dim):
        super().__init__()
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        return x


class FasterNet(nn.Module):
    """FasterNet模型

    Args:
        in_chans: 输入通道数
        num_classes: 分类数
        embed_dim: embedding维度
        depths: 每个stage的深度
        mlp_ratio: MLP扩展比例
        n_div: partial conv分割数
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=None,
        mlp_ratio=2.,
        n_div=4
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]

        self.num_classes = num_classes
        self.num_stages = len(depths)

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans, embed_dim)

        # Stages
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.Sequential(*[
                FasterNetBlock(dims[i], mlp_ratio, n_div)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)

            # Patch merging (except last stage)
            if i < self.num_stages - 1:
                self.stages.append(PatchMerging(dims[i], dims[i + 1]))

        # Head
        self.norm = nn.BatchNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def fasternet_t0(num_classes=1000):
    """FasterNet-T0"""
    return FasterNet(
        embed_dim=40,
        depths=[1, 2, 8, 2],
        num_classes=num_classes
    )


def fasternet_t1(num_classes=1000):
    """FasterNet-T1"""
    return FasterNet(
        embed_dim=64,
        depths=[1, 2, 8, 2],
        num_classes=num_classes
    )


def fasternet_t2(num_classes=1000):
    """FasterNet-T2"""
    return FasterNet(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_classes=num_classes
    )


def fasternet_s(num_classes=1000):
    """FasterNet-S"""
    return FasterNet(
        embed_dim=128,
        depths=[2, 2, 13, 2],
        num_classes=num_classes
    )


__all__ = [
    'FasterNet',
    'fasternet_t0',
    'fasternet_t1',
    'fasternet_t2',
    'fasternet_s',
]

