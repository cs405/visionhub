"""More Transformer Backbones: PVT-V2, Twins, CSWin, LeViT, etc."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register_backbone


# ============ PVT-V2 (Pyramid Vision Transformer V2) ============
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PVTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PVTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            DWConv(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp[0](self.norm2(x))
        x = x + self.mlp[2](self.mlp[1](x, H, W))
        x = self.mlp[3](x)
        return x


@register_backbone('pvt_v2_b0')
class PVT_V2_B0(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()

        embed_dims = [32, 64, 160, 256]
        num_heads = [1, 2, 5, 8]
        depths = [2, 2, 2, 2]
        sr_ratios = [8, 4, 2, 1]

        self.patch_embed1 = nn.Conv2d(3, embed_dims[0], 7, 4, 3)
        self.patch_embed2 = nn.Conv2d(embed_dims[0], embed_dims[1], 3, 2, 1)
        self.patch_embed3 = nn.Conv2d(embed_dims[1], embed_dims[2], 3, 2, 1)
        self.patch_embed4 = nn.Conv2d(embed_dims[2], embed_dims[3], 3, 2, 1)

        self.block1 = nn.ModuleList([PVTBlock(embed_dims[0], num_heads[0], sr_ratio=sr_ratios[0]) for _ in range(depths[0])])
        self.block2 = nn.ModuleList([PVTBlock(embed_dims[1], num_heads[1], sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.block3 = nn.ModuleList([PVTBlock(embed_dims[2], num_heads[2], sr_ratio=sr_ratios[2]) for _ in range(depths[2])])
        self.block4 = nn.ModuleList([PVTBlock(embed_dims[3], num_heads[3], sr_ratio=sr_ratios[3]) for _ in range(depths[3])])

        self.norm = nn.LayerNorm(embed_dims[3])
        self.head = nn.Linear(embed_dims[3], num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Stage 1
        x = self.patch_embed1(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 2-4 similar...
        x = self.patch_embed2(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.patch_embed3(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.patch_embed4(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed1(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.patch_embed2(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.patch_embed3(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.patch_embed4(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)
        return x.mean(dim=1)


# ============ Twins ============
@register_backbone('twins_svt_small')
class Twins_SVT_Small(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()

        embed_dims = [64, 128, 256, 512]
        depths = [2, 2, 10, 4]
        num_heads = [2, 4, 8, 16]

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], 4, 4),
            nn.BatchNorm2d(embed_dims[0])
        )

        # Simple transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(len(depths)):
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(nn.Sequential(
                    nn.LayerNorm(embed_dims[i]),
                    nn.Linear(embed_dims[i], embed_dims[i])
                ))
            self.blocks.append(nn.ModuleList(stage_blocks))

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        for stage_blocks in self.blocks:
            for blk in stage_blocks:
                x = x + blk(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        for stage_blocks in self.blocks:
            for blk in stage_blocks:
                x = x + blk(x)

        x = self.norm(x)
        return x.mean(dim=1)


# ============ EfficientNetV2 ============
class MBConvV2(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, use_se=True):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ])

        if use_se:
            layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
                nn.Sigmoid()
            ))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


@register_backbone('efficientnetv2_s')
class EfficientNetV2_S(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()

        # [expand_ratio, channels, layers, stride]
        settings = [
            [1, 24, 2, 1],
            [4, 48, 4, 2],
            [4, 64, 4, 2],
            [4, 128, 6, 2],
            [6, 160, 9, 1],
            [6, 256, 15, 2]
        ]

        self.features = [nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )]

        in_channels = 24
        for t, c, n, s in settings:
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(MBConvV2(in_channels, c, t, stride))
                in_channels = c

        self.features.append(nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU()
        ))

        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============ AlexNet ============
@register_backbone('alexnet')
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============ GoogleNet (Inception V1) ============
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, 1),
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, 1),
            nn.ReLU(True),
            nn.Conv2d(ch3x3red, ch3x3, 3, padding=1),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, 1),
            nn.ReLU(True),
            nn.Conv2d(ch5x5red, ch5x5, 5, padding=2),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


@register_backbone('googlenet')
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

