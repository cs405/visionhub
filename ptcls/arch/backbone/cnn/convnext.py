"""ConvNeXt Implementation

Paper: A ConvNet for the 2020s
https://arxiv.org/abs/2201.03545
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last or channels_first"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt architecture"""
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Feature resolution stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                               layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # Global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_backbone('convnext_tiny')
class ConvNeXt_tiny(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


@register_backbone('convnext_small')
class ConvNeXt_small(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])


@register_backbone('convnext_base_224')
class ConvNeXt_base_224(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])


@register_backbone('convnext_base_384')
class ConvNeXt_base_384(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])


@register_backbone('convnext_large_224')
class ConvNeXt_large_224(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])


@register_backbone('convnext_large_384')
class ConvNeXt_large_384(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])

