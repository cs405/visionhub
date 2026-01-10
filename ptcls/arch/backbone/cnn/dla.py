"""DLA (Deep Layer Aggregation) Implementation

Paper: Deep Layer Aggregation
https://arxiv.org/abs/1707.06484
"""

import math
import torch
import torch.nn as nn
from ..registry import register_backbone


class BasicBlock(nn.Module):
    """DLA Basic Block"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """DLA Bottleneck Block"""
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, cardinality=1, base_width=64):
        super().__init__()
        width = int(math.floor(out_channels * (base_width / 64)) * cardinality)
        width = width // self.expansion

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation=dilation,
                              groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    """DLA Root aggregation"""
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):
    """DLA Tree structure"""
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, cardinality=1, base_width=64):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else nn.Identity()
        self.project = nn.Identity()
        self.levels = levels
        self.level_root = level_root
        self.root_dim = root_dim

        kwargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width)

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **kwargs)
            self.tree2 = block(out_channels, out_channels, 1, **kwargs)
            if in_channels != out_channels:
                self.project = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            kwargs.update(dict(root_kernel_size=root_kernel_size, root_residual=root_residual))
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride,
                             root_dim=0, **kwargs)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                             root_dim=root_dim + out_channels, **kwargs)

        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x


class DLA(nn.Module):
    """DLA Network"""
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock,
                 residual_root=False, cardinality=1, base_width=64):
        super().__init__()
        self.channels = channels
        self.cardinality = cardinality
        self.base_width = base_width

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, 1, 3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                          level_root=False, root_residual=residual_root,
                          cardinality=cardinality, base_width=base_width)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                          level_root=True, root_residual=residual_root,
                          cardinality=cardinality, base_width=base_width)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                          level_root=True, root_residual=residual_root,
                          cardinality=cardinality, base_width=base_width)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                          level_root=True, root_residual=residual_root,
                          cardinality=cardinality, base_width=base_width)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

        self._init_weights()

    def _make_conv_level(self, in_channels, out_channels, num_convs, stride=1, dilation=1):
        modules = []
        for i in range(num_convs):
            modules.append(nn.Conv2d(in_channels, out_channels, 3, stride if i == 0 else 1,
                                    dilation, dilation=dilation, bias=False))
            modules.append(nn.BatchNorm2d(out_channels))
            modules.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*modules)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@register_backbone('dla34')
class DLA34(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                        num_classes=num_classes, block=BasicBlock)


@register_backbone('dla46_c')
class DLA46_c(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256],
                        num_classes=num_classes, block=Bottleneck)


@register_backbone('dla46x_c')
class DLA46x_c(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256],
                        num_classes=num_classes, block=Bottleneck, cardinality=32, base_width=4)


@register_backbone('dla60')
class DLA60(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                        num_classes=num_classes, block=Bottleneck)


@register_backbone('dla60x')
class DLA60x(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                        num_classes=num_classes, block=Bottleneck, cardinality=32, base_width=4)


@register_backbone('dla60x_c')
class DLA60x_c(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 2, 3, 1], [16, 32, 64, 64, 128, 256],
                        num_classes=num_classes, block=Bottleneck, cardinality=32, base_width=4)


@register_backbone('dla102')
class DLA102(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                        num_classes=num_classes, block=Bottleneck, residual_root=True)


@register_backbone('dla102x')
class DLA102x(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                        num_classes=num_classes, block=Bottleneck, cardinality=32, base_width=4, residual_root=True)


@register_backbone('dla102x2')
class DLA102x2(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                        num_classes=num_classes, block=Bottleneck, cardinality=64, base_width=4, residual_root=True)


@register_backbone('dla169')
class DLA169(DLA):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                        num_classes=num_classes, block=Bottleneck, residual_root=True)

