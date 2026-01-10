"""HRNet (High-Resolution Network) Implementation

Paper: Deep High-Resolution Representation Learning for Visual Recognition
https://arxiv.org/abs/1908.07919
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import register_backbone


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """HRNet basic block"""
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """HRNet bottleneck block"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 3, stride)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class FuseLayer(nn.Module):
    """Multi-resolution fusion layer"""
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.num_branches = len(in_channels_list)
        self.fuse_layers = nn.ModuleList()

        for i in range(len(out_channels_list)):
            fuse_layer = nn.ModuleList()
            for j in range(len(in_channels_list)):
                if j > i:
                    # Upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[j], out_channels_list[i], 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Downsample
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(in_channels_list[j], out_channels_list[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(out_channels_list[i])
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(in_channels_list[j], in_channels_list[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(in_channels_list[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            self.fuse_layers.append(fuse_layer)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if len(x) == 1:
            return x

        out = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(len(x)):
                if self.fuse_layers[i][j] is not None:
                    y = y + self.fuse_layers[i][j](x[j])
                else:
                    y = y + x[j]
            out.append(self.relu(y))
        return out


class HighResolutionModule(nn.Module):
    """High resolution module"""
    def __init__(self, num_branches, num_channels, num_blocks):
        super().__init__()
        self.num_branches = num_branches

        self.branches = nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(
                nn.Sequential(*[BasicBlock(num_channels[i], num_channels[i])
                              for _ in range(num_blocks[i])])
            )

        self.fuse_layers = FuseLayer(num_channels, num_channels)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x = self.fuse_layers(x)
        return x


class HRNet(nn.Module):
    """HRNet architecture"""
    def __init__(self, width, num_classes=1000):
        super().__init__()

        # Stem
        self.conv1 = ConvBNReLU(3, 64, 3, 2)
        self.conv2 = ConvBNReLU(64, 64, 3, 2)

        # Stage 1 (Bottleneck blocks)
        self.layer1 = self._make_layer(BottleneckBlock, 64, 64, 4)

        # Transition to stage 2
        self.transition1 = self._make_transition_layer([256], [width, width * 2])

        # Stage 2
        self.stage2 = nn.Sequential(*[
            HighResolutionModule(2, [width, width * 2], [4, 4])
            for _ in range(1)
        ])

        # Transition to stage 3
        self.transition2 = self._make_transition_layer([width, width * 2],
                                                        [width, width * 2, width * 4])

        # Stage 3
        self.stage3 = nn.Sequential(*[
            HighResolutionModule(3, [width, width * 2, width * 4], [4, 4, 4])
            for _ in range(4)
        ])

        # Transition to stage 4
        self.transition3 = self._make_transition_layer([width, width * 2, width * 4],
                                                        [width, width * 2, width * 4, width * 8])

        # Stage 4
        self.stage4 = nn.Sequential(*[
            HighResolutionModule(4, [width, width * 2, width * 4, width * 8], [4, 4, 4, 4])
            for _ in range(3)
        ])

        # Classification head
        self.incre_modules = nn.ModuleList([
            ConvBNReLU(width, 128, 3),
            ConvBNReLU(width * 2, 256, 3),
            ConvBNReLU(width * 4, 512, 3),
            ConvBNReLU(width * 8, 1024, 3),
        ])

        self.downsamp_modules = nn.ModuleList([
            ConvBNReLU(128, 256, 3, 2),
            ConvBNReLU(256, 512, 3, 2),
            ConvBNReLU(512, 1024, 3, 2),
        ])

        self.final_layer = ConvBNReLU(1024, 2048, 1)
        self.classifier = nn.Linear(2048, num_classes)

        self._init_weights()

    def _make_layer(self, block, in_channels, out_channels, blocks):
        downsample = None
        if in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, 1, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(in_channels, out_channels, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        num_branches_pre = len(num_channels_pre)
        num_branches_cur = len(num_channels_cur)

        transition_layers = nn.ModuleList()

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_ch = num_channels_pre[-1]
                    out_ch = num_channels_cur[i] if j == i - num_branches_pre else in_ch
                    conv_downsamples.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return transition_layers

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.layer1(x)

        # Stage 2
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # Stage 4
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification head
        y = self.incre_modules[0](y_list[0])
        for i in range(3):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)
        y = F.adaptive_avg_pool2d(y, 1)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)

        return y


@register_backbone('hrnet_w18_c')
class HRNet_W18_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=18, num_classes=num_classes)


@register_backbone('hrnet_w30_c')
class HRNet_W30_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=30, num_classes=num_classes)


@register_backbone('hrnet_w32_c')
class HRNet_W32_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=32, num_classes=num_classes)


@register_backbone('hrnet_w40_c')
class HRNet_W40_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=40, num_classes=num_classes)


@register_backbone('hrnet_w44_c')
class HRNet_W44_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=44, num_classes=num_classes)


@register_backbone('hrnet_w48_c')
class HRNet_W48_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=48, num_classes=num_classes)


@register_backbone('hrnet_w60_c')
class HRNet_W60_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=60, num_classes=num_classes)


@register_backbone('hrnet_w64_c')
class HRNet_W64_C(HRNet):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(width=64, num_classes=num_classes)

