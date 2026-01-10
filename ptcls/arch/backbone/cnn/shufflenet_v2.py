"""ShuffleNetV2 Implementation

Paper: ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
from ..registry import register_backbone


def channel_shuffle(x, groups):
    """Channel shuffle operation"""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # Transpose
    x = x.transpose(1, 2).contiguous()
    # Flatten
    x = x.view(batch_size, -1, height, width)
    return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                     groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    """ShuffleNetV2 block for stride=1"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert (out_channels % 2 == 0)
        branch_channels = out_channels // 2

        self.branch2 = nn.Sequential(
            ConvBNReLU(branch_channels, branch_channels, 1, 1),
            ConvBNReLU(branch_channels, branch_channels, 3, 1, groups=branch_channels),
            ConvBNReLU(branch_channels, branch_channels, 1, 1),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(nn.Module):
    """ShuffleNetV2 block for stride=2 (downsample)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert (out_channels % 2 == 0)
        branch_channels = out_channels // 2

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, 3, 2, groups=in_channels),
            ConvBNReLU(in_channels, branch_channels, 1, 1),
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, 1, 1),
            ConvBNReLU(branch_channels, branch_channels, 3, 2, groups=branch_channels),
            ConvBNReLU(branch_channels, branch_channels, 1, 1),
        )

    def forward(self, x):
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, use_swish=False):
        super().__init__()

        # Stage configs: [out_channels, num_blocks]
        stage_configs = {
            0.25: ([24, 48, 96, 192], [4, 8, 4]),
            0.33: ([24, 64, 128, 256], [4, 8, 4]),
            0.5: ([24, 48, 96, 192, 1024], [4, 8, 4]),
            1.0: ([24, 116, 232, 464, 1024], [4, 8, 4]),
            1.5: ([24, 176, 352, 704, 1024], [4, 8, 4]),
            2.0: ([24, 244, 488, 976, 2048], [4, 8, 4]),
        }

        if width_mult not in stage_configs:
            raise ValueError(f"Unsupported width_mult: {width_mult}")

        channels, repeats = stage_configs[width_mult]

        # First conv
        input_channel = channels[0]
        self.conv1 = ConvBNReLU(3, input_channel, 3, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build stages
        self.stage2 = self._make_stage(channels[0], channels[1], repeats[0])
        self.stage3 = self._make_stage(channels[1], channels[2], repeats[1])
        self.stage4 = self._make_stage(channels[2], channels[3], repeats[2])

        # Last conv
        if len(channels) == 5:
            self.conv5 = ConvBNReLU(channels[3], channels[4], 1, 1)
            output_channel = channels[4]
        else:
            self.conv5 = nn.Identity()
            output_channel = channels[3]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(InvertedResidualDS(in_channels, out_channels))
        for _ in range(num_blocks - 1):
            layers.append(InvertedResidual(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


@register_backbone('shufflenet_v2_x0_25')
class ShuffleNetV2_x0_25(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=0.25)


@register_backbone('shufflenet_v2_x0_33')
class ShuffleNetV2_x0_33(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=0.33)


@register_backbone('shufflenet_v2_x0_5')
class ShuffleNetV2_x0_5(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=0.5)


@register_backbone('shufflenet_v2_x1_0')
class ShuffleNetV2_x1_0(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=1.0)


@register_backbone('shufflenet_v2_x1_5')
class ShuffleNetV2_x1_5(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=1.5)


@register_backbone('shufflenet_v2_x2_0')
class ShuffleNetV2_x2_0(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=2.0)


@register_backbone('shufflenet_v2_swish')
class ShuffleNetV2_swish(ShuffleNetV2):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__(num_classes, width_mult=1.0, use_swish=True)

