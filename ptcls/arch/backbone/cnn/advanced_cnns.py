"""Advanced CNN Architectures

包含:
- DLA (Deep Layer Aggregation)
- DPN (Dual Path Networks)
- PeleeNet (Efficient CNN)
- RedNet (Residual Encoder-Decoder Network)
- WideResNet
- MicroNet (Extremely Lightweight)
- MixNet (Mixed Depthwise Convolutions)
- MobileNeXt
- StarNet
- TinyNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============ DLA (Deep Layer Aggregation) ============
class DLABottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride != 1:
            x = nn.functional.avg_pool2d(x, self.stride, self.stride)
        out += x
        out = self.relu(out)

        return out


class DLA(nn.Module):
    """Deep Layer Aggregation"""
    def __init__(self, levels, channels, num_classes=1000, block=DLABottleneck):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, 1, 3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = self._make_tree_level(block, channels[1], channels[2], levels[2], stride=2)
        self.level3 = self._make_tree_level(block, channels[2], channels[3], levels[3], stride=2)
        self.level4 = self._make_tree_level(block, channels[3], channels[4], levels[4], stride=2)
        self.level5 = self._make_tree_level(block, channels[4], channels[5], levels[5], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[5], num_classes)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, 3, stride if i == 0 else 1,
                         padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def _make_tree_level(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.MaxPool2d(stride, stride=stride)

        layers = []
        layers.append(block(inplanes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(block(planes * block.expansion, planes, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ============ DPN (Dual Path Networks) ============
class DPNBlock(nn.Module):
    def __init__(self, in_channels, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal'):
        super().__init__()
        self.type = _type
        self.num_1x1_c = num_1x1_c

        if _type == 'proj':
            key_stride = 1
            self.has_proj = True
        if _type == 'down':
            key_stride = 2
            self.has_proj = True
        if _type == 'normal':
            key_stride = 1
            self.has_proj = False

        self.c1x1_w = nn.Sequential(
            nn.Conv2d(in_channels, num_1x1_a, 1, bias=False),
            nn.BatchNorm2d(num_1x1_a),
            nn.ReLU(inplace=True),
        )

        self.c3x3 = nn.Sequential(
            nn.Conv2d(num_1x1_a, num_3x3_b, 3, key_stride, 1, groups=G, bias=False),
            nn.BatchNorm2d(num_3x3_b),
            nn.ReLU(inplace=True),
        )

        self.c1x1_c = nn.Sequential(
            nn.Conv2d(num_3x3_b, num_1x1_c + inc, 1, bias=False),
            nn.BatchNorm2d(num_1x1_c + inc),
        )

        if self.has_proj:
            self.c1x1_w_proj = nn.Sequential(
                nn.Conv2d(in_channels, num_1x1_c + inc, 1, key_stride, bias=False),
                nn.BatchNorm2d(num_1x1_c + inc),
            )

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, list) else x
        out = self.c1x1_w(x_in)
        out = self.c3x3(out)
        out = self.c1x1_c(out)

        if self.has_proj:
            proj = self.c1x1_w_proj(x_in)
            dense = proj[:, :self.num_1x1_c, :, :]
            residual = proj[:, self.num_1x1_c:, :, :]
        else:
            x_s = x[0]
            x_d = x[1]
            dense = x_s
            residual = x_d

        out_d = out[:, :self.num_1x1_c, :, :]
        out_r = out[:, self.num_1x1_c:, :, :]

        dense = dense + out_d
        residual = torch.cat([residual, out_r], dim=1)

        return [F.relu(dense), F.relu(residual)]


class DPN(nn.Module):
    """Dual Path Networks"""
    def __init__(self, cfg, num_classes=1000):
        super().__init__()
        in_channels, k_R, G = cfg['in_channels'], cfg['k_R'], cfg['G']
        num_init_features = cfg['num_init_features']

        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 7, 2, 3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(num_init_features, cfg['k_sec'][0], cfg['inc_sec'][0],
                                       cfg['bw'][0], G, cfg['num_blocks'][0], k_R)
        self.layer2 = self._make_layer(cfg['k_sec'][0], cfg['k_sec'][1], cfg['inc_sec'][1],
                                       cfg['bw'][1], G, cfg['num_blocks'][1], k_R)
        self.layer3 = self._make_layer(cfg['k_sec'][1], cfg['k_sec'][2], cfg['inc_sec'][2],
                                       cfg['bw'][2], G, cfg['num_blocks'][2], k_R)
        self.layer4 = self._make_layer(cfg['k_sec'][2], cfg['k_sec'][3], cfg['inc_sec'][3],
                                       cfg['bw'][3], G, cfg['num_blocks'][3], k_R)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg['k_sec'][3], num_classes)

    def _make_layer(self, in_channels, k_sec, inc_sec, bw, G, num_blocks, k_R):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(DPNBlock(in_channels, k_R * bw, bw, k_sec, inc_sec, G, 'proj' if in_channels != k_sec else 'down'))
            else:
                layers.append(DPNBlock(k_sec + (i) * inc_sec, k_R * bw, bw, k_sec, inc_sec, G, 'normal'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.cat(x, dim=1) if isinstance(x, list) else x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ============ WideResNet ============
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """Wide Residual Networks"""
    def __init__(self, depth, widen_factor, dropout_rate, num_classes=1000):
        super().__init__()
        self.in_planes = 16

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


# Factory functions
def dla34(num_classes=1000, pretrained=False, **kwargs):
    """DLA-34"""
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], num_classes=num_classes)
    return model


def dla60(num_classes=1000, pretrained=False, **kwargs):
    """DLA-60"""
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024], num_classes=num_classes)
    return model


def dpn68(num_classes=1000, pretrained=False, **kwargs):
    """DPN-68"""
    cfg = {
        'in_channels': 10,
        'k_R': 128,
        'G': 32,
        'num_init_features': 10,
        'k_sec': [128, 256, 512, 768],
        'inc_sec': [16, 32, 32, 64],
        'bw': [64, 128, 256, 512],
        'num_blocks': [3, 4, 12, 3]
    }
    model = DPN(cfg, num_classes=num_classes)
    return model


def dpn92(num_classes=1000, pretrained=False, **kwargs):
    """DPN-92"""
    cfg = {
        'in_channels': 64,
        'k_R': 96,
        'G': 32,
        'num_init_features': 64,
        'k_sec': [256, 512, 1024, 2048],
        'inc_sec': [16, 32, 24, 128],
        'bw': [256, 512, 1024, 2048],
        'num_blocks': [3, 4, 20, 3]
    }
    model = DPN(cfg, num_classes=num_classes)
    return model


def wide_resnet28_10(num_classes=1000, pretrained=False, **kwargs):
    """Wide-ResNet-28-10"""
    model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)
    return model


def wide_resnet50_2(num_classes=1000, pretrained=False, **kwargs):
    """Wide-ResNet-50-2"""
    model = WideResNet(depth=50, widen_factor=2, dropout_rate=0.3, num_classes=num_classes)
    return model

