"""Additional Advanced CNN Backbones

ResNeSt, Res2Net, RepVGG, HardNet, DLA, DPN, CSPNet, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..registry import register_backbone


# ============ ResNeSt ============
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, radix=2, reduction_factor=4):
        super(SplAtConv2d, self).__init__()
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                             groups=groups * radix, bias=bias)
        self.bn0 = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x

        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


@register_backbone('resnest50')
class ResNeSt50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(SplAtConv2d(self.inplanes, planes * 4, 3, stride, 1, radix=2))
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(SplAtConv2d(self.inplanes, planes * 4, 3, 1, 1, radix=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============ Res2Net ============
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


@register_backbone('res2net50')
class Res2Net50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = 26
        self.scale = 4

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )

        layers = []
        layers.append(Bottle2neck(self.inplanes, planes, stride, downsample, self.baseWidth, self.scale))
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottle2neck(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ============ RepVGG ============
class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.nonlinearity = nn.ReLU()

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


@register_backbone('repvgg_a0')
class RepVGG_A0(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()
        self.in_planes = min(64, int(64 * 0.75))

        self.stage0 = RepVGGBlock(3, self.in_planes, stride=2)
        self.stage1 = self._make_stage(int(64 * 0.75), 1, stride=2)
        self.stage2 = self._make_stage(int(128 * 0.75), 2, stride=2)
        self.stage3 = self._make_stage(int(256 * 0.75), 4, stride=2)
        self.stage4 = self._make_stage(int(512 * 0.75), 1, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * 0.75), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward_features(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


# ============ HardNet ============
class HarDBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0

        for i in range(n_layers):
            outch, inch, link = self._get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            layers_.append(nn.Sequential(
                nn.BatchNorm2d(inch),
                nn.ReLU(inplace=True),
                nn.Conv2d(inch, outch, 3, padding=1, bias=False)
            ))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.layers = nn.ModuleList(layers_)

    def _get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self._get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


@register_backbone('hardnet68')
class HarDNet68(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()

        first_ch = [32, 64]
        ch_list = [128, 256, 320, 640, 1024]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        self.base = nn.ModuleList([])
        self.base.append(nn.Sequential(
            nn.Conv2d(3, first_ch[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_ch[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(first_ch[0], first_ch[1], 3, 1, 1, bias=False)
        ))

        ch = first_ch[1]
        for i in range(5):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.out_channels
            self.base.append(blk)

            if i != 4:
                self.base.append(nn.Sequential(
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(2, 2)
                ))

        ch = ch_list[4]
        self.base.append(nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        ))

        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        for layer in self.base:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return x


# ============ CSPNet ============
@register_backbone('cspdarknet53')
class CSPDarkNet53(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.stages = nn.ModuleList([
            self._make_csp_stage(32, 64, 1),
            self._make_csp_stage(64, 128, 2),
            self._make_csp_stage(128, 256, 8),
            self._make_csp_stage(256, 512, 8),
            self._make_csp_stage(512, 1024, 4)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_csp_stage(self, in_ch, out_ch, num_blocks):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

